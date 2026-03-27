"""
Long-context fine-tuning script for LLaDA2 using Nemotron-Pretraining-Specialized-v1.1.

Extends train_llada2_bd.py with two additions:
  1. data_type="text"  -> uses process_mdm_text_example (raw pretraining text)
  2. datasets_type="nemotron_streaming" -> uses NemotronStreamingDataset

All other training logic (FSDP2, MoE, diffusion loss, checkpointing) is unchanged.

Launch:
    bash train.sh tasks/train_longctx.py \\
        --config configs/longctx/llada2_mini_longctx_64k.yaml
"""
import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.distributed as dist
import wandb
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import build_dataloader, build_iterative_dataset, build_mapping_dataset
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import get_device_type, get_nccl_backend, get_torch_device, synchronize
from veomni.utils.dist_utils import all_reduce
from veomni.models.registry import ModelRegistry

ModelRegistry.register_modeling_path("models.llada2_moe")

from dataset.data_transform import (
    process_mdm_sft_example,
    process_mdm_text_example,
    process_mdm_tokenized_example,
)
from dataset import build_local_dataset, build_nemotron_streaming_dataset, NEMOTRON_SUBSETS

logger = helper.create_logger(__name__)


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLaDA2ModelArguments(ModelArguments):
    attn_implementation: Optional[Literal["eager", "sdpa", "flex_attention"]] = field(
        default="sdpa",
        metadata={"help": "Attention implementation to use."},
    )


@dataclass
class LLaDA2DataArguments(DataArguments):
    data_type: Literal["conversation", "tokenid", "text"] = field(
        default="text",
        metadata={"help": "Type of training data: 'text' for Nemotron raw text."},
    )
    datasets_type: Literal["mapping", "local", "nemotron_streaming"] = field(
        default="nemotron_streaming",
        metadata={"help": "Dataset backend type."},
    )
    text_keys: str = field(
        default="text",
        metadata={"help": "Column name to read text from."},
    )
    noise_range_low: float = field(default=0.3, metadata={"help": "Min mask ratio."})
    noise_range_high: float = field(default=0.8, metadata={"help": "Max mask ratio."})
    # Nemotron-specific
    nemotron_subsets: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Nemotron subset names to use. None = all 5 subsets."},
    )
    min_token_len: int = field(
        default=0,
        metadata={"help": "Minimum estimated token length for Nemotron filter (0=disabled)."},
    )
    max_token_len: int = field(
        default=0,
        metadata={"help": "Maximum estimated token length for Nemotron filter (0=disabled)."},
    )
    shuffle_buffer: int = field(
        default=10_000,
        metadata={"help": "Shuffle buffer size for streaming dataset."},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.noise_range_low > self.noise_range_high:
            raise ValueError(
                f"noise_range_low ({self.noise_range_low}) > noise_range_high ({self.noise_range_high})"
            )
        if self.nemotron_subsets is not None:
            unknown = set(self.nemotron_subsets) - set(NEMOTRON_SUBSETS)
            if unknown:
                raise ValueError(f"Unknown Nemotron subsets: {unknown}. Valid: {NEMOTRON_SUBSETS}")


@dataclass
class LLaDA2TrainingArguments(TrainingArguments):
    beta1: float = field(default=0.9, metadata={"help": "AdamW beta1."})
    beta2: float = field(default=0.999, metadata={"help": "AdamW beta2."})
    block_diffusion_mode: bool = field(
        default=False,
        metadata={"help": "Use block-diffusion attention mask. Keep False for context extension."},
    )
    block_size: int = field(default=32, metadata={"help": "Block size for block diffusion."})
    same_token_labels: bool = field(
        default=False,
        metadata={"help": "No-shift labels (True) vs next-token shift (False)."},
    )


@dataclass
class Arguments:
    model: LLaDA2ModelArguments = field(default_factory=LLaDA2ModelArguments)
    data:  LLaDA2DataArguments  = field(default_factory=LLaDA2DataArguments)
    train: LLaDA2TrainingArguments = field(default_factory=LLaDA2TrainingArguments)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _build_transform(args: Arguments, tokenizer):
    noise_range: Tuple[float, float] = (args.data.noise_range_low, args.data.noise_range_high)
    mask_token_id = 156895

    if args.data.data_type == "text":
        return partial(
            process_mdm_text_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=noise_range,
            mask_token_id=mask_token_id,
        )
    elif args.data.data_type == "conversation":
        if not tokenizer.chat_template:
            raise ValueError("No chat template found in the tokenizer.")
        return partial(
            process_mdm_sft_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=noise_range,
            mask_token_id=mask_token_id,
        )
    elif args.data.data_type == "tokenid":
        return partial(
            process_mdm_tokenized_example,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=noise_range,
            mask_token_id=mask_token_id,
        )
    else:
        raise NotImplementedError(f"Unsupported data_type: {args.data.data_type!r}")


def _build_train_dataset(args: Arguments, transform):
    dt = args.data.datasets_type

    if dt == "nemotron_streaming":
        logger.info_rank0("Building NemotronStreamingDataset (streaming, no full download)")
        return build_nemotron_streaming_dataset(
            subsets=args.data.nemotron_subsets,
            min_token_len=args.data.min_token_len,
            max_token_len=args.data.max_token_len,
            transform=transform,
            shuffle_buffer=args.data.shuffle_buffer,
            seed=args.train.seed,
        )
    elif dt == "mapping":
        logger.info_rank0("Building mapping dataset")
        return build_mapping_dataset(args.data.train_path, transform=transform)
    elif dt == "local":
        logger.info_rank0("Building local dataset")
        return build_local_dataset(args.data.train_path, transform=transform)
    elif dt == "iterable":
        logger.info_rank0("Building iterative dataset")
        return build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
    else:
        raise NotImplementedError(f"Unsupported datasets_type: {dt!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    dist.init_process_group(backend=get_nccl_backend())
    args = parse_args(Arguments)

    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()
    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(
        dist_backend=args.train.data_parallel_mode,
        ckpt_manager=args.train.ckpt_manager,
    )

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    transform = _build_transform(args, tokenizer)
    train_dataset = _build_train_dataset(args, transform)

    dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
    if args.data.datasets_type in ("mapping", "local") and dataset_length is not None:
        dataset_length = dataset_length / args.train.data_parallel_size
    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train.train_steps,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
        dyn_bsz_margin=args.train.dyn_bsz_margin,
        dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
    )

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM after model build")

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
    )

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        betas=(args.train.beta1, args.train.beta2),
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},
            )
        model_assets = [model_config, tokenizer]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        state: Dict[str, Any] = {"model": model, "optimizer": optimizer, "extra_state": {}}
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step  = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:
            iter(train_dataloader)
        dist.barrier()
        logger.info_rank0(f"Loaded checkpoint from {args.train.load_checkpoint_path}")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload,
        args.train.enable_gradient_checkpointing,
        args.train.activation_gpu_limit,
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, "
        f"train_steps={args.train.train_steps}, epochs={args.train.num_train_epochs}"
    )

    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)

        for _ in range(start_step, args.train.train_steps):
            global_step += 1
            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished (drop_last={args.data.drop_last})")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0.0
            synchronize()
            start_time = time.time()

            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("source_name", None)

                # Long-context training always uses full attention (no block diffusion)
                micro_batch["attention_mask"] = None

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }

                labels = micro_batch.pop("labels", None)
                # noisy_input_ids is the diffusion input; input_ids is kept as x0 reference
                micro_batch.pop("noisy_input_ids", None)

                with model_fwd_context:
                    logits: torch.Tensor = model(
                        **micro_batch, use_cache=False, output_router_logits=False
                    ).logits

                    if args.train.same_token_labels:
                        unscaled_loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.shape[-1]),
                            labels.view(-1),
                            reduction="none",
                        )
                        loss = unscaled_loss.sum() / (labels != -100).sum() / len(micro_batches)
                    else:
                        shifted_logits = logits[:, :-1, :].contiguous()
                        shifted_labels = labels[:, 1:].contiguous()
                        unscaled_loss = torch.nn.functional.cross_entropy(
                            shifted_logits.view(-1, shifted_logits.shape[-1]),
                            shifted_labels.view(-1),
                            reduction="none",
                        ).view(shifted_logits.shape[0], -1)
                        loss = unscaled_loss.sum() / (shifted_labels != -100).sum() / len(micro_batches)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            if hasattr(model, "clip_grad_norm_"):
                _gn = model.clip_grad_norm_(args.train.max_grad_norm)
                grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            else:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)
                )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            total_loss, grad_norm = all_reduce(
                (total_loss, grad_norm), group=get_parallel_state().fsdp_group
            )
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.3f}, grad_norm: {grad_norm:.3f}, lr: {lr:.2e}"
            )
            data_loader_tqdm.update()

            if args.train.global_rank == 0 and args.train.use_wandb:
                train_metrics.update(
                    {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                )
                wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(
                    args.train.save_checkpoint_path, f"global_step_{global_step}"
                )
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Checkpoint saved at {save_checkpoint_path}")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM after epoch {epoch + 1}")

        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(
                args.train.save_checkpoint_path, f"global_step_{global_step}"
            )
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Epoch checkpoint saved at {save_checkpoint_path}")

    synchronize()
    del optimizer, lr_scheduler
    helper.empty_cache()

    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"HuggingFace weights saved at {hf_weights_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
