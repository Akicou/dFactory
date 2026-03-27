"""
Patch a LLaDA2-mini config.json to extend the RoPE context window.

Supported methods
-----------------
linear    : Compress all RoPE frequencies by `factor`.  Fast/zero-cost but
            quality degrades beyond 2×. Good baseline.
dynamic   : NTK-aware dynamic scaling. Good zero-shot at 2–4×; still benefits
            from a short fine-tune.
yarn      : Yet Another RoPE extensioN (NTK-by-parts).  Best quality at 2–4×
            with a light fine-tune. Recommended for 64k / 128k.
longrope  : Per-dimension learned scaling factors. Best at 8× (256k) with
            progressive fine-tuning (32k→128k→256k).

Usage
-----
# 64k with YaRN (recommended)
python scripts/extend_rope_context.py \\
    --model_path  ./configs/model_configs/llada2_mini \\
    --output_path ./configs/model_configs/llada2_mini_64k \\
    --target_length 65536 \\
    --method yarn

# 256k with LongRoPE (progressive: first produce 128k checkpoint, then run again)
python scripts/extend_rope_context.py \\
    --model_path  ./configs/model_configs/llada2_mini_128k \\
    --output_path ./configs/model_configs/llada2_mini_256k \\
    --target_length 262144 \\
    --method longrope
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


# ---------------------------------------------------------------------------
# YaRN hyper-params keyed by target context length.
# beta_fast / beta_slow control which RoPE frequency bands are interpolated:
#   - high-freq dims (local syntax)  -> not interpolated
#   - low-freq dims  (long-range)    -> fully interpolated
# Values are from the YaRN paper (Su et al. 2023) scaled for rope_theta=600k.
# ---------------------------------------------------------------------------
_YARN_PRESETS: dict[int, dict] = {
    65_536:  {"factor": 2.0, "beta_fast": 32, "beta_slow": 1},
    131_072: {"factor": 4.0, "beta_fast": 32, "beta_slow": 1},
    262_144: {"factor": 8.0, "beta_fast": 32, "beta_slow": 1},
}

# LLaDA2-mini: partial_rotary_factor=0.5, head_dim=128  ->  rotary_dim=64
# inv_freq has length rotary_dim // 2 = 32
# longrope long_factor / short_factor must have exactly this length.
_LONGROPE_FACTOR_LEN = 32


def _load_config(model_path: str) -> dict:
    config_file = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"config.json not found at {config_file}")
    with open(config_file) as fh:
        return json.load(fh)


def _copy_model_dir(src: str, dst: str) -> None:
    """Copy all files except config.json — we write that separately."""
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name == "config.json":
            continue
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def patch_config(
    model_path: str,
    output_path: str,
    target_length: int,
    method: str,
) -> None:
    cfg = _load_config(model_path)
    original_length: int = cfg["max_position_embeddings"]
    original_theta: float = float(cfg.get("rope_theta", 10_000.0))
    factor = target_length / original_length

    print(f"Original max_position_embeddings : {original_length}")
    print(f"Target  max_position_embeddings  : {target_length}")
    print(f"Scale factor                     : {factor:.2f}x")
    print(f"Method                           : {method}")
    print(f"Original rope_theta              : {original_theta}")

    cfg["max_position_embeddings"] = target_length

    if method == "linear":
        cfg["rope_scaling"] = {
            "rope_type": "linear",
            "factor": factor,
        }

    elif method == "dynamic":
        cfg["rope_scaling"] = {
            "rope_type": "dynamic",
            "factor": factor,
        }

    elif method == "yarn":
        preset = _YARN_PRESETS.get(target_length)
        if preset is None:
            # Fall back to generic params for non-standard lengths
            preset = {"factor": factor, "beta_fast": 32, "beta_slow": 1}
        cfg["rope_scaling"] = {
            "rope_type": "yarn",
            "original_max_position_embeddings": original_length,
            **preset,
        }
        # Scale rope_theta proportionally to avoid high-frequency aliasing
        cfg["rope_theta"] = original_theta * preset["factor"]

    elif method == "longrope":
        cfg["rope_scaling"] = {
            "rope_type": "longrope",
            "factor": factor,
            "original_max_position_embeddings": original_length,
            # Bootstrap with 1.0; fine-tuning learns the optimal per-dim values.
            "long_factor":  [1.0] * _LONGROPE_FACTOR_LEN,
            "short_factor": [1.0] * _LONGROPE_FACTOR_LEN,
        }
        cfg["rope_theta"] = original_theta * factor

    else:
        raise ValueError(f"Unknown method {method!r}")

    print(f"New rope_theta                   : {cfg['rope_theta']}")
    print(f"rope_scaling                     : {cfg['rope_scaling']}")

    if model_path != output_path:
        _copy_model_dir(model_path, output_path)

    out_config = os.path.join(output_path, "config.json")
    with open(out_config, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"\nPatched config written -> {out_config}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend LLaDA2-mini RoPE context window via config patch"
    )
    parser.add_argument("--model_path", required=True,
                        help="Directory containing the source config.json")
    parser.add_argument("--output_path", required=True,
                        help="Directory to write the patched config.json (and copied assets)")
    parser.add_argument(
        "--target_length", type=int, required=True,
        choices=[65_536, 131_072, 262_144],
        help="Target context length: 65536 (64k) | 131072 (128k) | 262144 (256k)",
    )
    parser.add_argument(
        "--method", default="yarn",
        choices=["linear", "dynamic", "yarn", "longrope"],
        help="RoPE extension method (default: yarn)",
    )
    args = parser.parse_args()
    patch_config(args.model_path, args.output_path, args.target_length, args.method)


if __name__ == "__main__":
    main()
