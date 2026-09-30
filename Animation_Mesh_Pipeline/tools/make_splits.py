"""
Write splits.json for Stage 3, split by character (top-level output folder).

Keys use the same sanitized names as Stage 1 output folders, so characters
listed in --test never appear in train/val.

  python tools/make_splits.py \\
    --input_dir .../Mixamo_Data/ModelWithAnimationSelected \\
    --output .../configs/splits_mixamo_v1.json \\
    --test Abe Alex "Castle Guard 02" --val_ratio 0.1 --seed 0

Pass --input_dir a Stage 1 output root instead to split whatever was sampled.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.naming import sanitize_name  # noqa: E402


def list_keys(input_dir: Path) -> list[str]:
    return sorted(
        sanitize_name(p.name)
        for p in input_dir.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test", nargs="*", default=[], help="Held-out characters (raw or sanitized names)")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    keys = list_keys(args.input_dir)
    test = sorted({sanitize_name(name) for name in args.test})
    missing = [name for name in test if name not in keys]
    if missing:
        raise SystemExit(f"--test names not found under {args.input_dir}: {missing}")

    remaining = [k for k in keys if k not in test]
    random.Random(args.seed).shuffle(remaining)
    n_val = round(len(remaining) * args.val_ratio)
    splits = {
        "train": sorted(remaining[n_val:]),
        "val": sorted(remaining[:n_val]),
        "test": test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(splits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"train={len(splits['train'])} val={len(splits['val'])} test={len(test)} -> {args.output}")


if __name__ == "__main__":
    main()
