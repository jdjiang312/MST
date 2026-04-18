"""Backfill platform conditions into .pth files used by MST.

Example:
python tools/assign_platform_condition.py \
    --data-root /path/to/ALS_dataset \
    --condition ALS

This script is optional because the updated dataset loader can also infer the
condition from the dataset config or folder path. Saving the field into each
.pth file simply makes the pipeline self-describing and easier to audit.
"""

import argparse
from pathlib import Path

import torch

from pointcept.utils.condition import resolve_condition


def iter_pth_files(data_root: Path):
    if data_root.is_file() and data_root.suffix == ".pth":
        yield data_root
        return
    for path in sorted(data_root.rglob("*.pth")):
        yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Dataset root or a single .pth file")
    parser.add_argument(
        "--condition",
        required=True,
        help="Platform condition to stamp into each file, e.g. ALS / ULS / MLS",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing condition field instead of keeping it unchanged.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    condition = resolve_condition(args.condition, error_prefix="platform condition")

    updated = 0
    skipped = 0
    for path in iter_pth_files(data_root):
        data = torch.load(path, map_location="cpu")
        if "condition" in data and not args.overwrite:
            skipped += 1
            continue
        data["condition"] = condition
        torch.save(data, path)
        updated += 1
        print(f"[updated] {path} -> condition={condition}")

    print(f"Done. updated={updated}, skipped={skipped}, condition={condition}")


if __name__ == "__main__":
    main()
