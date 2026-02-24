"""Generate a tiny fake TF dataset fixture for local dry-run testing.

Creates two subdirectories under tests/fixtures/dryrun_tf_dataset/:

    2000-1_tf/   -- used as the "train" split
    2001-1_tf/   -- used as the "val" split

Each snapshot contains a small number of random batches whose element shapes
and dtypes match the real on-cluster dataset:

    inputs:  (128, 288, 7, 9)  float16
    targets: (128, 288, 6)     float16

Run once from the repo root before doing a local dry run:

    python scripts/make_dryrun_data.py

The generated files are checked in to .gitignore (see tests/fixtures/.gitignore)
so they are not committed to the repo.
"""

import argparse
import os

import numpy as np
import tensorflow as tf

# Element shapes matching the real dataset (from cluster inspection).
INPUT_SHAPE = (128, 288, 7, 9)
TARGET_SHAPE = (128, 288, 6)
DTYPE = tf.float16

# How many elements (batches) to write per snapshot — small enough to be fast.
NUM_ELEMENTS = 4

FIXTURE_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "tests", "fixtures", "dryrun_tf_dataset"
)

SPLITS = {
    "train": "2000-1_tf",
    "val": "2001-1_tf",
}


def make_snapshot(out_dir: str, num_elements: int, seed: int) -> None:
    rng = np.random.default_rng(seed)

    def _gen():
        for _ in range(num_elements):
            inputs = rng.random(INPUT_SHAPE).astype(np.float16)
            targets = rng.random(TARGET_SHAPE).astype(np.float16)
            yield tf.constant(inputs, dtype=DTYPE), tf.constant(targets, dtype=DTYPE)

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=INPUT_SHAPE, dtype=DTYPE),
            tf.TensorSpec(shape=TARGET_SHAPE, dtype=DTYPE),
        ),
    )
    os.makedirs(out_dir, exist_ok=True)
    ds.save(out_dir)
    print(f"  Saved {num_elements} element(s) → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num_elements",
        type=int,
        default=NUM_ELEMENTS,
        help=f"Number of elements per snapshot (default: {NUM_ELEMENTS})",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=FIXTURE_ROOT,
        help="Root directory to write snapshots into",
    )
    args = parser.parse_args()

    print(f"Writing dry-run fixtures to: {os.path.abspath(args.out_dir)}")
    for i, (split, subdir) in enumerate(SPLITS.items()):
        out = os.path.join(args.out_dir, subdir)
        print(f"  [{i+1}/{len(SPLITS)}] {split}: {subdir}")
        make_snapshot(out, num_elements=args.num_elements, seed=i)

    print("Done. Run the local dry-run with:")
    print(
        "  python -m fronts.train "
        "--train_config_path configs/1702_tf_dryrun.yaml --dry_run"
    )


if __name__ == "__main__":
    main()
