"""Data diagnostic script for the FrontFinder training pipeline.

Loads training and validation datasets from the config YAML, takes N sample
elements, and prints a comprehensive diagnostic report. Designed to catch
data issues that cause CSI=0 flatline (all-zero labels, dtype mismatches,
NaN contamination, missing front classes, etc.).

Usage:
    python scripts/diagnose_data.py -tc configs/1702.yaml --num-samples 5
    python scripts/diagnose_data.py -tc configs/1702.yaml --num-samples 3 --save-snapshots /tmp/diag
    python scripts/diagnose_data.py -tc configs/1702.yaml --fixture-dir tests/fixtures/dryrun_tf_dataset
"""

import argparse
import dataclasses
import os
import sys

import numpy as np
import tensorflow as tf

# Ensure the src/ package is importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fronts.train import TrainConfig, open_config_yaml_as_dataclass


# ── Helpers ──────────────────────────────────────────────────────────────────


def _header(title: str) -> None:
    """Print a section header."""
    rule = "=" * 72
    print(f"\n{rule}\n  {title}\n{rule}")


def _subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n  {title}\n  {'─' * len(title)}")


# ── Statistics collectors ────────────────────────────────────────────────────


def collect_tensor_stats(
    arrays: list[np.ndarray],
    name: str,
    per_channel: bool = True,
) -> dict:
    """Compute summary statistics over a list of numpy arrays.

    Returns a dict with keys: name, shape, dtype, min, max, mean, std,
    nan_count, inf_count, total_elements, and optionally channel_stats.
    """
    stacked = np.stack(arrays)  # (N, *element_shape)
    f32 = stacked.astype(np.float32)

    stats = {
        "name": name,
        "shape": arrays[0].shape,
        "dtype": str(arrays[0].dtype),
        "min": float(np.nanmin(f32)),
        "max": float(np.nanmax(f32)),
        "mean": float(np.nanmean(f32)),
        "std": float(np.nanstd(f32)),
        "nan_count": int(np.count_nonzero(np.isnan(stacked.astype(np.float32)))),
        "inf_count": int(np.count_nonzero(np.isinf(stacked.astype(np.float32)))),
        "total_elements": int(stacked.size),
    }

    if per_channel and stacked.ndim >= 2:
        n_channels = stacked.shape[-1]
        ch_stats = []
        for c in range(n_channels):
            ch = f32[..., c]
            ch_stats.append({
                "idx": c,
                "min": float(np.nanmin(ch)),
                "max": float(np.nanmax(ch)),
                "mean": float(np.nanmean(ch)),
                "std": float(np.nanstd(ch)),
            })
        stats["channel_stats"] = ch_stats

    return stats


def collect_target_diagnostics(
    arrays: list[np.ndarray],
    num_classes: int,
) -> dict:
    """Analyze one-hot validity and class distribution across target samples."""
    stacked = np.stack(arrays).astype(np.float32)  # (N, H, W, C)
    flat = stacked.reshape(-1, num_classes)

    # Class distribution via argmax
    class_ids = np.argmax(flat, axis=-1)
    counts = np.bincount(class_ids, minlength=num_classes)
    total_pixels = len(class_ids)
    fractions = counts / total_pixels

    # One-hot validity: each pixel should sum to 1.0
    row_sums = flat.sum(axis=-1)
    onehot_valid = np.isclose(row_sums, 1.0, atol=1e-3)
    onehot_valid_frac = float(onehot_valid.mean())
    onehot_max_dev = float(np.max(np.abs(row_sums - 1.0)))

    # Unique values
    unique_vals = np.unique(stacked.astype(np.float32))
    if len(unique_vals) > 10:
        unique_vals = unique_vals[:10]  # truncate for display

    # Count all-background samples
    n_all_bg = 0
    for arr in arrays:
        flat_sample = arr.reshape(-1, num_classes).astype(np.float32)
        sample_ids = np.argmax(flat_sample, axis=-1)
        if np.all(sample_ids == 0):
            n_all_bg += 1

    return {
        "class_pixel_counts": counts,
        "class_pixel_fractions": fractions,
        "onehot_valid_fraction": onehot_valid_frac,
        "onehot_max_deviation": onehot_max_dev,
        "unique_values": unique_vals,
        "num_all_background": n_all_bg,
        "num_samples": len(arrays),
        "total_pixels": total_pixels,
    }


def collect_deep_supervision_diagnostics(
    target_tuples: list[tuple[np.ndarray, ...]],
) -> dict:
    """Verify deep supervision tuple structure — all copies should be identical."""
    n_copies = len(target_tuples[0])
    all_identical = True
    for tup in target_tuples:
        for i in range(1, len(tup)):
            if not np.array_equal(tup[0], tup[i]):
                all_identical = False
                break

    return {
        "n_copies": n_copies,
        "all_identical": all_identical,
        "dtype": str(target_tuples[0][0].dtype),
    }


# ── Report printers ─────────────────────────────────────────────────────────

CLASS_NAMES = ["background", "CF", "WF", "SF", "OF", "DL"]


def print_tensor_stats(stats: dict) -> None:
    """Print formatted tensor statistics."""
    _subheader(stats["name"])
    print(f"    Shape:       {stats['shape']}")
    print(f"    Dtype:       {stats['dtype']}")
    print(f"    Range:       [{stats['min']:.6g}, {stats['max']:.6g}]")
    print(f"    Mean:        {stats['mean']:.6g}    Std: {stats['std']:.6g}")
    print(
        f"    NaN:         {stats['nan_count']:,} / {stats['total_elements']:,}    "
        f"Inf: {stats['inf_count']:,} / {stats['total_elements']:,}"
    )

    if "channel_stats" in stats:
        print(f"\n    Per-channel (dim=-1, {len(stats['channel_stats'])} channels):")
        for ch in stats["channel_stats"]:
            print(
                f"      ch[{ch['idx']:2d}]:  "
                f"min={ch['min']:9.4f}  max={ch['max']:9.4f}  "
                f"mean={ch['mean']:9.4f}  std={ch['std']:9.4f}"
            )


def print_target_diagnostics(diag: dict) -> None:
    """Print class distribution and one-hot validity."""
    _subheader("Target class distribution")
    num_classes = len(diag["class_pixel_counts"])
    for c in range(num_classes):
        label = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        count = diag["class_pixel_counts"][c]
        frac = diag["class_pixel_fractions"][c]
        print(f"    Class {c} ({label:>10s}):  {frac:7.3%}  ({count:>12,} px)")

    print()
    print(f"    One-hot validity:   {diag['onehot_valid_fraction']:7.3%} of pixels sum to 1.0")
    print(f"    Max deviation:      {diag['onehot_max_deviation']:.6f}")
    print(f"    Unique values:      {set(diag['unique_values'].round(4))}")
    print(
        f"    All-background:     {diag['num_all_background']} / {diag['num_samples']} "
        f"samples ({diag['num_all_background'] / max(diag['num_samples'], 1):.1%})"
    )


def print_deep_supervision_diagnostics(diag: dict) -> None:
    """Print deep supervision tuple verification."""
    _subheader("Deep supervision")
    print(f"    Number of copies:   {diag['n_copies']}")
    ident = "YES" if diag["all_identical"] else "NO"
    print(f"    All copies identical: {ident}")
    print(f"    Copy dtype:         {diag['dtype']}")


def print_warnings(
    input_stats: dict,
    target_stats: dict,
    target_diag: dict,
    ds_diag: dict | None,
) -> None:
    """Print summary of warnings and red flags."""
    _subheader("Warnings")
    issues = []

    # CRITICAL checks
    if input_stats["std"] == 0.0:
        issues.append(("CRITICAL", "Input tensor is constant (std=0) — model cannot learn"))
    if target_stats["std"] == 0.0:
        issues.append(("CRITICAL", "Target tensor is constant (std=0) — no gradient signal"))
    if input_stats["nan_count"] > 0:
        issues.append(("CRITICAL", f"NaN in inputs: {input_stats['nan_count']:,} values"))
    if target_stats["nan_count"] > 0:
        issues.append(("CRITICAL", f"NaN in targets: {target_stats['nan_count']:,} values"))
    if input_stats["inf_count"] > 0:
        issues.append(("CRITICAL", f"Inf in inputs: {input_stats['inf_count']:,} values"))
    if target_stats["inf_count"] > 0:
        issues.append(("CRITICAL", f"Inf in targets: {target_stats['inf_count']:,} values"))
    if target_diag["onehot_valid_fraction"] < 0.999:
        issues.append((
            "CRITICAL",
            f"One-hot encoding broken: only {target_diag['onehot_valid_fraction']:.3%} "
            f"of pixels sum to 1.0 (max deviation: {target_diag['onehot_max_deviation']:.6f})",
        ))
    if target_diag["num_all_background"] == target_diag["num_samples"]:
        issues.append((
            "CRITICAL",
            "ALL target samples are 100% background — CSI will be 0 by definition",
        ))
    # Check if no non-background class has any pixels
    non_bg_total = target_diag["class_pixel_counts"][1:].sum()
    if non_bg_total == 0:
        issues.append((
            "CRITICAL",
            "No non-background pixels found in any sample — CSI will be 0",
        ))

    # WARNING checks
    if target_stats["dtype"] == "float16":
        issues.append(("WARNING", "Targets are float16 — loss functions cast to float32"))
    bg_frac = target_diag["class_pixel_fractions"][0]
    if bg_frac > 0.995:
        issues.append(("WARNING", f"Extreme class imbalance: {bg_frac:.2%} background"))
    unique_set = set(target_diag["unique_values"].round(4))
    if unique_set - {0.0, 1.0}:
        issues.append((
            "WARNING",
            f"Unexpected target values (expected {{0.0, 1.0}}, got {unique_set})",
        ))
    if ds_diag is not None and not ds_diag["all_identical"]:
        issues.append(("WARNING", "Deep supervision copies are NOT bitwise identical"))

    # OK checks (print if no issue found for that category)
    if input_stats["nan_count"] == 0 and input_stats["inf_count"] == 0:
        issues.append(("OK", "No NaN or Inf in inputs"))
    if target_stats["nan_count"] == 0 and target_stats["inf_count"] == 0:
        issues.append(("OK", "No NaN or Inf in targets"))
    if non_bg_total > 0:
        issues.append(("OK", f"Non-background pixels present ({non_bg_total:,} total)"))
    if target_diag["onehot_valid_fraction"] >= 0.999:
        issues.append(("OK", "One-hot encoding valid"))
    if ds_diag is not None and ds_diag["all_identical"]:
        issues.append(("OK", f"Deep supervision: all {ds_diag['n_copies']} copies identical"))

    # Sort: CRITICAL first, then WARNING, then OK
    severity_order = {"CRITICAL": 0, "WARNING": 1, "OK": 2}
    issues.sort(key=lambda x: severity_order.get(x[0], 3))

    for severity, msg in issues:
        tag = f"[{severity}]"
        print(f"    {tag:12s} {msg}")

    crit_count = sum(1 for s, _ in issues if s == "CRITICAL")
    warn_count = sum(1 for s, _ in issues if s == "WARNING")
    if crit_count:
        print(f"\n    >>> {crit_count} CRITICAL issue(s) found! <<<")
    elif warn_count:
        print(f"\n    {warn_count} warning(s), no critical issues.")
    else:
        print("\n    All checks passed.")


def save_snapshots(
    save_dir: str,
    inputs: list[np.ndarray],
    targets: list[np.ndarray],
) -> None:
    """Save numpy arrays to disk for offline inspection."""
    os.makedirs(save_dir, exist_ok=True)
    for i, (x, y) in enumerate(zip(inputs, targets)):
        np.save(os.path.join(save_dir, f"inputs_sample_{i}.npy"), x)
        np.save(os.path.join(save_dir, f"targets_sample_{i}.npy"), y)
    print(f"    Saved {len(inputs)} snapshot(s) to {save_dir}/")


# ── Main ─────────────────────────────────────────────────────────────────────


def diagnose_split(
    name: str,
    dataset: tf.data.Dataset,
    num_samples: int,
    n_outputs: int,
    save_dir: str | None,
) -> None:
    """Run full diagnostics on one data split (train or val)."""

    _header(f"DIAGNOSING {name} SPLIT")

    # ── Stage 1: Raw unbatched elements ──────────────────────────────
    _header(f"Stage 1: Raw Elements ({name})")
    raw_inputs, raw_targets = [], []
    for x, y in dataset.take(num_samples):
        raw_inputs.append(x.numpy())
        raw_targets.append(y.numpy())

    num_classes = raw_targets[0].shape[-1]
    print(f"\n    Collected {len(raw_inputs)} raw element(s)  "
          f"(num_classes={num_classes})")

    input_stats = collect_tensor_stats(raw_inputs, f"inputs ({name}, raw)")
    target_stats = collect_tensor_stats(raw_targets, f"targets ({name}, raw)")
    target_diag = collect_target_diagnostics(raw_targets, num_classes)

    print_tensor_stats(input_stats)
    print_tensor_stats(target_stats)
    print_target_diagnostics(target_diag)

    # ── Stage 2: Batched ─────────────────────────────────────────────
    _header(f"Stage 2: Batched, batch_size=1 ({name})")
    batched_ds = dataset.batch(1, drop_remainder=True)
    batched_inputs, batched_targets = [], []
    for x, y in batched_ds.take(num_samples):
        batched_inputs.append(x.numpy())
        batched_targets.append(y.numpy())

    if batched_inputs:
        b_input_stats = collect_tensor_stats(
            batched_inputs, f"inputs ({name}, batched)"
        )
        b_target_stats = collect_tensor_stats(
            batched_targets, f"targets ({name}, batched)"
        )
        print_tensor_stats(b_input_stats)
        print_tensor_stats(b_target_stats)

    # ── Stage 3: Deep supervision ────────────────────────────────────
    ds_diag = None
    if n_outputs > 1:
        _header(f"Stage 3: Deep Supervision, n_outputs={n_outputs} ({name})")
        ds_dataset = batched_ds.map(
            lambda x, y: (x, (y,) * n_outputs),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds_targets = []
        for x, y_tuple in ds_dataset.take(num_samples):
            ds_targets.append(tuple(t.numpy() for t in y_tuple))

        if ds_targets:
            ds_diag = collect_deep_supervision_diagnostics(ds_targets)
            print_deep_supervision_diagnostics(ds_diag)

    # ── Stage 4: Cast simulation ─────────────────────────────────────
    _header(f"Stage 4: tf.cast(float32) Simulation ({name})")
    cast_targets = [tf.cast(t, tf.float32).numpy() for t in raw_targets]
    cast_stats = collect_tensor_stats(
        cast_targets, f"targets ({name}, cast to float32)"
    )
    cast_diag = collect_target_diagnostics(cast_targets, num_classes)
    print_tensor_stats(cast_stats)
    print_target_diagnostics(cast_diag)

    # ── Warnings ─────────────────────────────────────────────────────
    _header(f"WARNINGS ({name})")
    print_warnings(input_stats, target_stats, target_diag, ds_diag)

    # ── Save snapshots ───────────────────────────────────────────────
    if save_dir is not None:
        split_dir = os.path.join(save_dir, name.lower())
        save_snapshots(split_dir, raw_inputs, raw_targets)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-tc", "--train-config-path",
        type=str, required=True,
        help="Path to the training config YAML (e.g. configs/1702.yaml).",
    )
    parser.add_argument(
        "--num-samples",
        type=int, default=5,
        help="Number of dataset elements to inspect per split. Default: 5.",
    )
    parser.add_argument(
        "--save-snapshots",
        type=str, default=None,
        help="Directory to save numpy snapshots for offline inspection.",
    )
    parser.add_argument(
        "--fixture-dir",
        type=str, default=None,
        help="Override tf_dataset directory (for local dry-run with fixture data).",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────
    _header("LOADING CONFIG")
    print(f"    Config:       {args.train_config_path}")
    print(f"    Num samples:  {args.num_samples}")
    print(f"    Save dir:     {args.save_snapshots or '(none)'}")
    print(f"    Fixture dir:  {args.fixture_dir or '(none)'}")

    train_config = open_config_yaml_as_dataclass(
        path=args.train_config_path,
        config_class=TrainConfig,
        require=True,
    )

    # Override data directory for local fixture testing
    if args.fixture_dir is not None:
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(
                train_config.data,
                train_years=[2000],
                val_years=[2001],
                test_years=[],
                tf_dataset=dataclasses.replace(
                    train_config.data.tf_dataset,
                    directory=args.fixture_dir,
                ),
            ),
        )

    # ── Build data pipeline (no model needed) ────────────────────────
    _header("BUILDING DATA PIPELINE")
    model_data = train_config.data.build()

    # Derive n_outputs from model config
    deep_sup = getattr(train_config.model, "deep_supervision", False)
    # UNet 3+ with depth=D produces D deep supervision outputs
    n_outputs = getattr(train_config.model, "depth", 4) if deep_sup else 1
    print(f"    deep_supervision: {deep_sup}  →  n_outputs={n_outputs}")

    # ── Diagnose splits ──────────────────────────────────────────────
    if model_data.train_data is not None:
        diagnose_split(
            "TRAIN", model_data.train_data,
            args.num_samples, n_outputs, args.save_snapshots,
        )
    else:
        print("\n    (no training data)")

    if model_data.validation_data is not None:
        diagnose_split(
            "VAL", model_data.validation_data,
            args.num_samples, n_outputs, args.save_snapshots,
        )
    else:
        print("\n    (no validation data)")

    _header("DONE")


if __name__ == "__main__":
    main()
