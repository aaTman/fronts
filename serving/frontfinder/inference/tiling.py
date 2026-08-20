"""Patch-based tiling for CPU-only global-grid inference.

mandelhub has no GPU and a fixed RAM budget, so a single whole-globe forward
pass through a U-Net-style model is avoided. Instead the grid is split into
overlapping patches (each patch's H and W divisible by 16, per the models'
architecture constraint), inference runs patch-by-patch, and results are
stitched back together with linear-ramp blending across the overlap region
so class-probability seams don't appear at patch boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    row_start: int
    row_end: int  # exclusive, in the *padded* grid
    col_start: int
    col_end: int  # exclusive, in the *padded* grid

    @property
    def height(self) -> int:
        return self.row_end - self.row_start

    @property
    def width(self) -> int:
        return self.col_end - self.col_start


def pad_to_multiple(size: int, multiple: int) -> int:
    """Smallest size' >= size such that size' % multiple == 0."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    remainder = size % multiple
    return size if remainder == 0 else size + (multiple - remainder)


def generate_tiles(
    height: int,
    width: int,
    patch_size: int,
    overlap: int,
    multiple: int = 16,
) -> list[Tile]:
    """Cover a `height` x `width` grid with overlapping tiles.

    Each tile is `patch_size` x `patch_size` (except possibly the last row/col
    of tiles, which are still snapped to a multiple of `multiple`). Stride is
    `patch_size - overlap`, so overlap >= multiple is generally recommended to
    give the blend ramp room to work.
    """
    if patch_size % multiple != 0:
        raise ValueError(
            f"patch_size ({patch_size}) must be a multiple of {multiple}"
        )
    if overlap < 0 or overlap >= patch_size:
        raise ValueError(f"overlap ({overlap}) must be in [0, patch_size)")

    stride = patch_size - overlap
    tiles: list[Tile] = []

    row_starts = list(range(0, max(height - patch_size, 0) + 1, stride))
    if not row_starts or row_starts[-1] + patch_size < height:
        row_starts.append(max(height - patch_size, 0))
    row_starts = sorted(set(row_starts))

    col_starts = list(range(0, max(width - patch_size, 0) + 1, stride))
    if not col_starts or col_starts[-1] + patch_size < width:
        col_starts.append(max(width - patch_size, 0))
    col_starts = sorted(set(col_starts))

    for r in row_starts:
        for c in col_starts:
            tiles.append(
                Tile(
                    row_start=r,
                    row_end=r + patch_size,
                    col_start=c,
                    col_end=c + patch_size,
                )
            )
    return tiles


def blend_weight(patch_size: int, overlap: int) -> np.ndarray:
    """2D ramp weight for a patch, 1.0 in the interior, linearly ramping to a
    small nonzero floor at the outer edge of the overlap band, so neighboring
    tiles' contributions blend smoothly instead of hard-cutting.
    """
    if overlap == 0:
        return np.ones((patch_size, patch_size), dtype=np.float32)

    ramp = np.ones(patch_size, dtype=np.float32)
    edge = np.linspace(1.0 / overlap, 1.0, overlap, dtype=np.float32)
    ramp[:overlap] = edge
    ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp)


def stitch(
    tiles: list[Tile],
    predictions: list[np.ndarray],
    out_height: int,
    out_width: int,
    overlap: int,
) -> np.ndarray:
    """Weighted-average stitch of per-tile predictions back onto the full grid.

    `predictions[i]` has shape (patch_size, patch_size, n_classes) and
    corresponds to `tiles[i]`. Returns an (out_height, out_width, n_classes)
    array cropped to the original (unpadded) grid size.
    """
    if len(tiles) != len(predictions):
        raise ValueError("tiles and predictions must be the same length")
    if not tiles:
        raise ValueError("no tiles to stitch")

    patch_size = tiles[0].height
    n_classes = predictions[0].shape[-1]
    padded_h = max(t.row_end for t in tiles)
    padded_w = max(t.col_end for t in tiles)

    accum = np.zeros((padded_h, padded_w, n_classes), dtype=np.float64)
    weight_sum = np.zeros((padded_h, padded_w, 1), dtype=np.float64)
    w = blend_weight(patch_size, overlap)[..., None]

    for tile, pred in zip(tiles, predictions):
        if pred.shape[:2] != (tile.height, tile.width):
            raise ValueError(
                f"prediction shape {pred.shape[:2]} does not match tile "
                f"shape ({tile.height}, {tile.width})"
            )
        accum[tile.row_start:tile.row_end, tile.col_start:tile.col_end, :] += (
            pred.astype(np.float64) * w
        )
        weight_sum[tile.row_start:tile.row_end, tile.col_start:tile.col_end, :] += w

    weight_sum[weight_sum == 0] = 1.0  # guard: shouldn't happen if tiles cover the grid
    stitched = (accum / weight_sum).astype(np.float32)
    return stitched[:out_height, :out_width, :]
