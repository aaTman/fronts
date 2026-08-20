"""Tiled inference engine: runs a Keras model over a full grid patch-by-patch
via `frontfinder.inference.tiling`, and reduces the softmax output down to the
served front classes per `ModelManifest.served_class_indices()`.

The Keras model itself is accessed through a narrow `Predictor` protocol
(`predict_batch(patches) -> class_probs`) so tests can swap in a fake model
and never need TensorFlow installed or a real `.keras`/`.h5` file on disk.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from frontfinder.config.manifests import ModelManifest
from frontfinder.inference.tiling import generate_tiles, pad_to_multiple, stitch


class Predictor(Protocol):
    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        """patches: (n, patch, patch, n_channels) -> (n, patch, patch, n_classes)"""
        ...


class KerasPredictor:
    """Thin adapter around a loaded `keras.Model`. Not covered by unit tests --
    exercise this in an integration smoke test on the Proxmox VM where
    TensorFlow/Keras and the real weights files are available.
    """

    def __init__(self, model_path: str):
        import keras  # local import: keeps TensorFlow out of the unit-test path

        self._model = keras.models.load_model(model_path, compile=False)

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        output = self._model.predict(patches, verbose=0)
        # deep-supervision models (see sooner_ablations.yaml model_config) may
        # return a list of outputs at different resolutions; the final,
        # full-resolution head is the one used for serving.
        if isinstance(output, (list, tuple)):
            output = output[-1]
        return output


def run_tiled_inference(
    predictor: Predictor,
    input_grid: np.ndarray,
    manifest: ModelManifest,
    patch_size: int = 256,
    overlap: int = 32,
    batch_size: int = 8,
) -> np.ndarray:
    """Run `predictor` over `input_grid` (H, W, n_channels) patch-by-patch and
    return served-class probabilities on the original (unpadded) grid, shape
    (H, W, len(manifest.served_classes)).
    """
    if input_grid.ndim != 3:
        raise ValueError(f"input_grid must be (H, W, C), got shape {input_grid.shape}")
    height, width, n_channels = input_grid.shape
    if n_channels != manifest.n_channels:
        raise ValueError(
            f"input_grid has {n_channels} channels, manifest {manifest.name!r} "
            f"expects {manifest.n_channels}"
        )

    padded_h = pad_to_multiple(height, manifest.patch_multiple)
    padded_w = pad_to_multiple(width, manifest.patch_multiple)
    # patch_size itself must also be a multiple of patch_multiple, and large
    # enough to cover the padded grid in at least one tile.
    padded_h = max(padded_h, patch_size)
    padded_w = max(padded_w, patch_size)

    padded = np.zeros((padded_h, padded_w, n_channels), dtype=input_grid.dtype)
    padded[:height, :width, :] = input_grid

    tiles = generate_tiles(padded_h, padded_w, patch_size, overlap, multiple=manifest.patch_multiple)

    served_idx = manifest.served_class_indices()
    predictions: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[start:start + batch_size]
        batch = np.stack(
            [padded[t.row_start:t.row_end, t.col_start:t.col_end, :] for t in batch_tiles],
            axis=0,
        )
        out = predictor.predict_batch(batch)
        for i in range(out.shape[0]):
            predictions.append(out[i][..., served_idx])

    stitched = stitch(tiles, predictions, out_height=height, out_width=width, overlap=overlap)
    return stitched
