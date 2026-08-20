import numpy as np
import pytest

from frontfinder.inference.tiling import (
    Tile,
    blend_weight,
    generate_tiles,
    pad_to_multiple,
    stitch,
)


def test_pad_to_multiple_already_aligned():
    assert pad_to_multiple(256, 16) == 256


def test_pad_to_multiple_rounds_up():
    assert pad_to_multiple(721, 16) == 736  # global IFS 0.25deg lat dim
    assert pad_to_multiple(1440, 16) == 1440  # lon dim already aligned


def test_pad_to_multiple_rejects_nonpositive():
    with pytest.raises(ValueError):
        pad_to_multiple(0, 16)


def test_generate_tiles_covers_full_grid_with_no_gaps():
    tiles = generate_tiles(height=736, width=1440, patch_size=256, overlap=32)
    # every point in the grid must be covered by at least one tile
    covered = np.zeros((736, 1440), dtype=bool)
    for t in tiles:
        covered[t.row_start:t.row_end, t.col_start:t.col_end] = True
    assert covered.all()


def test_generate_tiles_all_tiles_are_multiple_of_16():
    tiles = generate_tiles(height=736, width=1440, patch_size=256, overlap=32, multiple=16)
    for t in tiles:
        assert t.height % 16 == 0
        assert t.width % 16 == 0


def test_generate_tiles_rejects_patch_size_not_multiple_of_16():
    with pytest.raises(ValueError):
        generate_tiles(height=736, width=1440, patch_size=250, overlap=32)


def test_generate_tiles_rejects_overlap_too_large():
    with pytest.raises(ValueError):
        generate_tiles(height=736, width=1440, patch_size=256, overlap=256)


def test_generate_tiles_small_grid_single_tile():
    tiles = generate_tiles(height=200, width=200, patch_size=256, overlap=32)
    assert len(tiles) == 1
    assert tiles[0] == Tile(0, 256, 0, 256)


def test_blend_weight_interior_is_one():
    w = blend_weight(patch_size=64, overlap=16)
    assert w[32, 32] == pytest.approx(1.0)


def test_blend_weight_no_overlap_is_uniform_one():
    w = blend_weight(patch_size=64, overlap=0)
    assert np.all(w == 1.0)


def test_stitch_recovers_constant_field_exactly():
    # a constant field, tiled and stitched, should come back out constant --
    # this is the key correctness property for the blend (no seam artifacts).
    height, width, n_classes = 736, 1440, 4
    tiles = generate_tiles(height, width, patch_size=256, overlap=32)
    predictions = [np.full((t.height, t.width, n_classes), 0.5, dtype=np.float32) for t in tiles]
    stitched = stitch(tiles, predictions, out_height=height, out_width=width, overlap=32)
    assert stitched.shape == (height, width, n_classes)
    np.testing.assert_allclose(stitched, 0.5, atol=1e-5)


def test_stitch_crops_to_original_unpadded_size():
    height, width, n_classes = 721, 1440, 4  # unpadded IFS global lat dim
    tiles = generate_tiles(pad_to_multiple(height, 16), width, patch_size=256, overlap=32)
    predictions = [np.zeros((t.height, t.width, n_classes), dtype=np.float32) for t in tiles]
    stitched = stitch(tiles, predictions, out_height=height, out_width=width, overlap=32)
    assert stitched.shape == (height, width, n_classes)


def test_stitch_rejects_mismatched_tile_and_prediction_count():
    tiles = generate_tiles(height=256, width=256, patch_size=256, overlap=32)
    with pytest.raises(ValueError):
        stitch(tiles, predictions=[], out_height=256, out_width=256, overlap=32)
