import pytest

from frontfinder.config.manifests import (
    ALL_CLASSES,
    SERVED_CLASSES,
    BEST_LOSS_MANIFEST,
    MODEL_1702_MANIFEST,
    ModelManifest,
    VariableSpec,
    get_manifest,
)


def test_best_loss_channel_count_matches_training_config():
    # sooner_ablations.yaml: model_config.n_channels: 30, 5 variables x 6 levels
    assert BEST_LOSS_MANIFEST.n_channels == 30


def test_model_1702_channel_count():
    # 5 pressure-level vars x 4 levels (20) + 5 single-level vars (5) = 25
    assert MODEL_1702_MANIFEST.n_channels == 25


def test_best_loss_channel_names_are_variable_level_pairs_in_order():
    names = BEST_LOSS_MANIFEST.channel_names()
    assert names[0] == "equivalent_potential_temperature_1000"
    assert names[5] == "equivalent_potential_temperature_300"
    assert names[6] == "u_component_of_wind_1000"
    assert len(names) == 30
    assert len(set(names)) == 30  # no duplicate channels


def test_model_1702_single_level_vars_have_bare_channel_names():
    names = MODEL_1702_MANIFEST.channel_names()
    assert "surface_pressure" in names
    assert "2m_temperature" in names
    # single-level vars never get a level suffix
    assert "surface_pressure_1000" not in names


def test_served_classes_excludes_dryline_and_background():
    assert "dryline" not in SERVED_CLASSES
    assert "background" not in SERVED_CLASSES
    assert set(SERVED_CLASSES) == {"cold", "warm", "stationary", "occluded"}
    assert set(SERVED_CLASSES).issubset(set(ALL_CLASSES))


def test_served_class_indices_align_with_all_classes_order():
    idx = BEST_LOSS_MANIFEST.served_class_indices()
    names_at_idx = [BEST_LOSS_MANIFEST.all_classes[i] for i in idx]
    assert names_at_idx == list(SERVED_CLASSES)


def test_get_manifest_looks_up_by_name():
    assert get_manifest("best_loss") is BEST_LOSS_MANIFEST
    assert get_manifest("model_1702") is MODEL_1702_MANIFEST


def test_get_manifest_unknown_model_raises():
    with pytest.raises(KeyError):
        get_manifest("not_a_real_model")


def test_variable_spec_rejects_empty_levels_tuple():
    with pytest.raises(ValueError):
        VariableSpec("temperature", levels=())


def test_variable_spec_rejects_duplicate_levels():
    with pytest.raises(ValueError):
        VariableSpec("temperature", levels=(1000, 1000))


def test_manifest_rejects_duplicate_variable_names():
    with pytest.raises(ValueError):
        ModelManifest(
            name="dup",
            weights_filename="dup.keras",
            variables=(
                VariableSpec("temperature", levels=(1000,)),
                VariableSpec("temperature", levels=(850,)),
            ),
        )


def test_manifest_rejects_empty_variables():
    with pytest.raises(ValueError):
        ModelManifest(name="empty", weights_filename="empty.keras", variables=())


def test_manifest_rejects_served_class_not_in_all_classes():
    with pytest.raises(ValueError):
        ModelManifest(
            name="bad",
            weights_filename="bad.keras",
            variables=(VariableSpec("temperature", levels=(1000,)),),
            all_classes=("background", "cold"),
            served_classes=("cold", "warm"),
        )
