"""Tests for fronts.utils.keras_builders — BaseConfig, registry subclasses, and
nullable config builders (BiasVectorConfig, KernelMatrixConfig, ConvOutputConfig).
"""

import pytest

from fronts.utils.keras_builders import (
    ActivationConfig,
    BaseConfig,
    BiasVector,
    BiasVectorConfig,
    ConstraintConfig,
    ConvOutput,
    ConvOutputConfig,
    InitializerConfig,
    KernelMatrix,
    KernelMatrixConfig,
    LossConfig,
    MetricConfig,
    OptimizerConfig,
    RegularizerConfig,
)


# ---------------------------------------------------------------------------
# BaseConfig
# ---------------------------------------------------------------------------


class TestBaseConfig:
    def test_build_raises_for_unknown_name(self):
        """BaseConfig.build raises ValueError for an unregistered name."""

        class DummyConfig(BaseConfig):
            @property
            def registry(self):
                return {"known": lambda **kw: "ok"}

        config = DummyConfig(name="unknown", config={})
        with pytest.raises(ValueError, match="Unsupported DummyConfig: unknown"):
            config.build()

    def test_build_calls_registered_callable(self):
        """BaseConfig.build dispatches to the correct registry entry."""

        class DummyConfig(BaseConfig):
            @property
            def registry(self):
                return {"my_thing": lambda x=1: x * 10}

        config = DummyConfig(name="my_thing", config={"x": 5})
        assert config.build() == 50

    def test_build_passes_config_as_kwargs(self):
        """Config dict is unpacked as kwargs to the registered callable."""
        calls = []

        def capture(**kwargs):
            calls.append(kwargs)
            return "captured"

        class DummyConfig(BaseConfig):
            @property
            def registry(self):
                return {"cap": capture}

        config = DummyConfig(name="cap", config={"a": 1, "b": 2})
        result = config.build()
        assert result == "captured"
        assert calls == [{"a": 1, "b": 2}]

    def test_build_with_empty_config(self):
        """Build works with an empty config dict."""

        class DummyConfig(BaseConfig):
            @property
            def registry(self):
                return {"no_args": lambda: "done"}

        config = DummyConfig(name="no_args", config={})
        assert config.build() == "done"


# ---------------------------------------------------------------------------
# Registry subclasses — ensure registries have expected keys
# ---------------------------------------------------------------------------


class TestConstraintConfig:
    @pytest.mark.parametrize("name", ["max_norm", "min_max_norm", "non_neg", "unit_norm"])
    def test_registry_contains_expected_keys(self, name):
        config = ConstraintConfig(name=name, config={})
        assert name in config.registry


class TestInitializerConfig:
    @pytest.mark.parametrize(
        "name",
        [
            "glorot_normal", "glorot_uniform", "he_normal", "he_uniform",
            "identity", "lecun_normal", "lecun_uniform", "ones", "orthogonal",
            "random_normal", "random_uniform", "truncated_normal",
            "variance_scaling", "zeros",
        ],
    )
    def test_registry_contains_expected_keys(self, name):
        config = InitializerConfig(name=name, config={})
        assert name in config.registry

    def test_build_zeros(self):
        """Build a zeros initializer with empty config."""
        config = InitializerConfig(name="zeros", config={})
        result = config.build()
        # Result is a MagicMock (standing in for tf.keras.initializers.Zeros)
        assert result is not None


class TestRegularizerConfig:
    @pytest.mark.parametrize("name", ["l1", "l2", "l1_l2", "orthogonal_regularizer"])
    def test_registry_contains_expected_keys(self, name):
        config = RegularizerConfig(name=name, config={})
        assert name in config.registry


class TestOptimizerConfig:
    def test_registry_contains_adam(self):
        config = OptimizerConfig(name="Adam", config={})
        assert "Adam" in config.registry

    def test_build_adam(self):
        config = OptimizerConfig(name="Adam", config={"beta_1": 0.9, "beta_2": 0.999})
        result = config.build()
        assert result is not None


class TestActivationConfig:
    @pytest.mark.parametrize(
        "name",
        [
            "elliott", "elu", "exponential", "gaussian", "gcu", "gelu",
            "hard_sigmoid", "hexpo", "isigmoid", "leaky_relu", "linear",
            "lisht", "prelu", "psigmoid", "ptanh", "ptelu", "relu", "resech",
            "selu", "sigmoid", "smelu", "snake", "softmax", "softplus",
            "softsign", "srs", "stanh", "swish", "tanh", "thresholded_relu",
        ],
    )
    def test_registry_contains_expected_keys(self, name):
        config = ActivationConfig(name=name, config={})
        assert name in config.registry


class TestLossConfig:
    @pytest.mark.parametrize(
        "name",
        [
            "brier_skill_score", "critical_success_index",
            "fractions_skill_score", "probability_of_detection",
        ],
    )
    def test_registry_contains_expected_keys(self, name):
        config = LossConfig(name=name, config={})
        assert name in config.registry


class TestMetricConfig:
    @pytest.mark.parametrize(
        "name",
        [
            "brier_skill_score", "critical_success_index",
            "fractions_skill_score", "heidke_skill_score",
            "probability_of_detection",
        ],
    )
    def test_registry_contains_expected_keys(self, name):
        """MetricConfig name Literal and registry keys must be in sync."""
        config = MetricConfig(name=name, config={})
        assert name in config.registry


# ---------------------------------------------------------------------------
# ConvOutputConfig — nullable regularizer
# ---------------------------------------------------------------------------


class TestConvOutputConfig:
    def test_build_with_none_regularizer(self):
        """Build succeeds when regularizer is None."""
        config = ConvOutputConfig(regularizer=None)
        result = config.build()
        assert isinstance(result, ConvOutput)
        assert result.activity_regularizer is None

    def test_build_with_regularizer(self):
        """Build delegates to the regularizer when provided."""
        config = ConvOutputConfig(
            regularizer=RegularizerConfig(name="l2", config={"l2": 0.01})
        )
        result = config.build()
        assert isinstance(result, ConvOutput)
        assert result.activity_regularizer is not None


# ---------------------------------------------------------------------------
# BiasVectorConfig — nullable constraint and regularizer
# ---------------------------------------------------------------------------


class TestBiasVectorConfig:
    def test_build_with_all_none(self):
        """Build succeeds when constraint and regularizer are both None."""
        config = BiasVectorConfig(
            constraint=None,
            initializer=InitializerConfig(name="zeros", config={}),
            regularizer=None,
        )
        result = config.build()
        assert isinstance(result, BiasVector)
        assert result.bias_constraint is None
        assert result.bias_regularizer is None
        # initializer should still be built
        assert result.bias_initializer is not None

    def test_build_with_constraint_and_regularizer(self):
        """Build works when both constraint and regularizer are provided."""
        config = BiasVectorConfig(
            constraint=ConstraintConfig(name="non_neg", config={}),
            initializer=InitializerConfig(name="zeros", config={}),
            regularizer=RegularizerConfig(name="l1", config={"l1": 0.01}),
        )
        result = config.build()
        assert isinstance(result, BiasVector)
        assert result.bias_constraint is not None
        assert result.bias_regularizer is not None


# ---------------------------------------------------------------------------
# KernelMatrixConfig — nullable constraint and regularizer
# ---------------------------------------------------------------------------


class TestKernelMatrixConfig:
    def test_build_with_all_none(self):
        """Build succeeds when constraint and regularizer are both None."""
        config = KernelMatrixConfig(
            constraint=None,
            initializer=InitializerConfig(name="glorot_uniform", config={}),
            regularizer=None,
        )
        result = config.build()
        assert isinstance(result, KernelMatrix)
        assert result.kernel_constraint is None
        assert result.kernel_regularizer is None
        assert result.kernel_initializer is not None

    def test_build_with_constraint_and_regularizer(self):
        """Build works when both constraint and regularizer are provided."""
        config = KernelMatrixConfig(
            constraint=ConstraintConfig(name="max_norm", config={}),
            initializer=InitializerConfig(name="he_normal", config={}),
            regularizer=RegularizerConfig(name="l2", config={"l2": 0.01}),
        )
        result = config.build()
        assert isinstance(result, KernelMatrix)
        assert result.kernel_constraint is not None
        assert result.kernel_regularizer is not None
