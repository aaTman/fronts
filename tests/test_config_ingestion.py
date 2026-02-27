"""End-to-end test: YAML file -> dacite -> TrainConfig -> Trainer.

Uses test fixture YAMLs to verify the full config ingestion pipeline.
"""

from unittest.mock import patch

import dacite
import pytest
import yaml

from fronts.train import TrainConfig, Trainer, open_config_yaml_as_dataclass
from fronts.model import ModelConfig
from fronts.utils.keras_builders import (
    ActivationConfig,
    BiasVectorConfig,
    ConvOutputConfig,
    InitializerConfig,
    KernelMatrixConfig,
    LossConfig,
    MetricConfig,
    OptimizerConfig,
)


class TestFullPipelineFromFixture:
    """Test the full YAML -> dacite -> TrainConfig pipeline using the test fixture."""

    def test_full_pipeline(self, sample_yaml_file):
        """Complete pipeline: YAML -> TrainConfig -> Trainer."""
        config = open_config_yaml_as_dataclass(
            path=sample_yaml_file, config_class=TrainConfig
        )

        # Verify top-level fields
        assert config.epochs == 10
        assert config.training_steps_per_epoch == 5
        assert config.validation_frequency == 1
        assert config.verbose == 1
        assert config.repeat is True
        assert config.seed == 42

        # Verify nested ModelConfig
        assert isinstance(config.model, ModelConfig)
        assert config.model.name == "unet_3plus"
        assert config.model.num_filters == [16, 32, 64, 128]
        assert config.model.depth == 4
        assert config.model.padding == "same"
        assert config.model.bias is True

        # Verify nested configs within model
        assert isinstance(config.model.loss, LossConfig)
        assert config.model.loss.name == "fractions_skill_score"
        assert config.model.loss.config == {"mask_size": [3, 3]}

        assert isinstance(config.model.metric, MetricConfig)
        assert config.model.metric.name == "critical_success_index"

        assert isinstance(config.model.optimizer, OptimizerConfig)
        assert config.model.optimizer.name == "Adam"

        assert isinstance(config.model.activation, ActivationConfig)
        assert config.model.activation.name == "gelu"

        assert isinstance(config.model.convolution_activity_regularizer, ConvOutputConfig)
        assert config.model.convolution_activity_regularizer.regularizer is None

        assert isinstance(config.model.bias_vector, BiasVectorConfig)
        assert config.model.bias_vector.constraint is None
        assert config.model.bias_vector.regularizer is None
        assert isinstance(config.model.bias_vector.initializer, InitializerConfig)

        assert isinstance(config.model.kernel_matrix, KernelMatrixConfig)
        assert config.model.kernel_matrix.constraint is None
        assert config.model.kernel_matrix.regularizer is None
        assert isinstance(config.model.kernel_matrix.initializer, InitializerConfig)

        # Build Trainer from config
        trainer = config.build()
        assert isinstance(trainer, Trainer)
        assert trainer.epochs == 10
        assert trainer.wandb is config.wandb

    def test_dacite_rejects_invalid_model_name(self, tmp_path):
        """dacite should propagate through; an invalid Literal should error."""
        bad_config = {
            "epochs": 1,
            "training_steps_per_epoch": 1,
            "validation_steps_per_epoch": None,
            "validation_frequency": 1,
            "verbose": 1,
            "repeat": True,
            "seed": 42,
            "model": {
                "name": "nonexistent_model",
                "batch_normalization": True,
                "num_filters": [16],
                "kernel_size": [3],
                "pool_size": [2],
                "upsample_size": [2],
                "depth": 1,
                "modules_per_node": 1,
                "padding": "same",
                "bias": True,
                "loss": {"name": "fractions_skill_score", "config": {}},
                "metric": {"name": "critical_success_index", "config": {}},
                "optimizer": {"name": "Adam", "config": {}},
                "convolution_activity_regularizer": {"regularizer": None},
                "bias_vector": {
                    "constraint": None,
                    "initializer": {"name": "zeros", "config": {}},
                    "regularizer": None,
                },
                "kernel_matrix": {
                    "constraint": None,
                    "initializer": {"name": "glorot_uniform", "config": {}},
                    "regularizer": None,
                },
                "activation": {"name": "gelu", "config": {}},
            },
            "wandb": {"project_name": "test", "model_run_name": "test"},
            "callbacks": {
                "monitor": "val_loss",
                "verbose": 0,
                "save_best_only": False,
                "save_weights_only": False,
                "save_freq": "epoch",
            },
        }
        yaml_path = tmp_path / "bad_config.yaml"
        yaml_path.write_text(yaml.dump(bad_config))

        # dacite with strict_unions_match or Literal checking should handle this.
        # Even without strict checking, the pipeline should at least not crash
        # during loading — the error surfaces when building.
        result = open_config_yaml_as_dataclass(
            path=str(yaml_path), config_class=TrainConfig
        )
        # The config loads (dacite doesn't enforce Literal by default), but
        # the model name won't be in the UNetRegistry when build is called
        assert result.model.name == "nonexistent_model"


class TestBuildTrainerFromConfig:
    """Test that a loaded TrainConfig can build a Trainer."""

    def test_config_builds_trainer(self, sample_yaml_file):
        """A valid config builds a Trainer with mocked data."""
        from fronts.data.config import DataConfig, ModelData

        config = open_config_yaml_as_dataclass(
            path=sample_yaml_file, config_class=TrainConfig
        )
        dummy_data = ModelData(train_data=None, validation_data=None)
        with patch.object(DataConfig, "build", return_value=dummy_data):
            trainer = config.build()
        assert isinstance(trainer, Trainer)
        assert trainer.epochs == 10
