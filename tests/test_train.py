"""Tests for fronts.train — config loading, WandB, Callbacks, Trainer, TrainConfig."""

from unittest.mock import MagicMock

import pytest
import yaml

from fronts.train import (
    CallbacksConfig,
    Trainer,
    TrainConfig,
    WandBConfig,
    open_config_yaml_as_dataclass,
)


# ---------------------------------------------------------------------------
# open_config_yaml_as_dataclass
# ---------------------------------------------------------------------------


class TestOpenConfigYamlAsDataclass:
    def test_loads_yaml_into_dataclass(self, sample_yaml_file):
        """YAML file is correctly loaded and converted to TrainConfig."""
        result = open_config_yaml_as_dataclass(
            path=sample_yaml_file, config_class=TrainConfig
        )
        assert result is not None
        assert isinstance(result, TrainConfig)
        assert result.epochs == 10
        assert result.seed == 42

    def test_returns_none_when_path_is_empty(self):
        """Returns None when path is falsy and require is False."""
        result = open_config_yaml_as_dataclass(
            path="", config_class=TrainConfig, require=False
        )
        assert result is None

    def test_returns_none_when_path_is_none(self):
        result = open_config_yaml_as_dataclass(
            path=None, config_class=TrainConfig, require=False
        )
        assert result is None

    def test_raises_when_require_true_and_no_path(self):
        """Raises ValueError when require=True but no path given."""
        with pytest.raises(ValueError, match="Path must be included"):
            open_config_yaml_as_dataclass(
                path="", config_class=TrainConfig, require=True
            )

    def test_loads_when_require_true_and_path_given(self, sample_yaml_file):
        """Successfully loads when require=True and a valid path is provided."""
        result = open_config_yaml_as_dataclass(
            path=sample_yaml_file, config_class=TrainConfig, require=True
        )
        assert result is not None
        assert isinstance(result, TrainConfig)

    def test_nested_model_config_parsed(self, sample_yaml_file):
        """Nested model config is properly deserialized."""
        result = open_config_yaml_as_dataclass(
            path=sample_yaml_file, config_class=TrainConfig
        )
        assert result.model.name == "unet_3plus"
        assert result.model.depth == 4
        assert result.model.num_filters == [16, 32, 64, 128]

    def test_nested_wandb_config_parsed(self, sample_yaml_file):
        """Nested wandb config is properly deserialized."""
        result = open_config_yaml_as_dataclass(
            path=sample_yaml_file, config_class=TrainConfig
        )
        assert result.wandb.project_name == "test_project"
        assert result.wandb.model_run_name == "test_run"


# ---------------------------------------------------------------------------
# WandBConfig
# ---------------------------------------------------------------------------


class TestWandBConfig:
    def test_post_init_calls_login(self):
        """WandBConfig.__post_init__ triggers wandb.login."""
        import wandb

        wandb.login.reset_mock()
        config = WandBConfig(
            project_name="proj", model_run_name="run", api_key="test_key_123"
        )
        wandb.login.assert_called_once_with(key="test_key_123")

    def test_build_init_config_structure(self):
        config = WandBConfig(project_name="proj", model_run_name="run", api_key="k")
        init_config = config.build_init_config({"lr": 0.001})
        assert init_config["project"] == "proj"
        assert init_config["name"] == "run"
        assert init_config["config"] == {"lr": 0.001}

    def test_build_all_callbacks_returns_two(self):
        config = WandBConfig(project_name="proj", model_run_name="run", api_key="k")
        callbacks = config.build_all_callbacks()
        assert len(callbacks) == 2

    def test_default_values(self):
        config = WandBConfig(project_name="proj", model_run_name="run", api_key="k")
        assert config.log_frequency == 1
        assert config.upload_checkpoints is False
        assert config.wandb_filepath == "models"


# ---------------------------------------------------------------------------
# CallbacksConfig
# ---------------------------------------------------------------------------


class TestCallbacksConfig:
    def test_build_empty_when_no_optional_paths(self):
        """Returns empty list when no checkpoint/csv/patience configured."""
        config = CallbacksConfig(
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
        )
        callbacks = config.build()
        assert callbacks == []

    def test_build_includes_checkpoint_when_path_set(self):
        config = CallbacksConfig(
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
            model_checkpoint_path="/tmp/model.h5",
        )
        callbacks = config.build()
        assert len(callbacks) == 1

    def test_build_includes_csv_logger_when_path_set(self):
        config = CallbacksConfig(
            monitor="val_loss",
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            save_freq="epoch",
            csv_logger_path="/tmp/log.csv",
        )
        callbacks = config.build()
        assert len(callbacks) == 1

    def test_build_includes_early_stopping_when_patience_set(self):
        config = CallbacksConfig(
            monitor="val_loss",
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            save_freq="epoch",
            patience=10,
        )
        callbacks = config.build()
        assert len(callbacks) == 1

    def test_build_includes_all_three(self):
        config = CallbacksConfig(
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            save_freq="epoch",
            model_checkpoint_path="/tmp/model.h5",
            csv_logger_path="/tmp/log.csv",
            patience=5,
        )
        callbacks = config.build()
        assert len(callbacks) == 3


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TestTrainer:
    def test_mutable_default_callbacks_isolation(self):
        """Each Trainer gets its own callback list (no shared mutable default)."""
        t1 = Trainer(
            model=MagicMock(),
            data=MagicMock(),
            epochs=1,
            validation_frequency=1,
            training_steps_per_epoch=1,
            validation_steps_per_epoch=1,
        )
        t2 = Trainer(
            model=MagicMock(),
            data=MagicMock(),
            epochs=1,
            validation_frequency=1,
            training_steps_per_epoch=1,
            validation_steps_per_epoch=1,
        )
        t1.callbacks.append("extra")
        assert "extra" not in t2.callbacks

    def test_wandb_callbacks_merged(self):
        """When wandb is provided, its callbacks are merged in."""
        mock_wandb = MagicMock(spec=WandBConfig)
        mock_wandb.build_all_callbacks.return_value = ["wandb_cb1", "wandb_cb2"]
        t = Trainer(
            model=MagicMock(),
            data=MagicMock(),
            epochs=1,
            validation_frequency=1,
            training_steps_per_epoch=1,
            validation_steps_per_epoch=1,
            callbacks=["my_cb"],
            wandb=mock_wandb,
        )
        assert "my_cb" in t.callbacks
        assert "wandb_cb1" in t.callbacks
        assert "wandb_cb2" in t.callbacks

    def test_no_wandb_passthrough(self):
        """Without wandb, callbacks are passed through unchanged."""
        t = Trainer(
            model=MagicMock(),
            data=MagicMock(),
            epochs=1,
            validation_frequency=1,
            training_steps_per_epoch=1,
            validation_steps_per_epoch=1,
            callbacks=["my_cb"],
        )
        assert t.callbacks == ["my_cb"]

    def test_train_calls_fit_without_wandb(self):
        """Trainer.train calls model.fit when wandb is None."""
        mock_model = MagicMock()
        mock_data = MagicMock()
        t = Trainer(
            model=mock_model,
            data=mock_data,
            epochs=5,
            validation_frequency=1,
            training_steps_per_epoch=10,
            validation_steps_per_epoch=3,
        )
        t.train(model={})
        mock_model.fit.assert_called_once()

    def test_train_calls_fit_with_wandb(self):
        """Trainer.train calls model.fit inside wandb.init context when wandb is set."""
        import wandb

        mock_model = MagicMock()
        mock_data = MagicMock()
        mock_wandb_config = MagicMock(spec=WandBConfig)
        mock_wandb_config.build_init_config.return_value = {
            "project": "test",
            "config": {},
            "name": "run",
        }
        mock_wandb_config.build_all_callbacks.return_value = []

        t = Trainer(
            model=mock_model,
            data=mock_data,
            epochs=5,
            validation_frequency=1,
            training_steps_per_epoch=10,
            validation_steps_per_epoch=3,
            wandb=mock_wandb_config,
        )
        t.train(model={"lr": 0.001})
        mock_wandb_config.build_init_config.assert_called_once_with({"lr": 0.001})
        mock_model.fit.assert_called_once()


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------


class TestTrainConfig:
    def test_build_returns_trainer(self, sample_yaml_file):
        """TrainConfig.build() returns a Trainer instance."""
        config = open_config_yaml_as_dataclass(
            path=sample_yaml_file, config_class=TrainConfig
        )
        trainer = config.build()
        assert isinstance(trainer, Trainer)

    def test_build_passes_self_wandb(self, sample_yaml_file):
        """TrainConfig.build() passes self.wandb (not the wandb module)."""
        config = open_config_yaml_as_dataclass(
            path=sample_yaml_file, config_class=TrainConfig
        )
        trainer = config.build()
        assert trainer.wandb is config.wandb

    def test_build_propagates_training_params(self, sample_yaml_file):
        """Training parameters flow through to the Trainer."""
        config = open_config_yaml_as_dataclass(
            path=sample_yaml_file, config_class=TrainConfig
        )
        trainer = config.build()
        assert trainer.epochs == 10
        assert trainer.seed == 42
        assert trainer.repeat is True
        assert trainer.validation_frequency == 1
