"""Train a FrontFinder model with optional Weights and Biases tracking."""

import tensorflow as tf
import os
import wandb
from wandb.integration import keras as wandb_keras
import dataclasses
from typing import Literal, Any, Optional
import argparse
import dacite
import yaml


@dataclasses.dataclass
class WandBConfig:
    """Configuration dataclass for Weights and Biases.

    Initializing a WandBConfig object will automatically login using api_key
    which defaults to the WANDB_KEY environment variable.

    Attributes:

    project_name: the WandB project where model training data will be stored.
    model_run_name: the name of the model run.
    log_frequency: the rate in epochs of log storage. Defaults to each epoch.
    upload_checkpoints: whether or not to upload the model checkpoints. Defaults to
        False.
    api_key: the API key for a WandB account. Defaults to the WANDB_KEY environment var.
    """

    project_name: str
    model_run_name: str
    log_frequency: int = 1
    upload_checkpoints: bool = False
    api_key: str = os.environ["WANDB_KEY"]
    wandb_filepath: str = "models"

    def __post_init__(self):
        self.login()

    def login(self):
        """Helper method to automatically login to WandB."""
        wandb.login(key=self.api_key)

    def build_init_config(self, init_config: dict) -> dict:
        """Builds the keyword arguments to apply to wandb.init.

        Args:

        init_config: the dictionary of all model properties to pass into the WandB run
            instance.

        Returns a dictionary of project, config, and name, arguments for wandb.init.
        """
        init_config = {
            "project": self.project_name,
            "config": init_config,
            "name": self.model_run_name,
        }
        return init_config

    def build_keras_metriclogger_callback(
        self,
    ) -> wandb_keras.WandbMetricsLogger:
        """Returns an instance of a MetricsLogger callback.

        Returns the WandbMetricsLogger callback using the log_frequency attribute.
        """
        return wandb_keras.WandbMetricsLogger(log_freq=self.log_frequency)

    def build_keras_modelcheckpoint_callback(
        self,
    ) -> tf.keras.callbacks.ModelCheckpoint:
        """Return a list of MetricLogger and ModelCheckpoint WandB callbacks.

        Returns the logger and model checkpoint callbacks in a list.
        """

        return wandb_keras.WandbModelCheckpoint(self.wandb_filepath)

    def build_all_callbacks(self) -> list[Any]:
        """Returns both ModelCheckpoint and MetricsLogger callbacks."""

        return [
            self.build_keras_modelcheckpoint_callback(),
            self.build_keras_metriclogger_callback(),
        ]


@dataclasses.dataclass
class ModelDataConfig:
    """A configuration holding the resulting cleaned and prepared data for training.

    Attributes:

    train_data: data including the inputs and targets for training.
    validation_data: data including the inputs and targets for validation.
    test_data: data including the inputs and targets for testing.
    """

    train_data: tf.data.Dataset
    validation_data: tf.data.Dataset
    test_data: tf.data.Dataset


class Trainer:
    def __init__(
        self,
        model: tf.keras.Model,
        data: ModelDataConfig,
        epochs: int,
        validation_frequency: int,
        training_steps_per_epoch: int,
        validation_steps_per_epoch: int,
        callbacks: list = [],
        verbose: Literal["auto", 0, 1, 2] = "auto",
        wandb_config: Optional[WandBConfig] = None,
        repeat: bool = True,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.wandb_config = wandb_config
        self.data = data
        self.epochs = epochs
        self.validation_frequency = validation_frequency
        self.training_steps_per_epoch = training_steps_per_epoch
        self.validation_steps_per_epoch = validation_steps_per_epoch
        self.verbose = verbose
        self.repeat = repeat
        self.callbacks = self.build_callbacks(callbacks)
        self.seed = seed

    def train(self, model_config: dict) -> None:
        """Triggers a keras training run using model.fit().

        Args:

        model_config: the complete metadata of configuration of the model
        """

        # Set the seed for fitting the model
        tf.keras.utils.set_random_seed(self.seed)

        # If indefinite repeat is enabled, instantiate the bound method
        if self.repeat:
            training_data = self.data.train_data.repeat()
        else:
            training_data = self.data.train_data

        # Set up the arguments to fit
        fit_args = {
            "x": training_data,
            "validation_data": self.data.validation_data,
            "validation_freq": self.validation_frequency,
            "epochs": self.epochs,
            "steps_per_epoch": self.training_steps_per_epoch,
            "validation_steps": self.validation_steps_per_epoch,
            "verbose": self.verbose,
            "callbacks": self.callbacks,
        }
        if self.wandb_config:
            wandb_init_config = self.wandb_config.build_init_config(model_config)
            with wandb.init(**wandb_init_config) as _:
                self.model.fit(**fit_args)

        else:
            self.model.fit(**fit_args)

    def build_callbacks(self, callbacks: list):
        """Combine all callbacks that exist.

        Acts as a passthrough if WandB is not being used, only including the callbacks
        provided when initializing the Trainer.

        Args:

        callbacks: a list of 0 or more callbacks to include when training the model.

        Returns a list of 0 or more callbacks.
        """
        # If WandB is being used, add the callbacks in the dataclass.
        if self.wandb_config:
            callbacks.extend(self.wandb_config.build_all_callbacks())
        return callbacks


@dataclasses.dataclass
class TrainConfig:
    """Model training dataclass.

    Configuration details mostly from
    https://www.tensorflow.org/api_docs/python/tf/keras/Model.

    Attributes:

    epochs: number of epochs to run to train the model.
    training_steps_per_epoch: total number of batches of samples to run per epoch.
    validation_steps_per_epoch: total number of batches of samples for validation.
        If validation_steps is specified and only part of the dataset will be consumed,
        the evaluation will start from the beginning of the dataset at each epoch.
    callbacks: CallbackObject specifying which callbacks to include with training.
    validation_freq: specifies how many training epochs to run before a new validation
        run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
    verbose: "auto", 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one
        line per epoch. "auto" ~= 1. Defaults to "auto".
    seed: the seed to use for for all of the backend seeds to allow for determinism.
        Defaults to 42.
    """

    epochs: int
    training_steps_per_epoch: int
    validation_steps_per_epoch: int
    validation_frequency: int
    verbose: Literal["auto", 0, 1, 2] = "auto"
    repeat: bool = True
    seed: int = 42

    def build(
        self,
        model: tf.keras.Model,
        data: ModelDataConfig,
        wandb_config: Optional[WandBConfig] = None,
        callbacks: list = [],
    ) -> Trainer:
        """Builds the Trainer to later train the model with.

        Args:

        model: the model to use for training
        data: the ModelDataConfig which holds the prepared train, valid, and test data.
        wandb_config: the Weights and Biases configuration object to use, if exists.
        callbacks: optional list of callbacks (not including WandB callbacks) to use
            when training the model.
        """
        trainer = Trainer(
            model=model,
            data=data,
            epochs=self.epochs,
            validation_frequency=self.validation_frequency,
            training_steps_per_epoch=self.training_steps_per_epoch,
            validation_steps_per_epoch=self.validation_steps_per_epoch,
            callbacks=callbacks,
            verbose=self.verbose,
            wandb_config=wandb_config,
            repeat=self.repeat,
            seed=self.seed,
        )
        return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    epochs: int
    training_steps_per_epoch: int
    validation_steps_per_epoch: int
    validation_frequency: int
    callbacks: list = []
    verbose: Literal["auto", 0, 1, 2] = "auto"
    repeat: bool = True
    parser.add_argument(
        "-tc",
        "--train_config",
        type=str,
        required=True,
        help=(
            "Path to the training configuration.This config must include epochs, "
            "training_steps_per_epoch, validation_steps_per_epoch, "
            "validation_frequency, and optionally verbose and repeat. See TrainConfig "
            "for more information on each of these attributes."
        ),
    )
    parser.add_argument(
        "-wc",
        "--wandb_config",
        type=str,
        required=False,
        help=(
            "Path to the Weights and Biases configuration. This config must include "
            "project_name and model_run_name, optionally log_frequency, "
            "upload_checkpoints, api_key, and wandb_filepath. See WandBConfig for more "
            "information on each of these attributes."
        ),
    )
    args = parser.parse_args()

    with open(file=args.train_config) as f:
        train_config = yaml.safe_load(f)

    # WandB configuration is not required
    if args.wandb_config:
        with open(file=args.wandb_config) as f:
            wandb_config = yaml.safe_load(f)
        wandb_config: WandBConfig = dacite.from_dict(
            data_class=WandBConfig, data=args.wandb_config
        )
    # Set wandb_config to None if not included in args
    else:
        wandb_config = None

    train_config: TrainConfig = dacite.from_dict(
        data_class=TrainConfig, data=args.train_config
    )

    # Load the data
    # TODO: build out training data builder
    data = ModelDataConfig(
        train_data=tf.data.Dataset.range(10),
        validation_data=tf.data.Dataset.range(10),
        test_data=tf.data.Dataset.range(10),
    )

    # Load the tf.keras.Model type
    # TODO: build out model builder
    model = None

    # Build full dictionary of model config options
    # TODO: build out the dictionary builder
    model_config = {}

    # Build trainer
    trainer = train_config.build(
        model=model, data=data, wandb_config=wandb_config, callbacks=callbacks
    )

    # Trigger training run
    trainer.train(model_config=model_config)
