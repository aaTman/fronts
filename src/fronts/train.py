"""Train a FrontFinder model with optional Weights and Biases tracking."""

import tensorflow.keras  # ty: ignore[unresolved-import]
import os
import wandb
from wandb.integration import keras as wandb_keras
import dataclasses
from typing import Literal, Any, Optional, Union, TypeVar, Type
import argparse
import dacite
import yaml
from fronts.model import ModelConfig

T = TypeVar("T")


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
        api_key: the API key for a WandB account. Defaults to the WANDB_KEY environment
            var.
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
            init_config: the dictionary of all model properties to pass into the WandB
                run instance.

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
        """Returns an instance of a MetricsLogger callback using the log_frequency
        attribute.
        """
        return wandb_keras.WandbMetricsLogger(log_freq=self.log_frequency)

    def build_keras_modelcheckpoint_callback(
        self,
    ) -> tensorflow.keras.callbacks.ModelCheckpoint:
        """Return the ModelCheckpoint WandB callback using the wandb_filepath
        attribute.
        """

        return wandb_keras.WandbModelCheckpoint(self.wandb_filepath)

    def build_all_callbacks(self) -> list[Any]:
        """Returns both ModelCheckpoint and MetricsLogger callbacks."""

        return [
            self.build_keras_modelcheckpoint_callback(),
            self.build_keras_metriclogger_callback(),
        ]


@dataclasses.dataclass
class CallbacksConfig:
    """A configuration for non-Weights and Biases callbacks.

    Certain attributes are shared amongst callbacks, including monitor and verbose.

    Attributes:
        monitor: the metric to monitor.
        verbose: integer determining the amount of logs returned from the callbacks.
        save_best_only: if True, will only save if the model checkpoint has the best
            metrics so far.
        save_weights_only: will only save weights if set to True.
        save_freq: how frequently to save the model checkpoint. If int n, will save
            after n batches.
        model_checkpoint_path: the path where the model will be saved. Defaults to None.
            If None, does not initialize ModelCheckpoint callback.
        csv_logger_path: the path where the csv logger will be saved. Defaults to None.
            If None, does not initialize CSVLogger callback.
        patience: the number of epochs to run with no improvement before stopping early.
            Defaults to None. If None, does not initialize EarlyStopping callback.
    """

    monitor: str
    verbose: int
    save_best_only: bool
    save_weights_only: bool
    save_freq: Union[Literal["epoch"], int]
    model_checkpoint_path: Optional[str] = None
    csv_logger_path: Optional[str] = None
    patience: Optional[int] = None

    def build(self) -> list[tensorflow.keras.callbacks.Callback]:
        # Initialize list
        callback_list = []

        # Only append the list with the callback if the conditions are met, i.e. the
        # key attributes are not None. It is possible to return an empty list
        if self.model_checkpoint_path:
            checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
                filepath=self.model_checkpoint_path,
                monitor=self.monitor,
                verbose=self.verbose,
                save_best_only=self.save_best_only,
                save_weights_only=self.save_weights_only,
                save_freq=self.save_freq,
            )
            callback_list.append(checkpoint_callback)
        if self.csv_logger_path:
            history_logger_callback = tensorflow.keras.callbacks.CSVLogger(
                filepath=self.csv_logger_path, append=True
            )
            callback_list.append(history_logger_callback)
        if self.patience:
            early_stopping_callback = tensorflow.keras.callbacks.EarlyStopping(
                monitor=self.monitor, patience=self.patience, verbose=self.verbose
            )
            callback_list.append(early_stopping_callback)

        return callback_list


class Trainer:
    """Main class to build and trigger model training for FrontFinder."""

    def __init__(
        self,
        model,
        data,
        epochs: int,
        validation_frequency: int,
        training_steps_per_epoch: int,
        validation_steps_per_epoch: int,
        callbacks: list = [],
        verbose: Literal["auto", 0, 1, 2] = "auto",
        wandb: Optional[WandBConfig] = None,
        repeat: bool = True,
        seed: int = 42,
    ) -> None:
        """Initialize the Trainer class and maybe build callbacks.

        Arguments:
            model: the model to use for training.
            data: the ModelDataConfig which holds the prepared train, valid, and test
                data.
            epochs: number of epochs to run to train the model.
            validation_frequency: specifies how many training epochs to run before a new
                validation run is performed, e.g. validation_freq=2 runs validation
                every 2 epochs.
            training_steps_per_epoch: total number of batches of samples to run per
                epoch.
            validation_steps_per_epoch: total number of batches of samples for
                validation. If validation_steps is specified and only part of the
                dataset will be consumed, the evaluation will start from the beginning
                of the dataset at each epoch.
            callbacks: optional list of callbacks (not including WandB callbacks) to use
                when training the model.
            verbose: "auto", 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar,
                2 = one line per epoch. "auto" ~= 1. Defaults to "auto".
            wandb: the Weights and Biases configuration object to use, if exists.
            repeat: whether or not the training dataset will repeat indefinitely.
                Defaults to True. If True, training_steps_per_epoch will determine how
                many batches will run per epoch.
            seed: the seed to use for for all of the backend seeds to allow for
                determinism. Defaults to 42.


        """
        self.model = model
        self.wandb = wandb
        self.data = data
        self.epochs = epochs
        self.validation_frequency = validation_frequency
        self.training_steps_per_epoch = training_steps_per_epoch
        self.validation_steps_per_epoch = validation_steps_per_epoch
        self.verbose = verbose
        self.repeat = repeat
        self.callbacks = self.build_callbacks(callbacks)
        self.seed = seed

    def train(self, model: dict) -> None:
        """Triggers a keras training run using model.fit().

        Args:
            model: the complete metadata of configuration of the model
        """

        # Set the seed for fitting the model
        tensorflow.keras.utils.set_random_seed(self.seed)

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

        # Use WandB if exists
        if self.wandb:
            wandb_init = self.wandb.build_init(model)
            with wandb.init(**wandb_init) as _:  # ty: ignore[invalid-context-manager]
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
        if self.wandb:
            callbacks.extend(self.wandb.build_all_callbacks())
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
            If validation_steps is specified and only part of the dataset will be
            consumed, the evaluation will start from the beginning of the dataset at
            each epoch.
        callbacks: CallbackObject specifying which callbacks to include with training.
        validation_freq: specifies how many training epochs to run before a new
            validation run is performed, e.g. validation_freq=2 runs validation every 2
            epochs.
        verbose: "auto", 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar,
            2 = one line per epoch. "auto" ~= 1. Defaults to "auto".
        repeat: whether or not the training dataset will repeat indefinitely. Defaults
            to True. If True, training_steps_per_epoch will determine how many batches
            will run per epoch.
        seed: the seed to use for for all of the backend seeds to allow for determinism.
            Defaults to 42.
    """

    model: ModelConfig
    # need to build out data config module
    # data: DataConfig
    wandb: WandBConfig
    callbacks: CallbacksConfig
    epochs: int
    training_steps_per_epoch: int
    validation_steps_per_epoch: int | None
    validation_frequency: int
    verbose: Literal["auto", 0, 1, 2]
    repeat: bool
    seed: int

    def build(
        self,
    ) -> Trainer:
        """Builds the Trainer object which can be used to train the model.

        Args:
            model: the model to use for training.
            data: the ModelDataConfig which holds the prepared train, valid, and test
                data.
            wandb: the Weights and Biases configuration object to use, if exists.
            callbacks: optional list of callbacks (not including WandB callbacks) to use
                when training the model.

        Returns a Trainer object that can be used to instantiate a training run.
        """
        callbacks = self.callbacks.build()
        trainer = Trainer(
            # TODO: add + build model and data code to TrainConfig
            model="",
            data="",
            epochs=self.epochs,
            validation_frequency=self.validation_frequency,
            training_steps_per_epoch=self.training_steps_per_epoch,
            validation_steps_per_epoch=self.validation_steps_per_epoch,
            callbacks=callbacks,
            verbose=self.verbose,
            wandb=wandb,
            repeat=self.repeat,
            seed=self.seed,
        )
        return trainer


def open_config_yaml_as_dataclass(
    path: str, config_class: Type[T], require: bool = False
) -> Optional[T]:
    """Opens a configuration yaml if exists and returns it as the relevant dataclass.

    Args:
        path: the absolute path to the configuration file.
        config_class: the configuration dataclass that the incoming yaml will be
            converted to via dacite.
        require: If True, code will throw an error if the path is not provided.
            Defaults to False.

    Returns either None or the dataclass if path is provided.
    """
    if path and not require:
        with open(file=path) as f:
            config_yaml = yaml.safe_load(f)
        _class_instance = dacite.from_dict(data_class=config_class, data=config_yaml)
        return _class_instance
    elif require:
        raise ValueError("Path must be included when require is True.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-tc",
        "--train_config_path",
        type=str,
        required=True,
        help=(
            "Path to the training configuration yaml. This config must include epochs, "
            "training_steps_per_epoch, validation_steps_per_epoch, "
            "validation_frequency, and optionally verbose and repeat. See TrainConfig "
            "for more information on each of these attributes."
        ),
    )

    args = parser.parse_args()

    # Build the training configuration
    train_config = open_config_yaml_as_dataclass(
        path=args.train_config, config_class=TrainConfig, require=True
    )

    # Load the data
    # TODO: build out training data builder
    data = ""

    # Load the tensorflow.keras.Model type
    # TODO: build out model builder
    model = None

    # Build full dictionary of model config options
    # TODO: build out the dictionary builder
    model_config = {}

    # Build trainer
    trainer = train_config.build()  # ty:ignore[possibly-missing-attribute]

    # Trigger training run
    trainer.train(model=model)
