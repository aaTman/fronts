"""Train a FrontFinder model with optional Weights and Biases tracking."""

import tensorflow as tf  # ty: ignore[unresolved-import]
import tensorflow.keras  # ty: ignore[unresolved-import]
import logging
import os
import wandb
from wandb.integration import keras as wandb_keras
import dataclasses
import datetime
from typing import Literal, Any, TypeVar, Type
import argparse
import dacite
import yaml
from fronts.model import ModelConfig
from fronts.data.config import DataConfig, PredictConfig, AugmentationConfig

# ---------------------------------------------------------------------------
# Module-level logger — writes to stderr so output appears in Slurm logs even
# when stdout is redirected. Log level can be overridden by setting the
# FRONTS_LOG_LEVEL environment variable, e.g. FRONTS_LOG_LEVEL=DEBUG.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("FRONTS_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fronts.train")

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
        wandb_filepath: path for WandbModelCheckpoint. Must end in `.keras`.
            Defaults to `models/<model_run_name>.keras`.
    """

    project_name: str
    model_run_name: str
    log_frequency: int = 1
    upload_checkpoints: bool = False
    api_key: str = os.environ.get("WANDB_KEY", "")
    wandb_filepath: str | None = None

    def __post_init__(self):
        # Default checkpoint path: models/<model_run_name>.keras
        # WandbModelCheckpoint requires a .keras extension (Keras 3 requirement).
        if self.wandb_filepath is None:
            self.wandb_filepath = f"models/{self.model_run_name}.keras"
        self.login()

    def login(self):
        """Helper method to automatically login to WandB.

        Skipped if no API key is configured (e.g. local dry-run without credentials).
        """
        if self.api_key:
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
    save_freq: Literal["epoch"] | int
    model_checkpoint_path: str | None = None
    csv_logger_path: str | None = None
    patience: int | None = None

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
                filename=self.csv_logger_path, append=True
            )
            callback_list.append(history_logger_callback)
        if self.patience:
            early_stopping_callback = tensorflow.keras.callbacks.EarlyStopping(
                monitor=self.monitor, patience=self.patience, verbose=self.verbose
            )
            callback_list.append(early_stopping_callback)

        return callback_list


def build_augment_fn(aug: AugmentationConfig):
    """Returns a tf.function that randomly flips input/target pairs.

    Wind components are negated when their spatial axis is flipped:
    v-wind is negated on latitude flip, u-wind on longitude flip.
    """
    flip_lat = aug.flip_chance_lat
    flip_lon = aug.flip_chance_lon
    v_idx = aug.v_wind_index
    u_idx = aug.u_wind_index

    @tf.function
    def augment(x, y):
        # Latitude flip
        if flip_lat > 0:
            if tf.random.uniform(()) <= flip_lat:
                x = tf.reverse(x, axis=[0])
                y = tf.reverse(y, axis=[0])
                # Negate v-wind across all levels
                if v_idx is not None:
                    n_vars = tf.shape(x)[-1]
                    sign = tf.where(
                        tf.equal(tf.range(n_vars), v_idx), -1.0, 1.0
                    )
                    x = x * tf.cast(sign, x.dtype)

        # Longitude flip
        if flip_lon > 0:
            if tf.random.uniform(()) <= flip_lon:
                x = tf.reverse(x, axis=[1])
                y = tf.reverse(y, axis=[1])
                # Negate u-wind across all levels
                if u_idx is not None:
                    n_vars = tf.shape(x)[-1]
                    sign = tf.where(
                        tf.equal(tf.range(n_vars), u_idx), -1.0, 1.0
                    )
                    x = x * tf.cast(sign, x.dtype)

        return x, y

    return augment


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
        callbacks: list = None,
        verbose: Literal["auto", 0, 1, 2] = "auto",
        wandb: WandBConfig | None = None,
        repeat: bool = True,
        seed: int = 42,
        num_replicas: int = 1,
        batch_size: int = 1,
        augmentation=None,
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
            num_replicas: number of distribution replicas (GPUs). Defaults to 1.
            batch_size: global batch size. MirroredStrategy shards this across
                replicas (batch_size / num_replicas per GPU). Defaults to 1.
            augmentation: optional AugmentationConfig for runtime data augmentation.

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
        self.callbacks = callbacks or []
        self.seed = seed
        self.num_replicas = num_replicas
        self.batch_size = batch_size
        self.augmentation = augmentation

    def train(self, model: dict) -> None:
        """Triggers a keras training run using model.fit().

        Args:
            model: the complete metadata of configuration of the model
        """

        # Set the seed for fitting the model
        tensorflow.keras.utils.set_random_seed(self.seed)

        # Apply augmentation to training data (before batching).
        unbatched_train = self.data.train_data
        if self.augmentation is not None:
            augment_fn = build_augment_fn(self.augmentation)
            unbatched_train = unbatched_train.map(
                augment_fn, num_parallel_calls=tf.data.AUTOTUNE
            )

        # Batch and optionally repeat.  MirroredStrategy automatically shards
        # the global batch across replicas (batch_size / num_replicas per GPU).
        # drop_remainder=True ensures no partial batch is emitted.
        training_data = unbatched_train.batch(self.batch_size, drop_remainder=True)
        if self.repeat:
            training_data = training_data.repeat()

        validation_data = (
            self.data.validation_data.batch(self.batch_size, drop_remainder=True)
            if self.data.validation_data is not None
            else None
        )

        # Deep supervision produces N outputs; the dataset yields one target.
        # Replicate the target so y_true structure matches y_pred structure.
        n_outputs = len(self.model.outputs)
        if n_outputs > 1:
            training_data = training_data.map(
                lambda x, y: (x, (y,) * n_outputs),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            if validation_data is not None:
                validation_data = validation_data.map(
                    lambda x, y: (x, (y,) * n_outputs),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )

        # Set up the arguments to fit
        fit_args = {
            "x": training_data,
            "validation_data": validation_data,
            "validation_freq": self.validation_frequency,
            "epochs": self.epochs,
            "steps_per_epoch": self.training_steps_per_epoch,
            "validation_steps": self.validation_steps_per_epoch,
            "verbose": self.verbose,
        }

        # Use WandB if exists — WandB callbacks must be built AFTER wandb.init()
        if self.wandb:
            wandb_init = self.wandb.build_init_config(model)
            with wandb.init(**wandb_init) as _:  # ty: ignore[invalid-context-manager]
                fit_args["callbacks"] = self.build_callbacks(self.callbacks)
                self.model.fit(**fit_args)
        else:
            fit_args["callbacks"] = self.build_callbacks(self.callbacks)
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
    wandb: WandBConfig | None
    callbacks: CallbacksConfig
    epochs: int
    training_steps_per_epoch: int
    validation_steps_per_epoch: int | None
    validation_frequency: int
    verbose: Literal["auto", 0, 1, 2]
    repeat: bool
    seed: int
    data: DataConfig | None = None
    predict: PredictConfig | None = None
    distribution: str | None = None
    batch_size: int = 1

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
        log.info("Building callbacks...")
        callbacks = self.callbacks.build()
        log.debug("Callbacks built: %s", [type(c).__name__ for c in callbacks])

        if self.data is not None:
            log.info("Building data pipeline (this may take several minutes)...")
            model_data = self.data.build()
            log.info("Data pipeline ready.")
        else:
            log.warning("No data config provided — trainer will have empty data.")
            model_data = ""

        # Derive input_shape and num_classes from the training dataset element spec.
        # input_shape: set spatial dims to None for variable-size inference.
        # num_classes: last dim of the target tensor (e.g. 6 for 5 front types + background).
        log.info("Deriving input_shape and num_classes from training dataset...")
        train_spec = model_data.train_data.element_spec
        raw_input_shape = list(train_spec[0].shape)
        input_shape = tuple([None, None] + raw_input_shape[2:])
        num_classes = train_spec[1].shape[-1]
        log.info("  input_shape=%s, num_classes=%d", input_shape, num_classes)

        if self.distribution == "mirrored":
            strategy = tf.distribute.MirroredStrategy()
            log.info(
                "Distribution strategy: MirroredStrategy (%d device(s))",
                strategy.num_replicas_in_sync,
            )
        else:
            strategy = tf.distribute.get_strategy()  # default single-device no-op scope
            log.info("Distribution strategy: default (single device)")

        log.info("Building model...")
        with strategy.scope():
            keras_model = self.model.build(input_shape=input_shape, num_classes=num_classes)
        keras_model.summary(print_fn=log.info)
        log.info("Model built and compiled.")

        augmentation = self.data.augmentation if self.data is not None else None

        log.info("Building Trainer...")
        trainer = Trainer(
            model=keras_model,
            data=model_data,
            epochs=self.epochs,
            validation_frequency=self.validation_frequency,
            training_steps_per_epoch=self.training_steps_per_epoch,
            validation_steps_per_epoch=self.validation_steps_per_epoch,
            callbacks=callbacks,
            verbose=self.verbose,
            wandb=self.wandb,
            repeat=self.repeat,
            seed=self.seed,
            num_replicas=strategy.num_replicas_in_sync,
            batch_size=self.batch_size,
            augmentation=augmentation,
        )
        return trainer


def open_config_yaml_as_dataclass(
    path: str, config_class: Type[T], require: bool = False
) -> T | None:
    """Opens a configuration yaml if exists and returns it as the relevant dataclass.

    Args:
        path: the absolute path to the configuration file.
        config_class: the configuration dataclass that the incoming yaml will be
            converted to via dacite.
        require: If True, code will throw an error if the path is not provided.
            Defaults to False.

    Returns either None or the dataclass if path is provided.
    """
    if path:
        with open(file=path) as f:
            config_yaml = yaml.safe_load(f)
        _class_instance = dacite.from_dict(
            data_class=config_class,
            data=config_yaml,
            config=dacite.Config(cast=[tuple, datetime.datetime], check_types=False),
        )
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
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help=(
            "Validate config parsing, data pipeline construction, and model building "
            "without running training or initializing WandB. Uses a small local fixture "
            "dataset instead of the real data directory, so no cluster data or GPU is "
            "needed. Exits 0 on success. Useful for catching bugs before submitting a "
            "SLURM job."
        ),
    )
    parser.add_argument(
        "--fixture_dir",
        type=str,
        default="tests/fixtures/dryrun_tf_dataset",
        help=(
            "Path to the fixture TF dataset used during --dry_run. "
            "Generate fixtures once with: python scripts/make_dryrun_data.py. "
            "Defaults to tests/fixtures/dryrun_tf_dataset."
        ),
    )

    args = parser.parse_args()

    log.info("Loading config from: %s", args.train_config_path)
    train_config = open_config_yaml_as_dataclass(
        path=args.train_config_path, config_class=TrainConfig, require=True
    )
    log.info("Config loaded. epochs=%d, steps_per_epoch=%d",
             train_config.epochs, train_config.training_steps_per_epoch)

    if args.dry_run:
        log.info("=== DRY RUN MODE — skipping WandB init and training ===")
        log.info("  fixture_dir: %s", args.fixture_dir)

        # Patch the loaded config so it uses the tiny local fixture dataset
        # instead of the real cluster data path, and runs for a single step.
        # This lets any production YAML work with --dry_run without a separate
        # dry-run config file.
        dry_data = dataclasses.replace(
            train_config.data,
            train_years=[2000],
            val_years=[2001],
            test_years=[],
            tf_dataset=dataclasses.replace(
                train_config.data.tf_dataset,
                directory=args.fixture_dir,
            ),
        )
        train_config = dataclasses.replace(
            train_config,
            data=dry_data,
            epochs=1,
            training_steps_per_epoch=2,
            validation_steps_per_epoch=1,
            repeat=False,
            wandb=None,  # skip WandB entirely in dry run
        )

        log.info("Building trainer (data pipeline + model)...")
        trainer = train_config.build()  # ty:ignore[possibly-missing-attribute]
        log.info("  train_data:      %s", trainer.data.train_data)
        log.info("  validation_data: %s", trainer.data.validation_data)
        log.info("  test_data:       %s", trainer.data.test_data)
        log.info("  model:           %s", trainer.model)
        log.info("=== DRY RUN complete. No errors. ===")
        raise SystemExit(0)

    log.info("Building trainer (data pipeline + model will be constructed here)...")
    trainer = train_config.build()  # ty:ignore[possibly-missing-attribute]
    log.info("Trainer ready. Starting training run...")

    # Build WandB run metadata — model architecture plus data/training provenance
    # to match the fields logged by the legacy pipeline.
    run_metadata = dataclasses.asdict(train_config.model)
    if train_config.data is not None:
        data = train_config.data
        run_metadata["training_years"] = data.train_years
        run_metadata["validation_years"] = data.val_years
        run_metadata["test_years"] = data.test_years
        run_metadata["steps_per_epoch"] = [
            train_config.training_steps_per_epoch,
            train_config.validation_steps_per_epoch,
        ]
        # Prefer ERA5 config for variable/level metadata; fall back to optional
        # metadata fields on TFDatasetConfig when using pre-built datasets.
        if data.era5 is not None:
            run_metadata["variables"] = data.era5.variables
            run_metadata["pressure_levels"] = [str(lvl) for lvl in data.era5.levels]
        elif data.tf_dataset is not None:
            if data.tf_dataset.variables is not None:
                run_metadata["variables"] = data.tf_dataset.variables
            if data.tf_dataset.levels is not None:
                run_metadata["pressure_levels"] = [
                    str(lvl) for lvl in data.tf_dataset.levels
                ]

    # Trigger training run
    trainer.train(model=run_metadata)
    log.info("Training complete.")
