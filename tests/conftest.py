"""Shared fixtures and TensorFlow/wandb mocking for tests.

Since tests target the config ingestion pipeline (YAML -> dacite -> dataclass -> build),
we mock TensorFlow and wandb so tests can run without GPU or heavy dependencies.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest


def _make_module(name: str) -> types.ModuleType:
    """Create a mock module and register it in sys.modules."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _setup_tensorflow_mocks():
    """Install mock TF modules before any fronts imports."""
    # Top-level tensorflow
    tf = _make_module("tensorflow")
    tf.Tensor = MagicMock
    tf.cast = MagicMock()
    tf.abs = MagicMock()
    tf.function = lambda f: f  # passthrough decorator
    tf.float32 = "float32"
    tf.float16 = "float16"
    tf.is_tensor = MagicMock(return_value=False)
    tf.data = types.ModuleType("tensorflow.data")
    tf.data.Dataset = MagicMock

    # tensorflow.keras
    keras = _make_module("tensorflow.keras")
    tf.keras = keras

    # Sub-modules that get imported
    for sub in [
        "tensorflow.keras.layers",
        "tensorflow.keras.models",
        "tensorflow.keras.callbacks",
        "tensorflow.keras.regularizers",
        "tensorflow.keras.optimizers",
        "tensorflow.keras.constraints",
        "tensorflow.keras.initializers",
        "tensorflow.keras.activations",
        "tensorflow.keras.losses",
        "tensorflow.keras.metrics",
        "tensorflow.keras.utils",
    ]:
        mod = _make_module(sub)
        # Attach as attribute on parent
        parts = sub.split(".")
        parent = sys.modules[".".join(parts[:-1])]
        setattr(parent, parts[-1], mod)

    # Common classes/functions that the code references on these modules
    layers = sys.modules["tensorflow.keras.layers"]
    class _MockLayer:
        """Minimal mock of tf.keras.layers.Layer for custom activation subclasses."""
        def __init__(self, *args, **kwargs):
            pass
        def build(self, input_shape):
            pass
        def call(self, inputs):
            return inputs
    layers.Layer = _MockLayer
    layers.Concatenate = MagicMock
    layers.Input = MagicMock(return_value=MagicMock())
    layers.PReLU = MagicMock
    layers.LeakyReLU = MagicMock
    layers.Activation = MagicMock
    layers.Conv2D = MagicMock
    layers.Conv3D = MagicMock
    layers.BatchNormalization = MagicMock
    layers.MaxPooling2D = MagicMock
    layers.MaxPooling3D = MagicMock
    layers.UpSampling2D = MagicMock
    layers.UpSampling3D = MagicMock
    layers.AveragePooling1D = MagicMock
    layers.AveragePooling2D = MagicMock
    layers.AveragePooling3D = MagicMock

    models = sys.modules["tensorflow.keras.models"]
    models.Model = MagicMock

    callbacks = sys.modules["tensorflow.keras.callbacks"]
    callbacks.Callback = MagicMock
    callbacks.ModelCheckpoint = MagicMock
    callbacks.CSVLogger = MagicMock
    callbacks.EarlyStopping = MagicMock

    regularizers = sys.modules["tensorflow.keras.regularizers"]
    regularizers.Regularizer = MagicMock
    regularizers.L1 = MagicMock
    regularizers.L2 = MagicMock
    regularizers.L1L2 = MagicMock
    regularizers.OrthogonalRegularizer = MagicMock

    optimizers = sys.modules["tensorflow.keras.optimizers"]
    optimizers.Optimizer = MagicMock
    optimizers.Adam = MagicMock

    constraints = sys.modules["tensorflow.keras.constraints"]
    constraints.Constraint = MagicMock
    constraints.MaxNorm = MagicMock
    constraints.MinMaxNorm = MagicMock
    constraints.NonNeg = MagicMock
    constraints.UnitNorm = MagicMock

    initializers = sys.modules["tensorflow.keras.initializers"]
    initializers.Initializer = MagicMock
    initializers.GlorotNormal = MagicMock
    initializers.GlorotUniform = MagicMock
    initializers.HeNormal = MagicMock
    initializers.HeUniform = MagicMock
    initializers.Identity = MagicMock
    initializers.LecunNormal = MagicMock
    initializers.LecunUniform = MagicMock
    initializers.Ones = MagicMock
    initializers.Orthogonal = MagicMock
    initializers.RandomNormal = MagicMock
    initializers.RandomUniform = MagicMock
    initializers.TruncatedNormal = MagicMock
    initializers.VarianceScaling = MagicMock
    initializers.Zeros = MagicMock

    activations = sys.modules["tensorflow.keras.activations"]
    for fn_name in [
        "elu", "exponential", "gelu", "hard_sigmoid", "linear",
        "relu", "selu", "sigmoid", "softmax", "softplus", "softsign",
        "swish", "tanh", "thresholded_relu",
    ]:
        setattr(activations, fn_name, MagicMock(name=f"tf.keras.activations.{fn_name}"))

    losses_mod = sys.modules["tensorflow.keras.losses"]
    losses_mod.Loss = MagicMock

    metrics_mod = sys.modules["tensorflow.keras.metrics"]
    metrics_mod.Metric = MagicMock

    utils_mod = sys.modules["tensorflow.keras.utils"]
    utils_mod.set_random_seed = MagicMock()

    # keras as a top-level attribute
    keras.Activation = MagicMock
    keras.Layer = layers.Layer


def _setup_geospatial_mocks():
    """Install mock modules for heavy geospatial/data dependencies."""
    # shapely — imported by data_utils.py
    shapely = _make_module("shapely")
    shapely_geom = _make_module("shapely.geometry")
    shapely_geom.LineString = MagicMock
    shapely_geom.MultiLineString = MagicMock
    shapely_geom.Point = MagicMock
    shapely.geometry = shapely_geom
    _make_module("shapely.ops")

    # regionmask — may be imported transitively
    _make_module("regionmask")

    # gcsfs — for ERA5 zarr store access
    _make_module("gcsfs")


def _setup_xbatcher_mocks():
    """Install mock xbatcher modules before any fronts imports."""
    xb = _make_module("xbatcher")
    xb.BatchGenerator = MagicMock(return_value=MagicMock())

    loaders = _make_module("xbatcher.loaders")
    xb.loaders = loaders

    loaders_keras = _make_module("xbatcher.loaders.keras")
    loaders_keras.CustomTFDataset = MagicMock(return_value=iter([]))
    loaders.keras = loaders_keras


def _setup_wandb_mocks():
    """Install mock wandb modules."""
    wandb = _make_module("wandb")
    wandb.login = MagicMock()
    wandb.init = MagicMock()

    integration = _make_module("wandb.integration")
    wandb.integration = integration

    wandb_keras = _make_module("wandb.integration.keras")
    wandb_keras.WandbMetricsLogger = MagicMock
    wandb_keras.WandbModelCheckpoint = MagicMock
    integration.keras = wandb_keras


# Install mocks before any test collection triggers fronts imports
_setup_tensorflow_mocks()
_setup_geospatial_mocks()
_setup_xbatcher_mocks()
_setup_wandb_mocks()


@pytest.fixture
def sample_config_dict():
    """A minimal valid config dict matching TrainConfig structure."""
    return {
        "epochs": 10,
        "training_steps_per_epoch": 5,
        "validation_steps_per_epoch": None,
        "validation_frequency": 1,
        "verbose": 1,
        "repeat": True,
        "seed": 42,
        "model": {
            "name": "unet_3plus",
            "batch_normalization": True,
            "num_filters": [16, 32, 64, 128],
            "kernel_size": [5, 5, 5],
            "pool_size": [2, 2, 1],
            "upsample_size": [2, 2, 1],
            "depth": 4,
            "modules_per_node": 2,
            "padding": "same",
            "bias": True,
            "loss": {
                "name": "fractions_skill_score",
                "config": {"mask_size": [3, 3]},
            },
            "metric": {
                "name": "critical_success_index",
                "config": {"class_weights": [0, 1, 1, 1, 1, 1]},
            },
            "optimizer": {
                "name": "Adam",
                "config": {"beta_1": 0.9, "beta_2": 0.999},
            },
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
        "wandb": {
            "project_name": "test_project",
            "model_run_name": "test_run",
        },
        "callbacks": {
            "monitor": "val_loss",
            "verbose": 1,
            "save_best_only": True,
            "save_weights_only": False,
            "save_freq": "epoch",
        },
    }



@pytest.fixture
def sample_yaml_file(tmp_path, sample_config_dict):
    """Write sample_config_dict as a YAML file and return its path."""
    import yaml

    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.dump(sample_config_dict, default_flow_style=False))
    return str(path)
