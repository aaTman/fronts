import dataclasses
from typing import Literal, Any, TypeVar, Generic
import tf.keras.regularizers
import tf.keras.optimizers
import tf.keras.constraints
import tf.keras.initializers
import tf.keras.activations
from fronts.models import activations

T = TypeVar("T")


@dataclasses.dataclass
class BaseConfig(Generic[T]):
    """Base configuration class for building objects from a registry.

    Type parameter T is the type of object this config builds.
    """

    name: str
    config: dict[str, Any]

    # Subclasses must define this
    _registry: dict[str, type] = dataclasses.field(
        default_factory=dict, init=False, repr=False
    )

    def build(self) -> T:
        """Builds the object from the configuration.

        Returns:
            An instance of the registered class.

        Raises:
            ValueError: If the name is not in the registry.
        """
        if self.name not in self._registry:
            raise ValueError(
                f"Unsupported {self.__class__.__name__}: {self.name}. "
                f"Valid options are: {list(self._registry.keys())}"
            )

        cls = self._registry[self.name]
        return cls(**self.config)


@dataclasses.dataclass
class ConvRegularizerConfig(BaseConfig[tf.keras.regularizers.Regularizer]):
    """Regularizer configuration for training a model.

    Attributes:
        name: the string name of the regularizer to use.
        config: a dictionary of keyword arguments to pass to the regularizer constructor.
    """

    name: Literal["L1", "L2", "L1_L2"]

    _registry = {
        "L1": tf.keras.regularizers.L1,
        "L2": tf.keras.regularizers.L2,
        "L1_L2": tf.keras.regularizers.L1L2,
    }


@dataclasses.dataclass
class OptimizerConfig(BaseConfig[tf.keras.optimizers.Optimizer]):
    """Optimizer configuration for training a model.

    Attributes:
        name: the string name of the optimizer to use.
        config: a dictionary of keyword arguments to pass to the optimizer constructor.
    """

    name: Literal["Adam"]

    _registry = {
        "Adam": tf.keras.optimizers.Adam,
    }


@dataclasses.dataclass
class BiasVectorConfig(BaseConfig[tf.keras.constraints]):
    """Constraint configuration for bias vectors in a model.

    Attributes:
        name: the string name of the constraint to use.
        config: a dictionary of keyword arguments to pass to the constraint constructor.
    """

    name: Literal["NonNeg"]

    _registry = {
        "NonNeg": tf.keras.constraints.NonNeg,
    }


@dataclasses.dataclass
class ActivationConfig(BaseConfig[tf.keras.Activation | tf.keras.Layer]):
    """Activation configuration for layers in a model.

    Attributes:
        name: the string name of the activation to use.
        config: a dictionary of keyword arguments to pass to the activation constructor.
    """

    name: Literal[
        "elliott",
        "elu",
        "exponential",
        "gaussian",
        "gcu",
        "gelu",
        "hard_sigmoid",
        "hexpo",
        "isigmoid",
        "leaky_relu",
        "linear",
        "lisht",
        "prelu",
        "psigmoid",
        "ptanh",
        "ptelu",
        "relu",
        "resech",
        "selu",
        "sigmoid",
        "smelu",
        "snake",
        "softmax",
        "softplus",
        "softsign",
        "srs",
        "stanh",
        "swish",
        "tanh",
        "thresholded_relu",
    ]

    _registry = {
        "elliott": activations.Elliott,
        "elu": tf.keras.activations.elu,
        "exponential": tf.keras.activations.exponential,
        "gaussian": activations.Gaussian,
        "gcu": activations.GCU,
        "gelu": tf.keras.activations.gelu,
        "hard_sigmoid": tf.keras.activations.hard_sigmoid,
        "hexpo": activations.Hexpo,
        "isigmoid": activations.ISigmoid,
        "linear": tf.keras.activations.linear,
        "lisht": activations.Lisht,
        "prelu": tf.keras.layers.PReLU,
        "psigmoid": activations.PSigmoid,
        "ptanh": activations.PTanh,
        "ptelu": activations.PTELU,
        "relu": tf.keras.activations.relu,
        "leaky_relu": tf.keras.layers.LeakyReLU,
        "resech": activations.ReSech,
        "selu": tf.keras.activations.selu,
        "sigmoid": tf.keras.activations.sigmoid,
        "smelu": activations.SmeLU,
        "snake": activations.Snake,
        "softmax": tf.keras.activations.softmax,
        "softplus": tf.keras.activations.softplus,
        "softsign": tf.keras.activations.softsign,
        "srs": activations.SRS,
        "stanh": activations.STanh,
        "swish": tf.keras.activations.swish,
        "tanh": tf.keras.activations.tanh,
        "thresholded_relu": tf.keras.activations.thresholded_relu,
    }
