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
    registry: dict[str, type] = dataclasses.field(
        default_factory=dict, init=False, repr=False
    )

    def build(self) -> T:
        """Builds the object from the configuration.

        Returns:
            An instance of the registered class.

        Raises:
            ValueError: If the name is not in the registry.
        """
        if self.name not in self.registry:
            raise ValueError(
                f"Unsupported {self.__class__.__name__}: {self.name}. "
                f"Valid options are: {list(self.registry.keys())}"
            )

        cls = self.registry[self.name]
        return cls(**self.config)


@dataclasses.dataclass
class ConstraintConfig(BaseConfig[tf.keras.constraints.Constraint]):
    """Generic constraint configuration for training a model.

    Attributes:
        name: the string name of the constraint to use.
        config: a dictionary of keyword arguments to pass to the constraint constructor.
        registry: a dictionary mapping string names to constraint classes.
    """

    registry = {
        "max_norm": tf.keras.constraints.MaxNorm,
        "min_max_norm": tf.keras.constraints.MinMaxNorm,
        "non_neg": tf.keras.constraints.NonNeg,
        "unit_norm": tf.keras.constraints.UnitNorm,
    }


@dataclasses.dataclass
class InitializerConfig(BaseConfig[tf.keras.initializers.Initializer]):
    """Initializer configuration for training a model.

    Attributes:
        name: the string name of the initializer to use.
        config: a dictionary of keyword arguments to pass to the initializer constructor.
    """

    registry = {
        "glorot_normal": tf.keras.initializers.GlorotNormal,
        "glorot_uniform": tf.keras.initializers.GlorotUniform,
        "he_normal": tf.keras.initializers.HeNormal,
        "he_uniform": tf.keras.initializers.HeUniform,
        "identity": tf.keras.initializers.Identity,
        "lecun_normal": tf.keras.initializers.LecunNormal,
        "lecun_uniform": tf.keras.initializers.LecunUniform,
        "ones": tf.keras.initializers.Ones,
        "orthogonal": tf.keras.initializers.Orthogonal,
        "random_normal": tf.keras.initializers.RandomNormal,
        "random_uniform": tf.keras.initializers.RandomUniform,
        "truncated_normal": tf.keras.initializers.TruncatedNormal,
        "variance_scaling": tf.keras.initializers.VarianceScaling,
        "zeros": tf.keras.initializers.Zeros,
    }


@dataclasses.dataclass
class RegularizerConfig(BaseConfig[tf.keras.regularizers.Regularizer]):
    """Generic regularizer configuration for training a model.

    Attributes:
        name: the string name of the regularizer to use.
        config: a dictionary of keyword arguments to pass to the regularizer constructor.
    """

    registry = {
        "l1": tf.keras.regularizers.L1,
        "l2": tf.keras.regularizers.L2,
        "l1_l2": tf.keras.regularizers.L1L2,
        "orthogonal_regularizer": tf.keras.regularizers.OrthogonalRegularizer,
    }


@dataclasses.dataclass
class OptimizerConfig(BaseConfig[tf.keras.optimizers.Optimizer]):
    """Optimizer configuration for training a model.

    Attributes:
        name: the string name of the optimizer to use.
        config: a dictionary of keyword arguments to pass to the optimizer constructor.
    """

    name: Literal["Adam"]

    registry = {
        "Adam": tf.keras.optimizers.Adam,
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

    registry = {
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


@dataclasses.dataclass
class ConvOutputConfig:
    """Convolution output config for training a model.

    Attributes:
        regularizer: a RegularizerConfig to apply to convolutional layer outputs.
    """

    regularizer: RegularizerConfig

    def build(self):
        """Builds the convolution output configuration.

        Returns:
            A dictionary of keyword arguments to pass to convolutional layer constructors.
        """

        regularizer_object = self.regularizer.build()
        return regularizer_object


@dataclasses.dataclass
class BiasVectorConfig:
    """Constraint configuration for bias vectors in a model.

    Attributes:
        constraint: a ConstraintConfig to apply to bias vectors.
        initializer: an InitializerConfig to use for bias vectors.
        regularizer: a RegularizerConfig to apply to bias vectors.
    """

    constraint: ConstraintConfig
    initializer: InitializerConfig
    regularizer: RegularizerConfig

    def build(self):
        """Builds the bias vector configuration.

        Returns:
            A dictionary of keyword arguments to pass to layer constructors for bias vectors.
        """

        constraint_object = self.constraint.build()
        initializer_object = self.initializer.build()
        regularizer_object = self.regularizer.build()

        return {
            "bias_constraint": constraint_object,
            "bias_initializer": initializer_object,
            "bias_regularizer": regularizer_object,
        }


@dataclasses.dataclass
class KernelMatrixConfig:
    """Constraint configuration for kernel matrices in a model.

    Attributes:
        constraint: a ConstraintConfig to apply to kernel matrices.
        initializer: an InitializerConfig to use for kernel matrices.
        regularizer: a RegularizerConfig to apply to kernel matrices.
    """

    constraint: ConstraintConfig
    initializer: InitializerConfig
    regularizer: RegularizerConfig

    def build(self):
        """Builds the kernel matrix configuration.

        Returns:
            A dictionary of keyword arguments to pass to layer constructors for kernel matrices.
        """

        constraint_object = self.constraint.build()
        initializer_object = self.initializer.build()
        regularizer_object = self.regularizer.build()

        return {
            "kernel_constraint": constraint_object,
            "kernel_initializer": initializer_object,
            "kernel_regularizer": regularizer_object,
        }
