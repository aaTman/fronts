import argparse
import os
import pickle
from collections.abc import MutableMapping

import dataclasses
import dacite
import datetime

# from fronts import file_manager, data_utils
from typing import Literal, Union, Sequence
import yaml
import logging
import pathlib
import utils.data_utils
import file_manager

IMAGE_SIZE_DEFAULT: list[int] = [1, 1]
BATCH_SIZE_DEFAULT: int = 1

logging.basicConfig()
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.INFO)


@dataclasses.dataclass
class NormalizationParameters:
    """Properties for normalizations of variables.

    Attributes:
    variable: the data variable, e.g. q, T, Td.
    pressure_level: the pressure level in hPa, e.g. 950.
    min: the minimum value of the data.
    max: the maximum value.
    mean: the mean value.
    std: the standard deviation.
    mean_weighted: the latitude-weighted mean.
    std_weighted: the latitude-weighted standard deviation.
    """

    variable: str
    pressure_level: int
    min: float
    max: float
    mean: float
    std: float
    mean_weighted: float
    std_weighted: float


@dataclasses.dataclass
class NormalizationProperties:
    """Normalization properties for incoming data for predictions.

    Attributes:
    normalize_method: normalize method for incoming data. Options are 'standard'
        (z-score standardization), 'standard_weighted' (same, but with latitude
        weighting), or 'min_max' (min-max normalization).
    normalize_parameters: a dictionary of (variable, list) key value pairs where the
        list consists of 'min', 'max', 'mean', 'std', 'mean_weighted', and
        'std_weighted' parameters.
    """

    normalize_method: Literal["standard", "standard_weighted", "min_max"]
    normalize_parameters: list[NormalizationParameters]


@dataclasses.dataclass
class CustomObjects:
    loss_args: str
    loss_full_name: str
    loss_function_name: str
    metric_args: str


@dataclasses.dataclass
class TensorflowProperties:
    """Properties to pass through into set_ai2es_tf_physical_devices.

    Attributes:
    gpu_device: the number of the device being used.
    memory_growth: whether or not to utilize memory growth, which if True will prevent
        the runtime initialization from allocating all of the memory on the device (see
        https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth).
    """

    gpu_device: int
    memory_growth: bool

    def build(self):
        import tensorflow as tf

        # From https://dopplerchase-ai2es-schooner-hpc.readthedocs.io/en/latest/general_gpu.html#sharing-gpus
        if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
            # Fetch list of logical GPUs that have been allocated
            # Will always be numbered 0, 1, …
            physical_devices = tf.config.get_visible_devices("GPU")
            n_physical_devices = len(physical_devices)
            if n_physical_devices > 0:
                tf.config.set_visible_devices(
                    devices=physical_devices[self.gpu_device], device_type="GPU"
                )
                if self.memory_growth:
                    tf.config.experimental.set_memory_growth(
                        device=physical_devices[self.gpu_device], enable=True
                    )

            # Set memory growth for each
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, self.memory_growth)
        else:
            # No allocated GPUs: do not delete this case!\
            logger.info(
                ("WARNING: No GPUs found, all computations will be performed on CPUs.")
            )
            tf.config.set_visible_devices([], "GPU")


def flatten_dict(dictionary, parent_key="", separator="_"):
    """
    Turn a nested dictionary into a flattened dictionary
    with concatenated keys.
    """
    items = []
    for key, value in dictionary.items():
        # Create the new key, handling the first level
        new_key = str(parent_key) + separator + str(key) if parent_key else str(key)

        # If the value is another dictionary, recurse
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))

    return dict(items)


def parse_arguments() -> dict:
    """Parse CLI arguments.

    Arguments set in the CLI will supercede any option set in a provided
    configuration yaml file.
    """
    # Set up and parse to see if config is in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, help="path to configuration yaml file"
    )

    # Add all other arguments relevant for prediction to run
    parser.add_argument(
        "--variable_netcdf_dir",
        type=str,
        help="Main directory for the netcdf files containing variable data.",
    )
    parser.add_argument(
        "--init_time",
        type=str,
        help=("Date and time of the data in the format yyyy-mm-dd-hh."),
    )
    parser.add_argument("--domain", type=str, help="Domain of the data.")
    parser.add_argument(
        "--num_images",
        type=int,
        nargs=2,
        help=(
            "Number of images for each dimension the final stitched map for "
            "predictions: lon, lat"
        ),
    )
    parser.add_argument("--gpu_device", type=int, help="GPU device number.")
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for the model predictions."
    )
    parser.add_argument(
        "--image_size",
        type=str,
        help=(
            "Number of pixels along each dimension of the model's output in the format"
            " lon,lat"
        ),
    )
    parser.add_argument(
        "--memory_growth", action="store_true", help="Use memory growth on the GPU"
    )
    parser.add_argument(
        "--model_dir", type=str, help="Directory for the models and properties."
    )
    parser.add_argument("--model_number", type=int, help="Model number.")
    parser.add_argument("--data_source", type=str, help="Data source for variables")

    # Set args as a dict instead of a Namespace
    args = vars(parser.parse_args())

    # Load configuration if exists
    if args["config"] is not None:
        with open(args["config"], "r") as config_file:
            prediction_config = yaml.safe_load(config_file)
    # If not, initialize an empty dict
    else:
        prediction_config = dict()

    for key, value in args.items():
        # Replace args if they were set in command line call
        if value is not None:
            prediction_config[key] = args[key]
        else:
            # Fill rest of prediction_config in with keys were not declared in the
            # config yaml
            if key not in prediction_config.keys():
                prediction_config[key] = None

    # For args with defaults, check if not None, else apply defaults

    return prediction_config


# def assign_config_to_dataclasses(
#     prediction_config: dict,
# ) -> tuple[TensorflowProperties, ModelProperties]:
#     """Initializes dataclasses with respective configuration arguments.

#     Using the incoming prediction_config based on the configuration yaml and/or the
#     command line arguments, builds dataclasses based on the keys in the dictionary.

#     Args:
#     prediction_config: the prediction configuration dictionary.

#     Returns ModelProperties and TensorflowProperties dataclasses.
#     """
#     tensorflow_dict = {
#         k: v
#         for k, v in prediction_config
#         if k in dataclasses.fields(TensorflowProperties)
#     }
#     tensorflow_properties = dacite.from_dict(
#         data_class=TensorflowProperties, data=tensorflow_dict
#     )

#     model_dict = {
#         k: v for k, v in prediction_config if k in dataclasses.fields(ModelProperties)
#     }
#     model_properties = dacite.from_dict(data_class=ModelProperties, data=model_dict)

#     return tensorflow_properties, model_properties


def load_model_properties(model_directory: Union[str, pathlib.Path]):
    """Generates a ModelProperties dataclass from incoming pickle file."""

    with open(model_directory, "rb") as f:
        model_properties_dict = pickle.load(f)

    model_properties_dict = flatten_dict(model_properties_dict)
    return model_properties_dict


def assign_normalization_properties_to_dataclass(
    normalization_config: dict,
) -> NormalizationProperties:
    """Initializes dataclasses for normalization parameters provided in the model
    properties.

    Args:
    normalization_config: dictionary of the configuration options for normalization
        procedures.

    Returns NormalizationProperties dataclass.
    """
    normalization_dict = {
        k: v
        for k, v in normalization_config
        if k in dataclasses.fields(NormalizationProperties)
    }
    normalization_properties = dacite.from_dict(
        data_class=NormalizationProperties, data=normalization_dict
    )

    return normalization_properties


def run_prediction() -> None:
    prediction_config = parse_arguments()

    # Convert model path string to Path object and convert to dataclass
    model_path = pathlib.Path(prediction_config["model_dir"])
    model_number = prediction_config["model_number"]
    model_subpath = f"model_{prediction_config['model_number']}"

    # Get the pickle file from the fronts training output
    model_pickle = (
        model_path / model_subpath / "_".join([model_subpath, "properties.pkl"])
    )

    model_h5 = model_path / model_subpath / "".join([model_subpath, ".h5"])
    model_properties = load_model_properties(model_pickle)
    model = file_manager.load_model(
        model_number=model_number, model_dir=model_path.as_posix()
    )
    logger.info(model_properties)
    breakpoint()


if __name__ == "__main__":
    run_prediction()
