import argparse
import os

import tensorflow as tf

import dataclasses
# import dacite
# from fronts import file_manager, data_utils
from typing import Literal
import yaml
import logging

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
    normalize_method: normalize method for incoming data. Options are 
        'standard' (z-score standardization), 'standard_weighted' (same, but 
        with latitude weighting), or 'min_max' (min-max normalization).
    normalize_parameters: a dictionary of (variable, list) key value pairs where 
        the list consists of 'min', 'max', 'mean', 'std', 'mean_weighted', 
        and 'std_weighted' parameters.
"""
    normalize_method: Literal['standard','standard_weighted','min_max']
    normalize_parameters: list[NormalizationParameters]


@dataclasses.dataclass
class ModelProperties:
    """ML model properties to determine functionality and outputs.
    
    Attributes:
    model_type: a string indicating which model type. Options include
        unet, attention_unet, or unet_3plus.
    front_types: which fronts are included in the model to predict. See
        more information in `utils.data_utils.reformat_fronts`. 
    variables: variables that will be used to predict front type.
    pressure_levels: pressure levels that will be used to predict front
        type.
    image_size: longitude, latitude size of model predictions (in degrees).
    """
    model_type: str
    front_types: list[str]
    variables: list[str]
    pressure_levels: list[str]
    image_size: tuple[int, int]


def set_ai2es_tf_physical_devices(gpu_device: int, memory_growth: bool=False):
    """Set tensorflow configuration for physical devices and options.
    
    Args:
    
    gpu_device: which GPU to use as an integer.
    memory_growth: whether or not to set memory growth for a PhysicalDevice (see 
        https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth). 
    """

    # From https://dopplerchase-ai2es-schooner-hpc.readthedocs.io/en/latest/general_gpu.html#sharing-gpus
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        # Fetch list of logical GPUs that have been allocated
        # Will always be numbered 0, 1, …
        physical_devices = tf.config.get_visible_devices('GPU')
        n_physical_devices = len(physical_devices)
        if n_physical_devices > 0:
            tf.config.set_visible_devices(devices=physical_devices[gpu_device], device_type='GPU')
            if memory_growth:
                tf.config.experimental.set_memory_growth(device=physical_devices[gpu_device], enable=True)

        # Set memory growth for each
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, memory_growth)
    else:
            #No allocated GPUs: do not delete this case!\
        logger.info('WARNING: No GPUs found, all computations will be performed on CPUs.')
        tf.config.set_visible_devices([], 'GPU')

def parse_arguments():
    """Parse CLI arguments. 
    
    Arguments set in the CLI will supercede any option set in a provided
    configuration yaml file.
    """
    # Set up and parse to see if config is in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to configuration yaml file")

    # Add all other arguments relevant for prediction to run
    parser.add_argument('--variable_netcdf_dir', type=str, help='Main directory for the netcdf files containing variable data.')
    parser.add_argument('--init_time', type=int, nargs=4, help='Date and time of the data. Pass 4 ints in the following order: year, month, day, hour')
    parser.add_argument('--domain', type=str, help='Domain of the data.')
    parser.add_argument('--num_images', type=int, nargs=2, help='Number of images for each dimension the final stitched map for predictions: lon, lat')
    parser.add_argument('--gpu_device', type=int, help='GPU device number.')
    parser.add_argument('--batch_size', type=int, help="Batch size for the model predictions.")
    parser.add_argument('--image_size', type=int, nargs=2, help="Number of pixels along each dimension of the model's output: lon, lat")
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth on the GPU')
    parser.add_argument('--model_dir', type=str, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, help='Model number.')
    parser.add_argument('--data_source', type=str, help='Data source for variables')

    # Set args as a dict instead of a Namespace
    args = vars(parser.parse_args())

    # Load configuration if exists
    if args['config'] is not None:
        with open(args['config'], "r") as config_file:
            prediction_config = yaml.safe_load(config_file)
    # If not, initialize an empty dict    
    else:
        prediction_config = dict()

    for key, value in args.items():
        # Replace args if they were set in command line call
        if value is not None:
            prediction_config[key] = args[key]
        else:
            # Fill rest of prediction_config in with keys were not declared in the config yaml
            if key not in prediction_config.keys():
                prediction_config[key] = None

if __name__ == '__main__':
    parse_arguments()