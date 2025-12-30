import argparse
import pandas as pd
import numpy as np
import xarray as xr
import os
import sys
import tensorflow as tf
import scipy
import dataclasses
import dacite
from fronts import file_manager, data_utils
from typing import Literal
import yaml

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


def parse_arguments_no_config(parser: argparse.ArgumentParser) -> dict:
    """Parse CLI arguments into a dict if a config file is not provided.
    
    Args:
    parser: ArgumentParser base to build off of.
    
    Returns a dictionary using `parse_args()` using arguments included in function.
    """

    parser.add_argument('--netcdf_indir', type=str, help='Main directory for the netcdf files containing variable data.')
    parser.add_argument('--mergir_indir', type=str, help="Input directory for the netCDF files containing MERGIR data.")
    parser.add_argument('--init_time', type=int, nargs=4, help='Date and time of the data. Pass 4 ints in the following order: year, month, day, hour')
    parser.add_argument('--domain', type=str, help='Domain of the data.')
    parser.add_argument('--num_images', type=int, nargs=2, default=[1, 1], help='Number of images for each dimension the final stitched map for predictions: lon, lat')
    parser.add_argument('--gpu_device', type=int, help='GPU device number.')
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for the model predictions.")
    parser.add_argument('--image_size', type=int, nargs=2, help="Number of pixels along each dimension of the model's output: lon, lat")
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth on the GPU')
    parser.add_argument('--model_dir', type=str, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, help='Model number.')
    parser.add_argument('--data_source', type=str, default='era5', help='Data source for variables')
    return parser.parse_args()

def parse_arguments():
    """Parse CLI arguments."""
    # Set up and parse to see if config is in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to configuration yaml file")
    args = parser.parse_args()
    if args.config is not None:
        model_properties = yaml.safe_load(args['config'])
    else:
        model_properties = parse_arguments_no_config(parser)
    



if __name__ == '__main__':
    pass