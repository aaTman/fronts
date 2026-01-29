"""
Scans for missing or corrupt ERA5 files in a directory.
* ERA5 files must be downloaded from the Climate Data Store using the 'download_era5.py' script.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2025.5.3
"""
import numpy as np
import itertools
import xarray as xr
import argparse
import os

YEARS = np.arange(2008, 2024.1, 1).astype(int)
VARS_TO_CHECK = ['u_component_of_wind', 'v_component_of_wind', 'temperature', 'geopotential', 'specific_humidity']
PRESSURE_LEVELS_TO_CHECK = ['1000', '950', '900', '850', '700', '500', '300']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help="Directory where the ERA5 files are stored.")
    args = vars(parser.parse_args())
    
    missing = []
    bad = []
    
    for var, lvl, yr in itertools.product(VARS_TO_CHECK, PRESSURE_LEVELS_TO_CHECK, YEARS):
        file = args['dir'] + '/era5_%s-%shPa_%d.nc' % (var, lvl, yr)
        
        if not os.path.isfile(file):
            missing.append([var, lvl, yr])
        else:
            try:
                xr.open_dataset(file)
            except:
                bad.append(file)
    
    print("\n=== MISSING FILES ===")
    print(missing)
    print("\n=== BAD DATASETS ===")
    print(bad)