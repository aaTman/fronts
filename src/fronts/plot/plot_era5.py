"""
Visualize ERA5 netCDF files.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2025.5.3
"""

import argparse
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import utils.misc
from utils.plotting import plot_background


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--netcdf_dir', type=str, required=True, help='Base directory for the ERA5 netcdf files.')
    parser.add_argument('--plot_outdir', type=str,
        help='Output directory for the generated plots. If no directory is declared, the plot will be shown with plt.show().')
    parser.add_argument('--valid_time', type=str, required=True, help='Valid time with format YYYY-MM-DD-HH.')
    parser.add_argument('--variable', type=str, required=True, help='Variable to plot.')
    parser.add_argument('--pressure_level', type=str, required=True, help='Pressure level of interest.')
    parser.add_argument('--plot_kwargs', type=str,
        help='Additional arguments to pass to plt.plot(). See utils.misc.string_arg_to_dict() for details.')
    args = vars(parser.parse_args())
    
    yr, mo, dy, hr = args['valid_time'].split('-')
    
    nc_file = f"{args['netcdf_dir']}/{yr}{mo}/era5_{yr}{mo}{dy}{hr}_global.nc"
    ds = xr.open_dataset(nc_file).sel(pressure_level=args['pressure_level'])[args['variable']]
    
    fig, ax = plt.subplots(figsize=(16, 8), dpi=500, subplot_kw={'projection': ccrs.PlateCarree()})
    plot_background(ax=ax)
    
    plot_kwargs = utils.misc.string_arg_to_dict(args['plot_kwargs'])
    ds.plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(), **plot_kwargs)
    
    if args['plot_outdir'] is not None:
        mpl.use('Agg')
        output_file = f"{args['plot_outdir']}/era5_{yr}{mo}{dy}{hr}_{args['variable']}-{args['pressure_level']}.png"
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()