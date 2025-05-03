"""
Generate bincounts for ERA5 variables. The bincounts generated across billions of points can allow us to get parameters
from which we can normalize the data.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2025.5.3
"""
import os
import numpy as np
import xarray as xr
from glob import glob
import argparse
from datetime import datetime


TRANSPOSE_DIMS = {'MERGIR': ('lat', 'lon', 'time'),
                  'era5': ('latitude', 'longitude', 'pressure_level', 'time'),
                  'goes-merged': ('latitude', 'longitude', 'time')}

BINS = {'T': np.arange(100, 400.1),
        'Td': np.arange(100, 400.1),
        'Tv': np.arange(100, 400.1),
        'Tw': np.arange(100, 400.1),
        'theta': np.arange(100, 500.1),
        'theta_e': np.arange(100, 500.1),
        'theta_v': np.arange(100, 500.1),
        'theta_w': np.arange(100, 500.1),
        'RH': np.arange(0, 1.001, 0.0025),
        'r': np.arange(0, 50.01, 0.125),
        'q': np.arange(0, 50.01, 0.125),
        'u': np.arange(-80, 150, 0.4),
        'v': np.arange(-80, 150, 0.4),
        'band_1': np.arange(0, 1.001, 0.0025),
        'band_2': np.arange(0, 1.001, 0.0025),
        'band_3': np.arange(0, 1.001, 0.0025),
        'band_4': np.arange(0, 1.001, 0.0025),
        'band_5': np.arange(0, 1.001, 0.0025),
        'band_6': np.arange(0, 1.001, 0.0025),
        'band_7': np.arange(150, 400.01, 0.5),
        'band_8': np.arange(150, 400.01, 0.5),
        'band_9': np.arange(150, 400.01, 0.5),
        'band_10': np.arange(150, 400.01, 0.5),
        'band_11': np.arange(150, 400.01, 0.5),
        'band_12': np.arange(150, 400.01, 0.5),
        'band_13': np.arange(150, 400.01, 0.5),
        'band_14': np.arange(150, 400.01, 0.5),
        'band_15': np.arange(150, 400.01, 0.5),
        'band_16': np.arange(150, 400.01, 0.5)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--netcdf_indir', type=str, required=True, help='Base directory for the input netcdf files.')
    parser.add_argument('--bincount_outdir', type=str, required=True, help='Output directory for the netcdf files containing bincounts.')
    parser.add_argument('--variables', type=str, nargs='+')
    parser.add_argument('--data_source', type=str, default='era5', help='Data source (e.g., "era5", "MERGIR", etc.)')
    parser.add_argument('--year_and_month', type=int, nargs=2, required=True)
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing datasets.')
    args = vars(parser.parse_args())
    
    year, month = args['year_and_month']
    transpose_dims = TRANSPOSE_DIMS[args['data_source']]
    
    netcdf_files = list(sorted(glob(args['netcdf_indir'] + '/%d%02d/%s*_%d%02d*.nc' % (year, month, args['data_source'], year, month))))
    
    print(f"{datetime.utcnow()}: Opening dataset")
    ds = xr.open_mfdataset(netcdf_files, combine='nested', concat_dim='time', engine='h5netcdf', chunks='auto')
    ds = ds.transpose(*transpose_dims).astype('float32')
    print(f"{datetime.utcnow()}: Dataset loaded")
    if args['variables'] is None:
        args['variables'] = list(ds.keys())
    
    if args['data_source'] == 'era5':
        
        # for pressure_level in ds['pressure_level'].values:
        for pressure_level in ['300',]:
            for var_str in args['variables']:
                
                if var_str == 'sp_z':
                    if pressure_level == 'surface':
                        bins = np.arange(450., 1080.1)
                    elif pressure_level == '700':
                        bins = np.arange(150., 450.1)
                    elif pressure_level == '500':
                        bins = np.arange(300., 700.1)
                    elif pressure_level == '300':
                        bins = np.arange(700., 1400.1)
                else:
                    bins = BINS[var_str]
                
                # if pressure level is read as a float, convert it to an integer
                pressure_level = int(pressure_level) if isinstance(pressure_level, float) else pressure_level
    
                N_bins = len(bins) - 1
                
                print("%d-%02d: (%s_%s)" % (year, month, var_str, pressure_level))
                save_var_str = "%s_%s" % (var_str, pressure_level)
                
                output_folder = '%s/%s_bins' % (args['bincount_outdir'], save_var_str)
                bincount_file = '%s/%s_bincounts_%d%02d.nc' % (output_folder, save_var_str, year, month)
                bincount_ds_exists = os.path.isfile(bincount_file)
                if bincount_ds_exists and not args['overwrite']:
                    print("%s already exists. If you want to overwrite the existing dataset, rerun this script with the "
                          "--overwrite flag attached." % bincount_file)
                    continue
                
                da = ds.sel(pressure_level=pressure_level)[var_str]
                lat = da[transpose_dims[0]].values
                time_array = da['time'].values
                Ntime = len(time_array)
                Nlon = len(da[transpose_dims[1]])
                Nlat = len(lat)
                bincounts_by_latitude = np.zeros([Nlat, N_bins], dtype=np.int64)
        
                var = da.to_numpy().reshape((Nlat, Nlon * Ntime))
                var = np.nan_to_num(var)
                for ilat in range(Nlat):
                    bincounts_by_latitude[ilat, :] += np.histogram(var[ilat, :], bins)[0]
                del var
                
                os.makedirs(output_folder, exist_ok=True)
                
                print("saving dataset")
                bincount_ds = xr.Dataset(data_vars={'%s_bincount' % save_var_str: (('latitude', 'bin'), bincounts_by_latitude)},
                                         coords={'latitude': lat, 'bin': bins[:-1]})
                bincount_ds = bincount_ds.expand_dims({'time': np.array(['%d-%02d' % (year, month)], dtype=datetime)})
                bincount_ds.to_netcdf(bincount_file, engine='netcdf4', mode='w')
                bincount_ds.close()
                
            ds.close()
    
    elif args['data_source'] == 'goes-merged':
        
        for var_str in args['variables']:
            
            bins = BINS[var_str]
            N_bins = len(bins) - 1
            print("%d-%02d: (%s)" % (year, month, var_str))
            save_var_str = var_str

            output_folder = '%s/%s_bins' % (args['bincount_outdir'], save_var_str)
            bincount_file = '%s/%s_bincounts_%d%02d.nc' % (output_folder, save_var_str, year, month)
            bincount_ds_exists = os.path.isfile(bincount_file)
            if bincount_ds_exists and not args['overwrite']:
                print("%s already exists. If you want to overwrite the existing dataset, rerun this script with the "
                      "--overwrite flag attached." % bincount_file)
                continue
            
            da = ds[var_str]
            lat = da[transpose_dims[0]].values
            time_array = da['time'].values
            Ntime = len(time_array)
            Nlon = len(da[transpose_dims[1]])
            Nlat = len(lat)
            bincounts_by_latitude = np.zeros([Nlat, N_bins], dtype=np.int64)
    
            var = da.to_numpy().reshape((Nlat, Nlon * Ntime))
            var = np.nan_to_num(var, nan=-99999)
            for ilat in range(Nlat):
                bincounts_by_latitude[ilat, :] += np.histogram(var[ilat, :], bins)[0]
            del var
            
            os.makedirs(output_folder, exist_ok=True)
            
            print("saving dataset")
            bincount_ds = xr.Dataset(data_vars={'%s_bincount' % save_var_str: (('latitude', 'bin'), bincounts_by_latitude)},
                                     coords={'latitude': lat, 'bin': bins[:-1]})
            bincount_ds = bincount_ds.expand_dims({'time': np.array(['%d-%02d' % (year, month)], dtype=datetime.datetime)})
            bincount_ds.to_netcdf(bincount_file, engine='netcdf4', mode='w')
            bincount_ds.close()