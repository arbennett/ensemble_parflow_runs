import click
import xarray as xr
import numpy as np
import pandas as pd
import parflow as pf
import logging
import os
os.environ['PARFLOW_DIR'] = '/home/SHARED/software/parflow/3.10.0'
os.environ['HYDRODATA_URL'] = '/hydrodata'

import parflow.tools.io

from datetime import datetime, timedelta
from hf_hydrodata import gridded
from parflow import Run
from parflow.tools.io import read_pfb, write_pfb, read_clm, read_pfb_sequence
from parflow.tools.fs import cp, mkdir
from parflow.tools.settings import set_working_directory
from subsettools.subsettools import  (
    create_mask_solid,
    subset_press_init,
    subset_forcing,
    subset_static,
    config_clm,
    dist_run,
    huc_to_ij,
    edit_runscript_for_subset,
    change_filename_values,
)

from utils import (
    update_conus1_indicator,
    update_all_parameters,
    calculate_water_table_depth,
    calculate_flow,
)
from subsettools.datasets import get_ref_yaml_path
from pathlib import Path
from glob import glob

logging.getLogger().setLevel(logging.WARNING)

@click.command()
@click.option('--out_dir', default='.', help='Output directory')
@click.option('--runname', default='subset_run', help='Name of the run')
@click.option('--grid', default='conus1', help='Name of the grid')
@click.option('--domain_type', default='box', help='Type of domain')
@click.option('--huc_id', default='15060202', help='HUC ID of the domain')
@click.option('--start', default='2005-10-01', help='Start date of the run')
@click.option('--end', default='2006-09-30', help='End date of the run')
@click.option('--p', default=4, help='Number of partitions in the x direction')
@click.option('--q', default=4, help='Number of partitions in the y direction')
@click.option('--number_runs', default=3, help='Number of runs to perform')
@click.option('--modify_indicator', is_flag=True, default=False, help='Whether to modify the indicator')
@click.option('--modify_parameters', is_flag=True, default=False, help='Whether to modify the parameters')
def main(
    out_dir,
    runname,
    grid,
    domain_type,
    huc_id,
    start,
    end,
    p,
    q,
    number_runs,
    modify_indicator,
    modify_parameters,
):
    #provide a way to create a subset from the conus domain (huc, lat/lon bbox currently supported)
    # Verde river is 15060202
    subset_target = [huc_id]

    #provide information about the datasets you want to access for run inputs using the data catalog
    # We have 2003-2006 NLDAS2 data for this region
    grid = "conus1"  
    baseline_dataset = "conus1_baseline_mod"
    static_dataset = "conus1_domain"
    forcing_dataset = "NLDAS2"

    dz = np.array([100.0, 1.0, 0.6, 0.3, 0.1])
    et_idx = 4
    swe_idx = 10
    timestamps = pd.date_range(start, end, freq='1H') 
    ntime = len(timestamps) - 1

    base_dir = f"{out_dir}"
    mkdir(base_dir)
    forcing_dir = f"{base_dir}/forcing/"
    mkdir(forcing_dir)
    settings_dir = f"{base_dir}/settings/"
    mkdir(settings_dir)
    output_dir = f"{base_dir}/outputs/"
    mkdir(output_dir) 
    zarr_dir = f'{base_dir}/zarr'
    mkdir(zarr_dir)

    target_runscript = f'{settings_dir}/{runname}.yaml'

    reference_run = get_ref_yaml_path(grid, 'transient', domain_type, settings_dir)
    ij_bounds = huc_to_ij(subset_target, grid) #[imin, jmin, imax, jmax]
    imin, jmin, imax, jmax = ij_bounds
    x_inds = np.arange(imin, imax)
    y_inds = np.arange(jmin, jmax)
    create_mask_solid(subset_target, grid, settings_dir)
    init_cond_path = subset_press_init(ij_bounds, dataset=baseline_dataset, date=start, write_dir=settings_dir, time_zone='UTC')
    init_cond_filename = init_cond_path.split('/')[-1]
    config_clm(ij_bounds, start=start, end=end, dataset=baseline_dataset, write_dir=settings_dir)
    subset_forcing(ij_bounds, grid=grid, start=start, end=end, dataset=forcing_dataset, write_dir=forcing_dir)
    subset_static(ij_bounds, dataset=static_dataset, write_dir=settings_dir)
    edit_runscript_for_subset(
        ij_bounds, 
        runscript_path=reference_run, 
        write_dir=settings_dir, 
        runname=runname, 
        forcing_dir=forcing_dir
    )

    change_filename_values(runscript_path=target_runscript, init_press=init_cond_filename)
    run = Run.from_definition(target_runscript)
    run.TimingInfo.StopTime = ntime
    run.Process.Topology.P = p
    run.Process.Topology.Q = q


    indicator_file = f'{settings_dir}/pf_indicator.pfb'
    indicator = pf.read_pfb(indicator_file)

    slope_x_file = f'{settings_dir}/slope_x.pfb'
    slope_x = pf.read_pfb(slope_x_file)

    slope_y_file = f'{settings_dir}/slope_y.pfb'
    slope_y = pf.read_pfb(slope_y_file)

    mannings = 2.0 * np.ones_like(slope_x)
    dx = 1000.0
    dy = 1000.0

    if modify_parameters:
        run = update_all_parameters(run)
    for i in range(number_runs):

        if modify_indicator:
            indicator = update_conus1_indicator(indicator)
            pf.write_pfb(indicator_file, indicator, p=p, q=q)


        dist_run(
            P=p, Q=q, 
            runscript_path=target_runscript, 
            working_dir=settings_dir,
            dist_clim_forcing=True
        )

        run.run(working_directory=settings_dir)

        # Postprocessing
        pressure_files = sorted(glob(f'{settings_dir}/{runname}.out.press.*.pfb')[1:])
        saturation_files = sorted(glob(f'{settings_dir}/{runname}.out.satur.*.pfb')[1:])
        clm_files = sorted(glob(f'{settings_dir}/{runname}.out.clm_output.*.pfb'))

        timesteps = pd.date_range(start, periods=len(pressure_files), freq='1H')
        ds = xr.Dataset()
        ds['pressure'] = xr.DataArray(
            read_pfb_sequence(pressure_files),
            coords={'time': timesteps}, 
            dims=('time', 'z', 'y', 'x')
        )
        mask = ds['pressure'].isel(time=0).values > -9999
        ds['saturation'] = xr.DataArray(
            read_pfb_sequence(saturation_files),
            coords={'time': timesteps}, 
            dims=('time', 'z', 'y', 'x')
        )
        clm = xr.DataArray(
            read_pfb_sequence(clm_files),
            coords={'time': timesteps}, 
            dims=('time', 'feature', 'y', 'x')
        )
        ds['wtd'] = calculate_water_table_depth(ds, dz)
        ds['streamflow'] = calculate_flow(
            ds, slope_x, slope_y, mannings, dx, dy, mask
        )
        ds['swe'] = clm.isel(feature=swe_idx)
        ds['et'] = clm.isel(feature=et_idx)
        ds = ds.resample(time='1D').mean()
        ds['porosity'] = xr.DataArray(
            read_pfb(f'{settings_dir}/{runname}.out.porosity.pfb'),
            dims=('z','y','x')
        )
        ds['permeability'] = xr.DataArray(
            read_pfb(f'{settings_dir}/{runname}.out.perm_x.pfb'),
            dims=('z','y','x')
        )
        ds['van_genuchten_alpha'] = xr.DataArray(
            read_pfb(f'{settings_dir}/{runname}.out.alpha.pfb'),
            dims=('z','y','x')
        )
        ds['van_genuchten_n'] = xr.DataArray(
            read_pfb(f'{settings_dir}/{runname}.out.n.pfb'),
            dims=('z','y','x')
        )
        lat = parflow.tools.io.read_clm(f'{settings_dir}/drv_vegm.dat', type='vegm')[:, :, 0]
        lon = parflow.tools.io.read_clm(f'{settings_dir}/drv_vegm.dat', type='vegm')[:, :, 1]
        ds = ds.assign_coords({
            'lat': xr.DataArray(lat, dims=['y', 'x']),
            'lon': xr.DataArray(lon, dims=['y', 'x']),
        })
        ds['x_inds'] = xr.DataArray(x_inds, dims=['x'])
        ds['y_inds'] = xr.DataArray(y_inds, dims=['y'])
        ds = ds.astype(np.float32)
        ds.to_zarr(f'{zarr_dir}/{runname}_{i}.zarr', mode='w')

        # Clean up
        del ds
        del clm
        _ = [os.remove(os.path.abspath(f)) for f in pressure_files]
        _ = [os.remove(os.path.abspath(f)+'.dist') for f in pressure_files]
        _ = [os.remove(os.path.abspath(f)) for f in saturation_files]
        _ = [os.remove(os.path.abspath(f)+'.dist') for f in saturation_files]
        _ = [os.remove(os.path.abspath(f)) for f in clm_files]
        _ = [os.remove(os.path.abspath(f)+'.dist') for f in clm_files]
    _ = [os.remove(os.path.abspath(f)) for f in glob(f'{forcing_dir}/*')]



if __name__ == '__main__':
    main()