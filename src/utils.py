import parflow
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def potts_update(x, temperature, n_states):
    print(x.shape)
    N, M = x.shape
    # Test location and state
    a = np.random.randint(1, N-1)
    b = np.random.randint(1, M-1)
    # States are distributed around the circle
    s =  2 * np.pi * x[a, b] / n_states
    # Cost calculation
    neighbor_states = np.array([x[a+1, b], x[a, b+1], x[a-1, b], x[a, b-1]])
    neighbor_circle = 2 * np.pi * neighbor_states / n_states
    cost = np.sum(np.cos(s - neighbor_circle))
    if cost < 0 or np.random.rand() < np.exp(-cost*temperature):
        # Choose a new state
        snew = np.random.choice(neighbor_states)
    else:
        snew = x[a, b]
    x[a, b] = snew
    return x
 

def simulate_2d_potts(x, temp, n_states, n_steps=10_000):
    xnew = x.copy()
    for _ in range(n_steps):
        xnew = potts_update(xnew, temp, n_states)
    return xnew


def update_conus1_indicator(indicator, n_steps=100_000):
    num_classes = [
        len(np.unique(indicator[0])),
        len(np.unique(indicator[1])),
        len(np.unique(indicator[2])),
    ]
    slices = [slice(0, 1), slice(1, 2), slice(3, None)]
    for i, s in enumerate(slices):
        indicator[s] = simulate_2d_potts(indicator[i], 5.0, num_classes[i], n_steps=n_steps)
    return indicator


def plot_indicator(indicator):
    fig, axes = plt.subplots(1, 5, figsize=(15,3))
    for i, ax in enumerate(axes):
        ax.imshow(indicator[i])


def update_parameter(default, scale=0.25):
    new_val = np.random.normal(
        loc=default,
        scale=scale*default,
        size=1
    )[0]
    return max(0, new_val)


def update_all_parameters(run):
    run.Geom.s1.Perm.Value = update_parameter(run.Geom.s1.Perm.Value)
    run.Geom.s2.Perm.Value = update_parameter(run.Geom.s2.Perm.Value)
    run.Geom.s3.Perm.Value = update_parameter(run.Geom.s3.Perm.Value)
    run.Geom.s4.Perm.Value = update_parameter(run.Geom.s4.Perm.Value)
    run.Geom.s5.Perm.Value = update_parameter(run.Geom.s5.Perm.Value)
    run.Geom.s6.Perm.Value = update_parameter(run.Geom.s6.Perm.Value)
    run.Geom.s7.Perm.Value = update_parameter(run.Geom.s7.Perm.Value)
    run.Geom.s8.Perm.Value = update_parameter(run.Geom.s8.Perm.Value)
    run.Geom.s9.Perm.Value = update_parameter(run.Geom.s9.Perm.Value)
    run.Geom.s10.Perm.Value = update_parameter(run.Geom.s10.Perm.Value)
    run.Geom.s11.Perm.Value = update_parameter(run.Geom.s11.Perm.Value)
    run.Geom.s12.Perm.Value = update_parameter(run.Geom.s12.Perm.Value)
    run.Geom.s13.Perm.Value = update_parameter(run.Geom.s13.Perm.Value)
    run.Geom.g1.Perm.Value = update_parameter(run.Geom.g1.Perm.Value)
    run.Geom.g2.Perm.Value = update_parameter(run.Geom.g2.Perm.Value)
    run.Geom.g3.Perm.Value = update_parameter(run.Geom.g3.Perm.Value)
    run.Geom.g4.Perm.Value = update_parameter(run.Geom.g4.Perm.Value)
    run.Geom.g5.Perm.Value = update_parameter(run.Geom.g5.Perm.Value)
    run.Geom.g6.Perm.Value = update_parameter(run.Geom.g6.Perm.Value)
    run.Geom.g7.Perm.Value = update_parameter(run.Geom.g7.Perm.Value)
    run.Geom.g8.Perm.Value = update_parameter(run.Geom.g8.Perm.Value)
    run.Geom.b1.Perm.Value = update_parameter(run.Geom.b1.Perm.Value)
    run.Geom.b2.Perm.Value = update_parameter(run.Geom.b2.Perm.Value)

    run.Geom.s1.Porosity.Value = update_parameter(run.Geom.s1.Porosity.Value)
    run.Geom.s2.Porosity.Value = update_parameter(run.Geom.s2.Porosity.Value)
    run.Geom.s3.Porosity.Value = update_parameter(run.Geom.s3.Porosity.Value)
    run.Geom.s4.Porosity.Value = update_parameter(run.Geom.s4.Porosity.Value)
    run.Geom.s5.Porosity.Value = update_parameter(run.Geom.s5.Porosity.Value)
    run.Geom.s6.Porosity.Value = update_parameter(run.Geom.s6.Porosity.Value)
    run.Geom.s7.Porosity.Value = update_parameter(run.Geom.s7.Porosity.Value)
    run.Geom.s8.Porosity.Value = update_parameter(run.Geom.s8.Porosity.Value)
    run.Geom.s9.Porosity.Value = update_parameter(run.Geom.s9.Porosity.Value)
    run.Geom.s10.Porosity.Value = update_parameter(run.Geom.s10.Porosity.Value)
    run.Geom.s11.Porosity.Value = update_parameter(run.Geom.s11.Porosity.Value)
    run.Geom.s12.Porosity.Value = update_parameter(run.Geom.s12.Porosity.Value)
    run.Geom.s13.Porosity.Value = update_parameter(run.Geom.s13.Porosity.Value)
    run.Geom.g1.Porosity.Value = update_parameter(run.Geom.g1.Porosity.Value)
    run.Geom.g2.Porosity.Value = update_parameter(run.Geom.g2.Porosity.Value)
    run.Geom.g3.Porosity.Value = update_parameter(run.Geom.g3.Porosity.Value)
    run.Geom.g4.Porosity.Value = update_parameter(run.Geom.g4.Porosity.Value)
    run.Geom.g5.Porosity.Value = update_parameter(run.Geom.g5.Porosity.Value)
    run.Geom.g6.Porosity.Value = update_parameter(run.Geom.g6.Porosity.Value)
    run.Geom.g7.Porosity.Value = update_parameter(run.Geom.g7.Porosity.Value)
    run.Geom.g8.Porosity.Value = update_parameter(run.Geom.g8.Porosity.Value)

    return run


def calculate_water_table_depth(ds, dz):
    wtd_list = []
    for t in range(len(ds['time'])):
        wtd_list.append(parflow.tools.hydrology.calculate_water_table_depth(
            ds['pressure'].values[t], 
            ds['saturation'].values[t], 
            dz=dz
        ))
    wtd = xr.DataArray(
        np.stack(wtd_list),
        coords={'time': ds['time']},
        dims=('time', 'y', 'x')
    )
    return wtd


def calculate_flow(ds, slope_x, slope_y, mannings, dx, dy, mask):
    flow_list = []
    for t in range(len(ds['time'])):
        flow_list.append(parflow.tools.hydrology.calculate_overland_flow_grid(
            ds['pressure'].values[t], 
            slope_x, slope_y, mannings, dx, dy, mask=mask
        ))
    flow = xr.DataArray(
        np.stack(flow_list),
        coords={'time': ds['time']},
        dims=('time', 'y', 'x')
    )
    return flow

