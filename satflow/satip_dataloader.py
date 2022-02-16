import xarray as xr
import dask
import torch
import numpy as np
from numbers import Number
import pandas as pd

def transform(mask):
    mask = torch.from_numpy(np.array(mask)).float()
    return mask

def get_train_val_split(dataset):
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train, val = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train,val

def create_dataset(X,y):
    return list(zip(X,y))

def get_spatial_region_of_interest(data_array: xr.DataArray, x_index_at_center: Number, y_index_at_center: Number
) -> xr.DataArray:
    # Get the index into x and y nearest to x_center_osgb and y_center_osgb:
    # x_index_at_center = np.searchsorted(data_array.x_osgb.values, x_center_osgb) - 1
    # y_index_at_center = np.searchsorted(data_array.y_osgb.values, y_center_osgb) - 1
    # Put x_index_at_center and y_index_at_center into a pd.Series so we can operate
    # on them both in a single line of code.
    x_and_y_index_at_center = pd.Series(
        {"x_osgb": x_index_at_center, "y_osgb": y_index_at_center}
    )
    half_image_size_pixels = 256 // 2
    min_x_and_y_index = x_and_y_index_at_center - half_image_size_pixels
    max_x_and_y_index = x_and_y_index_at_center + half_image_size_pixels
    suggested_reduction_of_image_size_pixels = (
            max(
                (-min_x_and_y_index.min() if (min_x_and_y_index < 0).any() else 0),
                (max_x_and_y_index.x_osgb - len(data_array.x)),
                (max_x_and_y_index.y_osgb - len(data_array.y)),
            )
            * 2
    )
    if suggested_reduction_of_image_size_pixels > 0:
        new_suggested_image_size_pixels = (
                256 - suggested_reduction_of_image_size_pixels
        )
        raise RuntimeError(
            "Requested region of interest of satellite data steps outside of the available"
            " geographical extent of the Zarr data.  The requested region of interest extends"
            f" from pixel indicies"
            f" x={min_x_and_y_index.x_osgb} to x={max_x_and_y_index.x_osgb},"
            f" y={min_x_and_y_index.y_osgb} to y={max_x_and_y_index.y_osgb}.  In the Zarr data,"
            f" len(x)={len(data_array.x_osgb)}, len(y)={len(data_array.y_osgb)}. Try reducing"
            f" image_size_pixels from {256} to"
            f" {new_suggested_image_size_pixels} pixels."
        )
    data_array = data_array.isel(
        x=slice(min_x_and_y_index.x_osgb, max_x_and_y_index.x_osgb),
        y=slice(min_x_and_y_index.y_osgb, max_x_and_y_index.y_osgb),
    )
    return data_array

def listify(masks):
    return [item for sublist in masks for item in sublist]

def get_batch(data_array):
    k,m = divmod(len(data_array), 25)
    return list(data_array[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(25))

def get_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

#batch - the batch number you want returned.
#get_batch splits dataset into 25 batches (hardcoded, can change this in get_batch method)
#e.g. for each epoch, you can loop range(25) to get each batch
#timesteps - the number of timesteps ahead you want to predict
#i.e. timesteps = 3 will return x array, and corresponding y array, of tensors of size 3
#e.g. x = [[time,time+1,time+2],...], y = [[time+1,time+2,time+3],...]
#might need to cut down on dataset size cause its pretty massive!
def get_data(batch, timesteps):
    SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
    dataset = xr.open_dataset(
        SATELLITE_ZARR_PATH,
        engine="zarr",
        chunks="auto",  # Loads the data as a dask array
    )
    dask.config.set(**{"array.slicing.split_large_chunks": False})
    data_array = dataset["data"]
    data_array = data_array.sortby('time')
    data_array = get_batch(data_array)[batch]
    print(f"""Retrieving batch {batch} of 25""")
    print("Processing...0%")
    X_tensors = [transform(timestep[:-1]) for timestep in data_array]
    print("Processing...25%")
    y_tensors = [transform(timestep[1:]) for timestep in data_array]
    print("Processing...50%")
    X = list(get_chunks(X_tensors,timesteps))
    X = X[:-1]
    print("Processing...75%")
    y = list(get_chunks(y_tensors,timesteps))
    y = y[:-1]
    print("Processing...100%")
    dataset = create_dataset(X, y)
    train, val = get_train_val_split(dataset)
    return train, val