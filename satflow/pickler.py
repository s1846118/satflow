import xarray
import numpy as np
from numbers import Number
import pandas as pd
import torch
from itertools import chain
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
import os
import pathlib
import sys
import gc
import dask
import pickle

def to_tensor(dataset):
        return torch.from_numpy(np.array(dataset.variable))

def get_spatial_region_of_interest(data_array, x_index_at_center: Number,
                                   y_index_at_center: Number
                                   ) -> xarray.DataArray:
    x_and_y_index_at_center = pd.Series(
        {"x": x_index_at_center, "y": y_index_at_center}
    )
    half_image_size_pixels = 256 // 2
    min_x_and_y_index = x_and_y_index_at_center - half_image_size_pixels
    max_x_and_y_index = x_and_y_index_at_center + half_image_size_pixels
    data_array = data_array.isel(
            x=slice(min_x_and_y_index.x, max_x_and_y_index.x),
            y=slice(min_x_and_y_index.y, max_x_and_y_index.y),
        )
    return data_array

SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
dataset = xarray.open_dataset(
    SATELLITE_ZARR_PATH,
    engine="zarr",
    chunks="auto",
)
dask.config.set(**{"array.slicing.split_large_chunks": False})
data_array = dataset["data"]
data_array = data_array.sortby('time')
data_array = data_array[:20]
gc.collect()
regions = []
centers = [(512,512)]
for (x_osgb,y_osgb) in centers:
    regions.append(get_spatial_region_of_interest(data_array,x_osgb,y_osgb))
print("Processing...0%")
X_tensors = [to_tensor(timestep[:-1]) for timestep in regions]
X_tensors = list(chain.from_iterable(X_tensors))
print("Processing...25%")
y_tensors = [to_tensor(timestep[1:]) for timestep in regions]
y_tensors = list(chain.from_iterable(y_tensors))
print("Processing...50%")
X_tensors = [torch.reshape(t,[1,256, 256]) for t in X_tensors]
y_tensors = [torch.reshape(t,[1,256, 256]) for t in y_tensors]
X_t = list(zip(*[iter(X_tensors)]*5))
X_t = [torch.stack(x) for x in X_t][:-1]
print("Processing...75%")
y_t = list(zip(*[iter(y_tensors)] * 5))
y_t = [torch.stack(y) for y in y_t][:-1]
print("Processing...100%")
gc.collect()
data = [X_t,y_t]

with open('data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
