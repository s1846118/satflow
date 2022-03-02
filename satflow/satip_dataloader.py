import dask
from numbers import Number
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, Dataset
import xarray
import numpy as np
import pandas as pd
import gc
import psutil

class CustomDataset(Dataset):
    def __init__(self, images, future_images):
        self.future_images = future_images
        self.images = images

    def __len__(self):
        return len(self.future_images)

    def __getitem__(self, idx):
        future_image = self.future_images[idx]
        image = self.images[idx]
        sample = [image, future_image]
        return sample

class DataModuleClass(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.download_dir = ''
        self.batch_size = 4
        self.pin_memory: bool = True
        self.num_workers = 32
        self.configuration_filename="satflow/configs/masks.yaml"

    def listify(self, masks):
        return [item for sublist in masks for item in sublist]

    def transform(self, mask):
        mask = torch.from_numpy(np.array(mask)).float()
        return mask

    def prepare_data(self):
        SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
        dataset = xarray.open_dataset(
            SATELLITE_ZARR_PATH,
            engine="zarr",
            chunks="auto",
        )
        dask.config.set(**{"array.slicing.split_large_chunks": False})
        data_array = dataset["data"]
        self.data_array = data_array.sortby('time')

    def get_spatial_region_of_interest(self, data_array, x_index_at_center: Number,
                                       y_index_at_center: Number
                                       ) -> xarray.DataArray:
        x_and_y_index_at_center = pd.Series(
            {"x": x_index_at_center, "y": y_index_at_center}
        )
        half_image_size_pixels = 256 // 2
        min_x_and_y_index = x_and_y_index_at_center - half_image_size_pixels
        max_x_and_y_index = x_and_y_index_at_center + half_image_size_pixels
        suggested_reduction_of_image_size_pixels = (
                max(
                    (-min_x_and_y_index.min() if (min_x_and_y_index < 0).any() else 0),
                    (max_x_and_y_index.x - len(data_array.x)),
                    (max_x_and_y_index.y - len(data_array.y)),
                )
                * 2
        )
        if suggested_reduction_of_image_size_pixels > 0:
            new_suggested_image_size_pixels = (
                256 - suggested_reduction_of_image_size_pixels
        )
        data_array = data_array.isel(
            x=slice(min_x_and_y_index.x, max_x_and_y_index.x),
            y=slice(min_x_and_y_index.y, max_x_and_y_index.y),
        )
        return data_array

    def setup(self, stage=None):
        self.data_array = self.data_array[:100000]
        gc.collect()
        regions = []
        centers = [(512,512)]
        for (x_osgb,y_osgb) in centers:
            regions.append(self.get_spatial_region_of_interest(self.data_array,x_osgb,y_osgb))
        print("Processing...0%")
        print(str(psutil.virtual_memory()))
        X_tensors = [self.transform(timestep[:-1]) for timestep in regions]
        print(str(psutil.virtual_memory()))
        X_tensors = self.listify(X_tensors)
        print(str(psutil.virtual_memory()))
        print("Processing...25%")
        y_tensors = [self.transform(timestep[1:]) for timestep in regions]
        print(str(psutil.virtual_memory()))
        y_tensors = self.listify(y_tensors)
        print(str(psutil.virtual_memory()))
        print("Processing...50%")
        X_tensors = [torch.reshape(t,[1,256, 256]) for t in X_tensors]
        print(str(psutil.virtual_memory()))
        y_tensors = [torch.reshape(t,[1,256, 256]) for t in y_tensors]
        print(str(psutil.virtual_memory()))
        X_t = list(zip(*[iter(X_tensors)]*5))
        print(str(psutil.virtual_memory()))
        X_t = [torch.stack(x) for x in X_t][:-1]
        print(str(psutil.virtual_memory()))
        #X = [torch.reshape(x, [1,5,1,890, 1843]) for x in X_t][:-1]
        print("Processing...75%")
        y_t = list(zip(*[iter(y_tensors)]*5))
        print(str(psutil.virtual_memory()))
        y_t = [torch.stack(y) for y in y_t][:-1]
        print(str(psutil.virtual_memory()))
        print("Processing...100%")
        dataset = CustomDataset(X_t,y_t)
        #dataset = list(zip(X,y))
        train_size = int(0.8 * len(y_t))
        print(train_size)
        val_size = len(y_t) - train_size
        print(val_size)
        self.train_data, self.val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        gc.collect()
        return DataLoader(self.train_data, batch_size=self.batch_size, pin_memory=True, drop_last=False)

    def val_dataloader(self):
        gc.collect()
        return DataLoader(self.val_data, batch_size=self.batch_size, pin_memory=True, drop_last=False)

