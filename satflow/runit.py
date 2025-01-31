from typing import List
import models.conv_lstm as conv_lstm
from pytorch_lightning import (
    Callback,
    LightningModule,
    LightningDataModule,
    Trainer,
    seed_everything,
)
from core import utils
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import gc

import xarray as xr
import numpy as np
from numbers import Number
import pandas as pd
import torch
from itertools import chain
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
import os
import pathlib
import gc
import dask
import pickle
import sys
from pytorch_lightning.callbacks import ModelCheckpoint

class SimpleDataset(Dataset):
    def __init__(self, images, future_images):
        self.future_images = future_images
        self.images = images

    def __len__(self):
        return len(self.future_images)
    def __getitem__(self, idx):
        future_image = self.future_images[idx]
        image = self.images[idx]
        return image, future_image

def to_tensor(dataset):
    return torch.from_numpy(np.array(dataset.variable)).float()

def get_spatial_region_of_interest(data_array, x_index_at_center: Number, y_index_at_center: Number) -> xr.DataArray:
    x_and_y_index_at_center = pd.Series({"x_osgb": x_index_at_center, "y_osgb": y_index_at_center})
    half_image_size_pixels = 256 // 2
    min_x_and_y_index = x_and_y_index_at_center - half_image_size_pixels
    max_x_and_y_index = x_and_y_index_at_center + half_image_size_pixels
    data_array = data_array.isel(x=slice(min_x_and_y_index.x_osgb, max_x_and_y_index.x_osgb),
                                    y=slice(min_x_and_y_index.y_osgb, max_x_and_y_index.y_osgb))
    return data_array

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy("file_system")
    model = conv_lstm.EncoderDecoderConvLSTM(input_channels = 1, out_channels = 1, forecast_steps = 5)
    new_epochs = 1000
    checkpoint_callback = ModelCheckpoint(
        dirpath='./new_model/lightning_logs/version_0/checkpoints/',
        filename='checky', save_last=True)
    trainer = pl.Trainer(strategy="ddp_spawn", gpus=[0], max_epochs=new_epochs, enable_checkpointing=True,
                         callbacks=[checkpoint_callback])
    i = 0
    SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
    data = xr.open_dataset(
        SATELLITE_ZARR_PATH,
        engine="zarr",
        chunks="auto",
    )
    dask.config.set(**{"array.slicing.split_large_chunks": False})
    data_array = data["data"]
    data_array = data_array.sortby('time')
    #data_array = data_array[:1000]
    data_array = data_array[119:206]
    gc.collect()
    regions = []
    centers = [(512, 512)]
    for (x_osgb, y_osgb) in centers:
        regions.append(get_spatial_region_of_interest(data_array, x_osgb, y_osgb))
    X_tensors = [to_tensor(timestep[:-1]) for timestep in regions]
    X_tensors = list(chain.from_iterable(X_tensors))
    y_tensors = [to_tensor(timestep[1:]) for timestep in regions]
    y_tensors = list(chain.from_iterable(y_tensors))
    X_tensors = [torch.reshape(t, [1, 256, 256]) for t in X_tensors]
    y_tensors = [torch.reshape(t, [1, 256, 256]) for t in y_tensors]
    X_t = list(zip(*[iter(X_tensors)] * 5))
    X_t = [torch.stack(x) for x in X_t][:-1]
    y_t = list(zip(*[iter(y_tensors)] * 5))
    y_t = [torch.stack(y) for y in y_t][:-1]
    dataset = SimpleDataset(X_t, y_t)
    train_size = int(0.9 * dataset.__len__())
    val_size = int(dataset.__len__() - train_size)
    print(f"""Train size = {train_size}""")
    print(f"""Val size = {val_size}""")
    train, val = torch.utils.data.random_split(dataset, [train_size, val_size])
    log = utils.get_logger(__name__)
    log.info("Starting training!")
    trainer.fit(model, DataLoader(train, num_workers=0, batch_size=3), DataLoader(val, num_workers=0, batch_size=3))
    torch.save(model.state_dict(), "./new_model/model.pth")


