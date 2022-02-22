import dask
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, Dataset
import xarray
import numpy as np

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
        self.batch_size = 32
        self.pin_memory: bool = True
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

    def setup(self, stage=None):
        self.data_array = self.data_array[:50]
        print("Processing...0%")
        X_tensors = [self.transform(timestep[:-1]) for timestep in self.data_array]
        print("Processing...25%")
        y_tensors = [self.transform(timestep[1:]) for timestep in self.data_array]
        print("Processing...50%")
        X_tensors = [torch.reshape(t,[1,890, 1843]) for t in X_tensors]
        y_tensors = [torch.reshape(t,[1,890, 1843]) for t in y_tensors]
        X_t = list(zip(*[iter(X_tensors)]*5))
        X_t = [torch.stack(x) for x in X_t][:-1]
        #X = [torch.reshape(x, [1,5,1,890, 1843]) for x in X_t][:-1]
        print("Processing...75%")
        y_t = list(zip(*[iter(y_tensors)]*5))
        y_t = [torch.stack(y) for y in y_t][:-1]
        print("Processing...100%")
        dataset = CustomDataset(X_t,y_t)
        #dataset = list(zip(X,y))
        train_size = int(0.8 * len(y_t))
        val_size = len(y_t) - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, pin_memory=True, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, pin_memory=True, drop_last=False)

