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
        sample = {"Images": image, "FutureImage": future_image}
        return sample

class DataModuleClass(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.download_dir = ''
        self.batch_size = 32
        self.num_workers: int = 8,
        self.pin_memory: bool = True,
        self.configuration_filename="satflow/configs/masks.yaml",

    def get_chunks(self,l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

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
        self.data_array = self.data_array[:300]
        print("Processing...0%")
        X_tensors = [self.transform(timestep[:-1]) for timestep in self.data_array]
        print("Processing...25%")
        y_tensors = [self.transform(timestep[1:]) for timestep in self.data_array]
        print("Processing...50%")
        X = list(self.get_chunks(X_tensors, 5))
        X = [torch.stack(x) for x in X]
        X = [torch.reshape(x, [1, 5, 890, 1843]) for x in X][:-1]
        print("Processing...75%")
        y = list(self.get_chunks(y_tensors, 5))
        y = [torch.stack(y) for y in y]
        y = [torch.reshape(y, [1, 5, 890, 1843]) for y in y][1:]
        print("Processing...100%")
        # dataset = CustomDataset(X,y)
        dataset = list(zip(X,y))
        train_size = int(0.8 * len(y))
        test_size = int(0.1 * len(y))
        val_size = len(y) - train_size - test_size
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, pin_memory=True, drop_last=False, num_workers = 8)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, pin_memory=True, drop_last=False, num_workers = 8)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, pin_memory=True, drop_last=False, num_workers = 8)

