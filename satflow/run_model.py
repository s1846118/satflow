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
from nowcasting_utils.models.base import BaseModel
import gc
import torch
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
import sys
import gc
import dask
import pickle
import sys

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

if __name__ == '__main__':   
	print(torch.__version__)
	torch.multiprocessing.set_sharing_strategy("file_system")
	#torch.cuda.memory_summary(device=None, abbreviated=False)
	print("Pickling...")
	with open('data.pickle', 'rb') as handle:
		data = pickle.load(handle)
	X,y = data
	dataset = SimpleDataset(X,y)
	train_size = int(0.9 * dataset.__len__())
	val_size = int(dataset.__len__() - train_size)
	print(f"""Train size = {train_size}""")
	print(f"""Val size = {val_size}""")
	train, val = torch.utils.data.random_split(dataset, [train_size, val_size])
	trainer = pl.Trainer(strategy="ddp_spawn", gpus = 2)
	log = utils.get_logger(__name__)
	model = conv_lstm.EncoderDecoderConvLSTM(input_channels = 1, out_channels = 1, forecast_steps = 5)
	log.info("Starting training!")
	torch.cuda.empty_cache()
	trainer.fit(model, DataLoader(train, num_workers=0, batch_size = 16, pin_memory = True), DataLoader(val, num_workers=0, batch_size = 16, pin_memory = True))
	torch.save(model.state_dict(), "model.pth")
