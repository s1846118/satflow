import logging
import numpy as np
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import math
import einops
import pandas as pd
import torch
import torch.nn.functional as F
import torch_optimizer as optim
from einops import rearrange, repeat
from nowcasting_dataloader.batch import BatchML
from nowcasting_dataset.consts import (
    DEFAULT_N_GSP_PER_EXAMPLE,
    DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
    GSP_DATETIME_INDEX,
    GSP_ID,
    GSP_YIELD,
    NWP_DATA,
    PV_SYSTEM_ID,
    PV_YIELD,
    SATELLITE_DATA,
    TOPOGRAPHIC_DATA,
)
from nowcasting_utils.metrics.validation import (
    make_validation_results,
    save_validation_results_to_logger,
)
from nowcasting_utils.models.base import BaseModel, register_model
from nowcasting_utils.models.loss import get_loss
from nowcasting_utils.visualization.line import plot_batch_results
from nowcasting_utils.visualization.visualization import plot_example
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from transformers import (
    PerceiverConfig,
    PerceiverForImageClassificationLearned,
    PerceiverForMultimodalAutoencoding,
    PerceiverForOpticalFlow,
    PerceiverModel,
)
import torch.nn.functional as F
import torch.nn as nn 
import pytorch_lightning as pl
from datetime import datetime

logger = logging.getLogger("satflow.model")
logger.setLevel(logging.WARN)

class HuggingFacePerceiver(BaseModel):
    def __init__(self, input_size: int = 256):
        super(HuggingFacePerceiver, self).__init__()
        self.model = PerceiverForOpticalFlow.from_pretrained(
            "deepmind/optical-flow-perceiver",
            ignore_mismatched_sizes=True,
            train_size=[input_size, input_size],
        )

        self.layer = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size = 1)
        self.criterion = nn.MSELoss()
        self.i = 0

    def extract_image_patches(self, x, kernel, stride=1, dilation=1):
        # Do TF 'SAME' Padding
        b,c,h,w = x.shape
        h2 = math.ceil(h / stride)
        w2 = math.ceil(w / stride)
        pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
        pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
        x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))

        # Extract patches
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        patches = patches.permute(0,4,5,1,2,3).contiguous()

        return patches.view(b,-1,patches.shape[-2], patches.shape[-1])
   
    def forward(self, x, **kwargs) -> Any:
        return model(inputs=x)

    def _train_or_validate_step(self, batch, batch_idx, is_training: bool = True):
        x, y = batch
        # Now run predictions for all the queries
        # Predicting all future ones at once
        x = x.detach().float()
        y = y.detach().float()
        losses = []
        batch_size, _, C, H, W = x.shape
        patches = self.extract_image_patches(x.view(batch_size*2,C,H,W), 3)
        _, C, H, W = patches.shape
        patches = patches.view(batch_size, -1, C, H, W).float().to(self.model.device) 
        hrv_sat_y_hat = self.model(inputs=patches)
        hrv_sat_y_hat = self.layer(torch.reshape(hrv_sat_y_hat['logits'],[1,2,256,256]))
        # HRV Satellite losses
        hrv_sat_loss = self.criterion(hrv_sat_y_hat, y)
        losses.append(hrv_sat_loss)
        loss = losses[0]
        for sat_loss in losses[1:]:
            loss += sat_loss
        self.log_dict({f"{'train' if is_training else 'val'}/loss": loss})
        f = open("losses.txt", "a")
        f.write(str(loss))
        f.close()
        if is_training:
            return loss
        else:
            # Return the model outputs as well
            return loss

    def configure_optimizers(self):
        if self.i == 0:
            self.i = 1
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
            torch.save(self.optimizer.state_dict(),"../../../../../../../../../disk/scratch/s1827995-new/optim.pt")
            return self.optimizer
        else:
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
            self.optimizer.load_state_dict("../../../../../../../../../disk/scratch/s1827995-new/optim.pt")
            return self.optimizer
   
