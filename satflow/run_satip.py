from typing import List
import satflow.models.conv_lstm as conv_lstm
import satflow.satip_dataloader as datamodule
from pytorch_lightning import (
    Callback,
    Trainer,
)
from satflow.core import utils
from pytorch_lightning.callbacks import LearningRateMonitor
dataloader = datamodule.DataModuleClass()
log = utils.get_logger(__name__)
model = conv_lstm.EncoderDecoderConvLSTM(input_channels = 1, out_channels = 1, forecast_steps = 5)
trainer = Trainer()
lr_monitor = LearningRateMonitor(logging_interval="step")
callbacks: List[Callback] = [lr_monitor]
log.info("Starting training!")
trainer.fit(model=model, datamodule=dataloader)
log.info("Starting tuning!")
trainer.tune(model=model, datamodule=dataloader)
log.info("Starting testing!")
trainer.test(model=model, datamodule=dataloader)
log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")