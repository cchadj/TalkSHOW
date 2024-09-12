from abc import ABC

import wandb

from trainer import Trainer

from trainer.Trainer import LossDict
from wandb.wandb_run import Run

from trainer.config import Object

import logging
import numpy as np
from typing import Optional


class TrainerCallback(ABC):
    def __init__(self):
        super().__init__()

    def on_train_begin(self, trainer: Trainer):
        pass

    def on_step_end(self, trainer: Trainer, step_losses: LossDict, step: int) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer, epoch_losses: LossDict, epoch: int) -> None:
        pass


class WandbLogger(TrainerCallback):
    run: Run
    step_print_frequency: int
    epoch_print_frequency: int

    def __init__(
            self,
            wandb_run: Run,
            step_print_frequency: int = 100,
            epoch_print_frequency: int = 5,
    ):
        super().__init__()
        self.run = wandb_run
        self.step_print_frequency = step_print_frequency
        self.epoch_print_frequency = epoch_print_frequency

    @classmethod
    def create_from_config(cls, config: Object, wandb_run: Optional[Run] = None):
        wandb_run = wandb_run or wandb.init()
        print_every: int = config.Log.print_every
        return cls(
            wandb_run,
            epoch_print_frequency=1,
            step_print_frequency=print_every
        )

    def on_step_end(self, trainer: Trainer, step_losses: LossDict, step: int) -> None:
        is_logging_step = step % self.step_print_frequency == 0
        if is_logging_step:
            self.run.log({
                "step": step_losses,
            }, step=step)

    def on_epoch_end(self, trainer: Trainer, epoch_losses: LossDict, epoch: int) -> None:
        if epoch == 0:
            self.run.define_metric("epoch", hidden=True)
            for loss_name in epoch_losses.keys():
                self.run.define_metric(f"epoch_{loss_name}", goal="min", step_metric="epoch")

        global_step = trainer.global_steps
        epoch_losses_log = {f"epoch_{loss_name}": loss for loss_name, loss in epoch_losses.items()}
        is_logging_epoch = epoch % self.epoch_print_frequency == 0
        if is_logging_epoch:
            self.run.log(
                {
                    **epoch_losses_log,
                    "epoch": epoch,
                },
                step=global_step
            )


class ModelCheckpoint(TrainerCallback):
    def __init__(
        self,
        monitor="val_loss",
        save_frequency=1,
        min_delta=0,
        patience=0,
        verbose=0,
        mode="min",
        baseline=None,
        start_from_epoch=0,
        wandb_run: Optional[Run] = None
    ):
        super().__init__()

        self.wandb_run = wandb_run
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.stopped_epoch = 0
        self.best_weights = None
        self.start_from_epoch = start_from_epoch
        self.save_frequency = save_frequency

        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be one of ['min', 'max] not {mode}")

        self.monitor_op = np.less if mode == "min" else np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    @classmethod
    def create_from_config(cls, config: Object, wandb_run: Optional[Run] = None):
        return cls(
            save_frequency=config.Log.save_every,
            wandb_run=wandb_run,
            monitor=config.Log.best_model_loss_metric,
        )

    def on_train_begin(self, trainer: Trainer):
        # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, trainer: Trainer, epoch_losses: LossDict, epoch: int) -> None:
        current = self.get_monitor_value(epoch_losses)
        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return

        generator = trainer.generator
        if self.best_weights is None:
            self.best_weights = generator.state_dict()

        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_weights = generator.state_dict()
            trainer.save_model(epoch, wandb_run=self.wandb_run, save_as_best=True)
            if self.wandb_run:
                self.wandb_run.summary[f"best_{self.monitor}"] = self.best
                self.wandb_run.summary[f"best_{self.monitor}_epoch"] = self.best_epoch

        is_save_epoch = (epoch+1) % self.save_frequency == 0 or (epoch+1) == 30
        if is_save_epoch:
            trainer.save_model(epoch, wandb_run=self.wandb_run)

    def get_monitor_value(self, epoch_losses: LossDict):
        monitor_value = epoch_losses.get(self.monitor, None)
        if monitor_value is None:
            available_losses = ",".join(list(epoch_losses.keys()))
            logging.warning(
                f"ModelCheckpoint conditioned on metric `{self.monitor}` "
                f"which is not available. Available metrics are: {available_losses}",
            )
        return monitor_value

    def _is_improvement(self, monitor_value: float, reference_value: float):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)
