from colorama import Fore
import torch
from lightning.pytorch.callbacks import Callback


class MyPrintingCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 10 == 0:
            metrics = trainer.callback_metrics

            train_metrics_list = []
            val_metrics_list = []
            for k, v in metrics.items():
                if "train_" in k:
                    train_metrics_list.append(f"{k}: {v:.4f}")
                if "val_" in k:
                    val_metrics_list.append(f"{k}: {v:.4f}")

            train_metrics_print = " ".join(train_metrics_list)
            val_metrics_print = " ".join(val_metrics_list)

            print(f"{Fore.YELLOW}Epoch: {trainer.current_epoch} - "
                  f"{Fore.BLUE}Train: {train_metrics_print} - "
                  f"{Fore.GREEN}Val: {val_metrics_print}")


class OverrideEpochStepCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer, pl_module):
        pl_module.log("step", torch.tensor(trainer.current_epoch, dtype=torch.float32))
