import torch
from abc import ABC
from lightning import LightningModule
from torch import Tensor

from .downstream import LinearProbingClassifier


class SSLBase(LightningModule, ABC):
    """Base model for Self-Supervised Learning (SSL), encapsulates downstream evaluation hooks."""

    def __init__(self, learning_rate: float, weight_decay: float, out_channels: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.downstream_model = LinearProbingClassifier(out_channels=out_channels)

    def on_validation_epoch_start(self) -> None:
        """Before computing reprs and scores for validation set,
        updates downstream model with train reprs.
        """
        self._update_downstream_model_with_train_representations()

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        """Computes representations of single validation set batch."""
        x, y = batch
        z = self.forward_repr(x)
        self.downstream_model.update_test(z, y)

    def on_validation_epoch_end(self) -> None:
        """Fits downstream model on train set and scores on validation set."""
        self.downstream_model.fit()
        metrics = self.downstream_model.score(metric_prefix="val/")
        self.log_dict(metrics)

    def on_test_epoch_start(self) -> None:
        """Before computing reprs and scores for test set,
         updates downstream model with train reprs.
        """
        self._update_downstream_model_with_train_representations()

    def test_step(self, batch: Tensor, batch_idx: int) -> None:
        """Computes representations of single test set batch."""
        x, y = batch
        z = self.forward_repr(x)
        self.downstream_model.update_test(z, y)

    def on_test_epoch_end(self) -> None:
        """Fits downstream model on train set and scores on test set."""
        self.downstream_model.fit()
        metrics = self.downstream_model.score(metric_prefix="test/")
        self.log_dict(metrics)

    def _update_downstream_model_with_train_representations(self) -> None:
        """Resets state of the downstream model and computes representations of the train set."""
        self.downstream_model.reset()

        for batch in self.trainer.datamodule.train_dataloader():  # type: ignore
            # Iterating data_loader manually requires manual device change:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            z = self.forward_repr(x)
            self.downstream_model.update_train(z, y)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Sets up the optimizer for the model."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
