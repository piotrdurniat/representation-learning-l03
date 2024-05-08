from typing import Any

import torch
from sklearn.linear_model import LogisticRegression
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class LinearProbingClassifier:
    def __init__(self, out_channels: int, **kwargs: Any) -> None:
        self.out_channels = out_channels
        self.model = LogisticRegression(solver="liblinear", **kwargs)

        self._z_train: list[Tensor] = []
        self._y_train: list[Tensor] = []
        self._z_test: list[Tensor] = []
        self._y_test: list[Tensor] = []

    @property
    def z_train(self) -> Tensor:
        return torch.cat(self._z_train, dim=0)

    @property
    def y_train(self) -> Tensor:
        return torch.cat(self._y_train, dim=0)

    @property
    def z_test(self) -> Tensor:
        return torch.cat(self._z_test, dim=0)

    @property
    def y_test(self) -> Tensor:
        return torch.cat(self._y_test, dim=0)

    def update_train(self, z: Tensor, y: Tensor) -> None:
        self._z_train.append(z.detach().cpu())
        self._y_train.append(y.detach().cpu())

    def update_test(self, z: Tensor, y: Tensor) -> None:
        self._z_test.append(z.detach().cpu())
        self._y_test.append(y.detach().cpu())

    def reset(self) -> None:
        self._z_train = []
        self._y_train = []
        self._z_test = []
        self._y_test = []

    def fit(self) -> None:
        self.model.fit(self.z_train, self.y_train)

    def score(self, metric_prefix: str = "") -> dict[str, Tensor]:
        y_score = torch.tensor(self.model.predict_proba(self.z_test))

        metrics = self._get_metrics(metric_prefix)
        metric_vals = metrics(y_score, self.y_test)

        return metric_vals

    def _get_metrics(self, metric_prefix: str) -> MetricCollection:
        assert self.out_channels > 1
        metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=self.out_channels),
                "f1": MulticlassF1Score(num_classes=self.out_channels, average="macro"),
            }
        )
        return metrics.clone(prefix=metric_prefix)
