import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        squared_error = (y_pred - y_true) ** 2
        sample_losses = np.mean(squared_error, axis=-1)
        return Tensor(np.mean(sample_losses))

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs
        grad_pred = 2 * (y_pred - y_true) / y_pred.size
        grad_true = np.zeros_like(y_true)
        return [Tensor(grad_pred), Tensor(grad_true)]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        p = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -np.sum(y_true * np.log(p), axis=-1)
        return Tensor(np.mean(sample_losses))

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs
        batch_size = y_pred.shape[0]
        p = np.clip(y_pred, 1e-7, 1 - 1e-7)
        grad = -y_true / p / batch_size
        return [Tensor(grad), Tensor(np.zeros_like(y_true))]
