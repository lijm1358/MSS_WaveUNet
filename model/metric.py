"""
Copyright 2020-2022 Lightning-AI team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file(class) has been modified by lijm1358(JongMok Lee) for
additional loss output
"""

from typing import Tuple

from torch import Tensor, flatten
from torchmetrics.audio import SignalDistortionRatio
from torchmetrics.functional.audio.sdr import signal_distortion_ratio


class ModifiedSDR(SignalDistortionRatio):
    """Modiifed version of `SignalDistortionRation` in "TorchMetrics"."""

    def update(self, preds: Tensor, target: Tensor) -> None:
        sdr_batch = signal_distortion_ratio(
            preds,
            target,
            self.use_cg_iter,
            self.filter_length,
            self.zero_mean,
            self.load_diag,
        )

        self.sdr_batch = sdr_batch
        self.sum_sdr += sdr_batch.sum()

    def compute(self) -> Tuple[Tensor, Tensor]:
        """returns SDR sum and Tensor of batch's SDR"""
        return self.sum_sdr, self.sdr_batch

def sdr_allsep(y_pred, y_target, device) -> Tuple[float, list, int]:
    """Calculate the sdr loss of batch.

    Each return value will be used for the average value of accompanies' loss, and median of vocal loss.

    Args:
        y_target: original target batch
        y_pred: prediction of y_target
        loss_fn: loss function(SDR)

    Returns:
        float: the sum of accompanies' loss
        list(Tensor): the list of Tensor of calculated vocal loss
        int: the number of accmpanies' loss
    """
    loss_fn = ModifiedSDR().to(y_pred.device)
    y_target_acc = y_target[:, 0:3]
    y_target_acc = flatten(y_target_acc, end_dim=1)
    y_pred_acc = y_pred[:, 0:3]
    y_pred_acc = flatten(y_pred_acc, end_dim=1)
    
    y_target_voc = y_target[:, 3]
    y_pred_voc = y_pred[:, 3]

    # remove tensor element only consists of zero.
    mask = (y_target_acc != 0).any(dim=1)
    y_target_acc = y_target_acc[mask]
    y_pred_acc = y_pred_acc[mask]

    mask = (y_target_voc != 0).any(dim=1)
    y_target_voc = y_target_voc[mask]
    y_pred_voc = y_pred_voc[mask]

    loss_acc = loss_fn(y_pred_acc, y_target_acc)[0].item()
    loss_voc = loss_fn(y_pred_voc, y_target_voc)[1]

    return loss_acc, loss_voc, y_target_acc.shape[0]

def sdr_vocsep(y_pred, y_target):
    loss_fn = ModifiedSDR().to(y_pred.device)
    y_target_acc = y_target[:, 1]
    y_pred_acc = y_pred[:, 1]
    
    y_target_voc = y_target[:, 0]
    y_pred_voc = y_pred[:, 0]
    
    mask = (y_target_acc != 0).any(dim=1)
    y_target_acc = y_target_acc[mask]
    y_pred_acc = y_pred_acc[mask]
    
    mask = (y_target_voc != 0).any(dim=1)
    y_target_voc = y_target_voc[mask]
    y_pred_voc = y_pred_voc[mask]
    
    loss_acc = loss_fn(y_pred_acc, y_target_acc)[0].item()
    loss_voc = loss_fn(y_pred_voc, y_target_voc)[1]
    
    return loss_acc, loss_voc, y_target_acc.shape[0]