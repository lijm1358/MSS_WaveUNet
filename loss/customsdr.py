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

from torch import Tensor
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
