import pytest

import torch
from model.metric import sdr_allsep, sdr_vocsep
from torchmetrics.audio import SignalDistortionRatio as SDR


def test_sdr_vocsep():
    pred = torch.randn((8, 2, 1000))
    target = torch.randn((8, 2, 1000))
    sdr_custom = sdr_vocsep(pred, target)
    sdr_real = SDR()(pred[:, 0, :], target[:, 0, :])
    
    assert sdr_custom[0] / sdr_custom[2] == pytest.approx(sdr_real)
    
    pred[2, 0] = torch.zeros((1000,))
    target[2, 0] = torch.zeros((1000,))
    pred[4, 0] = torch.zeros((1000,))
    target[4, 0] = torch.zeros((1000,))
    pred[6, 0] = torch.zeros((1000,))
    target[6, 0] = torch.zeros((1000,))
    pred[2, 1] = torch.zeros((1000,))
    target[2, 1] = torch.zeros((1000,))
    pred[4, 1] = torch.zeros((1000,))
    target[4, 1] = torch.zeros((1000,))
    pred[6, 1] = torch.zeros((1000,))
    target[6, 1] = torch.zeros((1000,))

    sdr_custom = sdr_vocsep(pred, target)
    assert sdr_custom[1].size() == torch.Size([5])
    assert sdr_custom[2] == 5
    
