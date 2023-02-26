import pytest
import torch

from model.waveunet import DownSampling, UpSampling, WaveUNet


def test_downsampling():
    t1 = torch.Tensor(1, 1, 1024)
    layer = DownSampling(in_ch=1, out_ch=24, kernel_size=15)

    out = layer(t1)
    assert out.shape == (1, 24, 1024)


@pytest.mark.parametrize(
    "t1, t2",
    [
        (torch.Tensor(1, 24, 24), torch.Tensor(1, 36, 48)),
        (torch.Tensor(1, 12, 21), torch.Tensor(1, 18, 43)),
    ],
)
def test_upsampling(t1, t2):
    layer = UpSampling(in_ch=t1.shape[1] + t2.shape[1], out_ch=12, kernel_size=5)

    out = layer(t1, t2)
    assert out.shape == (1, 12, t2.shape[2])


@pytest.mark.parametrize("n_src", (2, 4))
def test_waveunet(n_src):
    t1 = torch.Tensor(8, 1, 16384)
    model = WaveUNet(n_level=12, n_source=n_src)
    out = model(t1)

    assert out.shape == torch.Size([8, n_src, 16384])
