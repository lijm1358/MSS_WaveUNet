import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.functional import pad

class DownSampling(nn.Module):
    """Donwsampling block of WaveUNet.

    Args:
        in_ch: the number of input channels
        out_ch: the number of output channels
        kernel_size: kernel size of convolution layer
    """
    def __init__(self, in_ch=1, out_ch=24, kernel_size=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=7),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=7),
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self, x: Tensor):
        x = self.net(x)
        return x

class UpSampling(nn.Module):
    """Upsampling block of WaveUNet.

    Args:
        in_ch: the number of input channels
        out_ch: the number of output channels
        kernel_size: kernel size of convolution layer

    Attributes:
        upsample: upsampling layer to double the input size
        conv: convolution layers with LeakyReLU
    """
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=2),
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self, x, x_back):
        """Forward the upsampling block input.

        Before the forwarding output of upsample layers to convolution layers, concatenates the output of 
        corresponding Donwsampling block with the result of Upsample.
        """
        x = self.upsample(x)
        diff = x_back.shape[-1] - x.shape[-1]
        x = pad(x, (0, diff))
        x = torch.cat([x, x_back], axis=1)
        return self.conv(x)

class WaveUNet(nn.Module):
    """WaveUNet architecture"""
    def __init__(self, n_level=12, n_source=4):
        super().__init__()
        self.level = n_level
        
        layers=[DownSampling(in_ch=1,out_ch=24,kernel_size=15)]
        
        for i in range(self.level-1):
            layers.append(DownSampling(in_ch=24*(i+1),out_ch=24*(i+2),kernel_size=15))
            
        layers.append(DownSampling(in_ch=24*(self.level), out_ch=24*(self.level+1), kernel_size=15))
            
        for i in range(self.level):
            layers.append(UpSampling(in_ch=24*(self.level+1-i) + 24*(self.level - i), out_ch=24*(self.level-i), kernel_size=5))
            
        self.net = nn.ModuleList(layers)
        self.separation = nn.Sequential(
            nn.Conv1d(25, n_source, kernel_size=1),
            nn.Conv1d(n_source, n_source, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x: Tensor):
        layer_to_concat = []
        layer_to_concat.append(x)
        for layer in self.net[0: self.level]:
            x = layer(x)
            layer_to_concat.append(x)
            x = x[:, :, 1::2]
        x = self.net[self.level](x)
        layer_to_concat.append(x)
        for i, layer in enumerate(self.net[self.level+1:]):
            x = layer_to_concat[-1]
            x = layer(x, layer_to_concat[-1-i-1])
            layer_to_concat[-1] = x
            
        x = torch.cat([layer_to_concat[0], x], axis=1)
        x = self.separation(x)

        return x