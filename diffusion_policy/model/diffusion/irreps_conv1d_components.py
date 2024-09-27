import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.equiformer_v2.activation import SeparableS2Activation
from diffusion_policy.model.equiformer_v2.layer_norm import get_normalization_layer


# from einops.layers.torch import Rearrange


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class IrrepConv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, SO3_grid,
                 norm=True, max_lmax=3, scale=1, activation=True):
        super().__init__()
        d_irrep = (max_lmax + 1) ** 2

        # construct irrep conv weight based on direct sum
        weight = torch.zeros(out_channels, d_irrep, inp_channels, d_irrep, kernel_size)
        dense_weight = nn.Conv1d(inp_channels, out_channels * (max_lmax+1), kernel_size).weight
        dense_weight = dense_weight.reshape((max_lmax+1), out_channels, inp_channels, kernel_size)
        bias = torch.zeros(out_channels * d_irrep)
        bias[::d_irrep] = nn.Conv1d(1, out_channels, kernel_size).bias
        l_start, l_end = 0, 0
        for l in range(max_lmax+1):
            l_order = 2 * l + 1
            l_end += l_order
            # ToDo: check 'weight'
            weight[:, l_start+torch.arange(l_order).long(), :, l_start+torch.arange(l_order).long(), :] = \
                dense_weight[l, :, :, :].unsqueeze(0)
            l_start = l_end
        weight = weight.reshape(out_channels * d_irrep, inp_channels * d_irrep, kernel_size)
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)
        self.kernel_size, self.padding = kernel_size, kernel_size // 2
        self.out_channels = out_channels
        self.activation = activation
        if norm:
            self.norm = get_normalization_layer('rms_norm_sh', lmax=max_lmax, num_channels=out_channels)
        else:
            self.norm = nn.Identity()
        if activation:
            self.SO3_grid = SO3_grid
            self.gating_linear = torch.nn.Linear(self.out_channels, self.out_channels)
            self.s2_act = SeparableS2Activation(max_lmax, max_lmax)
        self.scale = scale

    def forward(self, x):
        # x in shape (B, (C, irrep), n_pts)
        assert x.dim() == 3
        h = nn.functional.conv1d(x, self.weight, self.bias, padding=self.padding)
        h = einops.rearrange(h, 'b (c i) n -> (b n) i c', c=self.out_channels)
        h = self.norm(h)
        if self.activation:
            gating_scalars = self.gating_linear(h.narrow(1, 0, 1))  # This is different from Equiformer
            h = self.s2_act(gating_scalars, h, self.SO3_grid)
        h = einops.rearrange(h, '(b n) i c -> b (c i) n', n=x.shape[-1])
        if self.scale != 1:
            h = nn.functional.interpolate(h, scale_factor=self.scale)
        return h
