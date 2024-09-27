from typing import Union
import logging

import numpy as np
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.irreps_conv1d_components import (
    Downsample1d, Upsample1d, IrrepConv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.equiformer_v2.equiformerv2_block import FeedForwardNetwork
from diffusion_policy.model.common.se3_transformation import rot_pcd
from diffusion_policy.model.equiformer_v2.module_list import ModuleListInfo
from diffusion_policy.model.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)

logger = logging.getLogger(__name__)


class IrrepConditionalResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=True,
                 max_lmax=3,
                 SO3_grid=None):
        super().__init__()
        self.max_lmax = max_lmax
        self.blocks = nn.ModuleList([
            IrrepConv1dBlock(in_channels, out_channels, kernel_size, SO3_grid, norm=True, max_lmax=max_lmax),
            IrrepConv1dBlock(out_channels, out_channels, kernel_size, SO3_grid, norm=True, max_lmax=max_lmax),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        self.cond_encoder = IrrepConv1dBlock(cond_dim, 2 * out_channels, 1, SO3_grid,
                                             norm=False, max_lmax=max_lmax, activation=False)
        # self.cond_encoder = nn.Sequential(
        #     nn.Mish(),
        #     nn.Linear(cond_dim, out_channels * 2),
        #     Rearrange('batch t -> batch t 1'),
        # )

        # make sure dimensions compatible
        self.residual_conv = IrrepConv1dBlock(in_channels, out_channels, 1, SO3_grid, norm=False,
                                              max_lmax=max_lmax, activation=False) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : (B, (Cin1, irrep), n_pts)
            cond : (B, (Cin2, irrep), 1)

            returns:
            out : (B, (Cout, irrep), n_pts)
        '''
        bs, npts = x.shape[0], x.shape[-1]
        out = self.blocks[0](x).reshape(bs, self.out_channels, -1, npts)
        embed = self.cond_encoder(cond).reshape(bs, 2 * self.out_channels, -1, 1)  # (B, 2*Cout, irrep, n_pts)

        # FiLM conditioning for each irrep type l
        l_start, l_end = 0, 0
        outs = []
        for l in range(self.max_lmax + 1):
            l_order = 2 * l + 1
            l_end += l_order
            l_scale, l_bias = embed[:, ::2, l_start:l_end, :], embed[:, 1::2, l_start:l_end, :]
            l_out = out[:, :, l_start:l_end, :]
            length = (l_out * l_scale).sum(dim=2, keepdim=True)
            direction = l_out / torch.norm(l_out, p=2, dim=2, keepdim=True)
            l_out = length * direction + l_bias
            outs.append(l_out)
            l_start = l_end
        out = torch.cat(outs, dim=2)

        out = out.reshape(bs, -1, npts)
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class IrrepConditionalUnet1D(nn.Module):
    def __init__(self,
                 input_dim,
                 local_cond_dim=None,
                 global_cond_dim=None,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=True,
                 max_lmax=3,
                 grid_resolution=12
                 ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # local_cond_encoder = None
        # if local_cond_dim is not None:
        #     _, dim_out = in_out[0]
        #     dim_in = local_cond_dim
        #     local_cond_encoder = nn.ModuleList([
        #         # down encoder
        #         nn.Conv1d(dim_in, dim_out, kernel_size, padding=kernel_size//2),
        #         # nn.Conv1d(dim_in, dim_out, kernel_size, padding=kernel_size//2),
        #         # # up encoder
        #         # nn.Conv1d(dim_in, dim_out, kernel_size, padding=kernel_size//2),
        #         # nn.Conv1d(dim_in, dim_out, kernel_size, padding=kernel_size//2),
        #     ])

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max_lmax, max_lmax))
        for l in range(max_lmax + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max_lmax + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l,
                        m,
                        resolution=grid_resolution,
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            IrrepConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
                max_lmax=max_lmax, SO3_grid=self.SO3_grid
            ),
            IrrepConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
                max_lmax=max_lmax, SO3_grid=self.SO3_grid
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            scale = 1 if is_last else 0.5
            down_modules.append(nn.ModuleList([
                IrrepConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    max_lmax=max_lmax, SO3_grid=self.SO3_grid),
                IrrepConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    max_lmax=max_lmax, SO3_grid=self.SO3_grid),
                IrrepConv1dBlock(dim_out, dim_out, 3, self.SO3_grid,
                                 norm=False, max_lmax=max_lmax, scale=scale)
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            scale = 1 if is_last else 2
            up_modules.append(nn.ModuleList([
                IrrepConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    max_lmax=max_lmax, SO3_grid=self.SO3_grid),
                IrrepConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    max_lmax=max_lmax, SO3_grid=self.SO3_grid),
                IrrepConv1dBlock(dim_in, dim_in, 3, self.SO3_grid,
                                 norm=False, max_lmax=max_lmax, scale=scale)
            ]))

        final_conv = IrrepConv1dBlock(start_dim, input_dim, 3, self.SO3_grid,
                                      norm=False, max_lmax=max_lmax, activation=False)

        self.diffusion_step_encoder = diffusion_step_encoder
        # self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        self.d_irrep = (max_lmax + 1) ** 2
        self.diffusion_step_embed_dim = diffusion_step_embed_dim

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                local_cond=None, global_cond=None, **kwargs):
        """
        sample: (B, T, (C, irrep))
        timestep: (B,) or int, diffusion step
        local_cond: (B, T, (C, irrep))
        global_cond: (B, (C, irrep))
        output: (B, T, (C, irrep))
        """
        bs, n_hist, _ = sample.shape
        sample = einops.rearrange(sample, 'b h c -> b c h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = torch.zeros(bs, self.diffusion_step_embed_dim, self.d_irrep).to(sample.device)
        global_feature[:, :, 0] = self.diffusion_step_encoder(timesteps)  # (B, dsed, irrep)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond.reshape(bs, -1, self.d_irrep)], axis=1)
            global_feature = global_feature.reshape(bs, -1, 1)
            # ToDo: check the feature

        # encode local features
        h_local = list()
        assert local_cond is None, "local cond is not implemented"
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            for conv in self.local_cond_encoder:
                x = conv(local_cond)
                h_local.append(x)
        all = [sample, ]
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            all.append(x)
            if idx == 0 and len(h_local) > 0:
                # x = x * h_local[0] + h_local[1]
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            all.append(x)
            h.append(x)
            x = downsample(x)
            all.append(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
            all.append(x)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            all.append(x)
            x = resnet2(x, global_feature)
            all.append(x)
            x = upsample(x)
            all.append(x)
            # if idx == len(self.up_modules) - 1 and len(h_local) > 0:
            #     x = x * h_local[2] + h_local[3]

        x = self.final_conv(x)  # (B, (C, irrep), n)

        x = einops.rearrange(x, 'b c h -> b h c')
        all = [einops.rearrange(i, 'b c h -> b h c') for i in all]
        return x, all

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    device = 'cuda:0'
    atol = 1e-4
    torch.manual_seed(0)
    np.random.seed(0)
    bs = 16
    max_lmax = 2
    d_irrep = (max_lmax + 1)**2
    trajectory = torch.zeros(bs, 16, 1 * d_irrep)
    trajectory[:, :, 1:4] = torch.rand(bs, 16, 3) - 0.5
    condition = torch.zeros(bs, 1 * d_irrep)
    condition[:, 1:4] = torch.rand(bs, 3) - 0.5
    trajectory, condition = trajectory.to(device), condition.to(device)
    model = IrrepConditionalUnet1D(1,
                                   local_cond_dim=None,
                                   global_cond_dim=1,
                                   diffusion_step_embed_dim=64,
                                   down_dims=[64],
                                   kernel_size=3,
                                   n_groups=8,
                                   cond_predict_scale=True,
                                   max_lmax=max_lmax).to(device)
    c4_xyz_rots = torch.zeros((10, 3)).to(device)
    c4_xyz_rots[1:4, 0] = torch.arange(1, 4) * torch.pi / 2
    c4_xyz_rots[4:7, 1] = torch.arange(1, 4) * torch.pi / 2
    c4_xyz_rots[7:10, 2] = torch.arange(1, 4) * torch.pi / 2
    print('total #param: ', model.num_params)

    with torch.no_grad():
        out, all = model(trajectory, 0, global_cond=condition)  # (B, T, (C, irrep))

    success = True
    for i in range(c4_xyz_rots.shape[0]):
        trajectory_tfm = torch.zeros_like(trajectory)
        trajectory_tfm[..., 1:4] = rot_pcd(trajectory[..., 1:4].clone(), c4_xyz_rots[i])
        condition_tfm = torch.zeros_like(condition)
        condition_tfm[..., 1:4] = rot_pcd(condition[..., 1:4].clone().unsqueeze(1), c4_xyz_rots[i]).squeeze(1)
        out_feats_tfm_after = torch.zeros_like(out)
        out_feats_tfm_after[..., 1:4] = rot_pcd(out[..., 1:4].clone(), c4_xyz_rots[i])
        all_tfm_after = []
        for tensor in all:
            tensor_tfm_after = torch.zeros_like(tensor)
            tensor_tfm_after[..., 1:4] = rot_pcd(tensor[..., 1:4].clone(), c4_xyz_rots[i])
            all_tfm_after.append(tensor_tfm_after)

        with torch.no_grad():
            out_feats_tfm_before, all_tfm_before = model(trajectory_tfm, 0, global_cond=condition_tfm)

        eerr = torch.linalg.norm(out_feats_tfm_before[..., 1:4] - out_feats_tfm_after[..., 1:4], dim=1).max()
        err = torch.linalg.norm(out_feats_tfm_after[..., 1:4] - out[..., 1:4], dim=1).max()
        all_eerr = [torch.round(torch.linalg.norm(i[..., 1:4] - j[..., 1:4], dim=1).max(), decimals=2).item()
                    for i, j in zip(all_tfm_before, all_tfm_after)]
        if not torch.allclose(out_feats_tfm_before[..., 1:4], out_feats_tfm_after[..., 1:4], atol=atol):
            print(f"FAILED on {c4_xyz_rots[i]}: {eerr:.1E} > {atol}, {err}")
            print("all eerr", all_eerr)
            success = False
        else:
            print(f"PASSED on {c4_xyz_rots[i]}: {eerr:.1E} < {atol}, {err}")

    # import matplotlib.pyplot as plt
    # f = plt.figure(figsize=(16, 4))
    # ax = [f.add_subplot(1, 4, i+1, projection='3d') for i in range(4)]
    # plt.show()

    if success:
        print('PASSED')
    print(1)
