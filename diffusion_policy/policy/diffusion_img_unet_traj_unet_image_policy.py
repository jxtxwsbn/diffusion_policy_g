from typing import Dict
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from collections import OrderedDict


class ResBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super(ResBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            # nn.GroupNorm(hidden_dim//16, hidden_dim),
            nn.ReLU(hidden_dim)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            # nn.GroupNorm(hidden_dim // 16, hidden_dim),
        )
        self.relu = nn.ReLU(hidden_dim)
        self.rescale_channel = None
        if input_channels != hidden_dim:
            self.rescale_channel = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            )

    def forward(self, xx):
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.rescale_channel:
            out += self.rescale_channel(residual)
        else:
            out += residual
        out = self.relu(out)
        return out


class ResUNet(torch.nn.Module):
    def __init__(self, n_input_channel=3, n_output_channel=32, n_hidden=32, kernel_size=3,
                 input_shape=(3, 96, 96), h=80, w=80, fixed_crop=False, local_cond_type=''):
        super().__init__()
        self.hid = n_hidden
        self.n_input_channel = n_input_channel
        self.n_output_channel = n_output_channel
        self.n_neck_channel = 8 * self.hid
        self.fixed_crop = fixed_crop
        self.local_cond_type = local_cond_type
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.register_buffer('h', torch.tensor(h))
        self.register_buffer('w', torch.tensor(w))
        self.build()

    def build(self):
        self.crop = dmvc.CropRandomizer(
                                        input_shape=self.input_shape,
                                        crop_height=self.h.item(),
                                        crop_width=self.w.item(),
                                        fixed_crop=self.fixed_crop
                                    )
        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-0', nn.Conv2d(self.n_input_channel, self.hid, kernel_size=self.kernel_size, padding=1)),
            ('enc-e2relu-0', nn.ReLU()),
            ('enc-e2res-1', ResBlock(self.hid, self.hid, kernel_size=self.kernel_size)),
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-2', nn.MaxPool2d(2)),
            ('enc-e2res-2', ResBlock(self.hid, 2 * self.hid, kernel_size=self.kernel_size)),
        ]))
        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.MaxPool2d(2)),
            ('enc-e2res-3', ResBlock(2 * self.hid, 4 * self.hid, kernel_size=self.kernel_size)),
        ]))
        self.conv_down_8 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-4', nn.MaxPool2d(2)),
            ('enc-e2res-4', ResBlock(4 * self.hid, 8 * self.hid, kernel_size=self.kernel_size)),
        ]))
        self.conv_down_16 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-5', nn.MaxPool2d(2)),
            ('enc-e2res-5', ResBlock(8 * self.hid, 8 * self.hid, kernel_size=self.kernel_size)),
        ]))

        self.conv_up_8 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-1', ResBlock(16 * self.hid, 4 * self.hid, kernel_size=self.kernel_size)),
        ]))
        self.conv_up_4 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-2', ResBlock(8 * self.hid, 2 * self.hid, kernel_size=self.kernel_size)),
        ]))
        self.conv_up_2 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-3', ResBlock(4 * self.hid, 1 * self.hid, kernel_size=self.kernel_size)),
        ]))
        self.conv_up_1 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-4', ResBlock(2 * self.hid, 1 * self.hid, kernel_size=self.kernel_size)),
            ('dec-e2conv-4', nn.Conv2d(1 * self.hid, self.n_output_channel, kernel_size=self.kernel_size, padding=1)),
        ]))

        self.upsample_16_8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_8_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_4_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.state_lin = nn.Linear(2, self.n_output_channel)
        self.trajactory_lin = nn.Linear(2, self.n_output_channel) if self.local_cond_type.find('input') > -1 \
            else None

        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Free parameters: ', self.total_params)

        self.activation = nn.ReLU()

    def forwardEncoder(self, obs):
        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        return feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16

    def forwardDecoder(self, feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16):
        concat_8 = torch.cat((feature_map_8, self.upsample_16_8(feature_map_16)), dim=1)
        feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat((feature_map_4, self.upsample_8_4(feature_map_up_8)), dim=1)
        feature_map_up_4 = self.conv_up_4(concat_4)

        concat_2 = torch.cat((feature_map_2, self.upsample_4_2(feature_map_up_4)), dim=1)
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat((feature_map_1, self.upsample_2_1(feature_map_up_2)), dim=1)
        feature_map_up_1 = self.conv_up_1(concat_1)

        return feature_map_up_1

    def forward(self, obs):
        img, state = obs['image'], obs['agent_pos']
        # Visualizing obs
        # plt.figure()
        # plt.imshow(img[0].permute(1, 2, 0).clone().detach().cpu().numpy())
        # pos = state[0].clone().detach().cpu().numpy()
        # # pos[1] = -pos[1]
        # pos += 1
        # pos /= 2
        # pos *= 95
        # plt.scatter(pos[0], pos[1], c='r')
        # plt.show()
        img = self.crop(img)
        feature_maps = self.forwardEncoder(img)
        global_info = None
        if self.local_cond_type.find('global'):
            global_info = feature_maps[-1].amax(dim=(-1,-2))
        feature_maps = self.forwardDecoder(*feature_maps)
        feature_maps += self.state_lin(state).unsqueeze(2).unsqueeze(3)
        # feature_maps = self.activation(feature_maps)
        return feature_maps, global_info

    def extrac_feature(self, embedding, trajectory):
        '''

        Parameters
        ----------
        embedding: image embedding in shape [(b To) c h w]
        trajectory: trajectory in shape [b k 2]

        Returns
        -------
        feature: feature in shape [b k (To c)]
        '''
        assert embedding.shape[2] == self.h
        assert embedding.shape[3] == self.w
        bs = trajectory.shape[0]
        To = embedding.shape[0] // bs
        embedding = rearrange(embedding, '(bs To) c h w -> bs (To c) (h w)', bs=bs)

        traj_w_h = trajectory.reshape(-1, 2).clone()
        traj_w_h *= (self.input_shape[1] - 1) / 2
        traj_w_h += (self.h - 1) / 2
        traj_w_h = torch.round(traj_w_h)

        idxs = torch.clamp(traj_w_h, 0, self.h - 1)
        idxs = idxs[:, 0] + idxs[:, 1] * self.w
        idxs = idxs.reshape(bs, -1).long()

        features = []
        for i in range(bs):
            features.append(embedding[i:i+1, :, idxs[i]])
        features = torch.cat(features).permute(0, 2, 1)

        if self.trajactory_lin is not None:
            trajectory_lin_project = self.trajactory_lin(trajectory.reshape(-1, 2))
            features += trajectory_lin_project.reshape(bs, -1, self.n_output_channel).repeat(1, 1, To)

        return features, traj_w_h


class DiffusionUnetUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(80, 80),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            fixed_crop=False,
            local_cond_type='',
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape in [None, 'None']:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # # load model
        # policy: PolicyAlgo = algo_factory(
        #         algo_name=config.algo_name,
        #         config=config,
        #         obs_key_shapes=obs_key_shapes,
        #         ac_dim=action_dim,
        #         device='cpu',
        #     )
        #
        # obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        assert eval_fixed_crop
        obs_encoder = ResUNet(n_input_channel=3, n_output_channel=32, n_hidden=32, kernel_size=3, input_shape=shape,
                              h=crop_shape[0], w=crop_shape[1], fixed_crop=fixed_crop, local_cond_type=local_cond_type)

        assert local_cond_type in ['input', 'film', 'input_global']
        self.local_cond_type = local_cond_type

        if obs_encoder_group_norm and crop_shape not in [None, 'None']:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        input_dim = obs_encoder.n_output_channel * n_obs_steps if self.local_cond_type.find('input') > -1 else action_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None if self.local_cond_type.find('input') > -1 else obs_encoder.n_output_channel * n_obs_steps,
            global_cond_dim=obs_encoder.n_neck_channel * n_obs_steps if self.local_cond_type.find('global') > -1 else None,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        # self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask, spacial_embedding,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. extract local features at the trajectory
            local_feature, _ = self.obs_encoder.extrac_feature(spacial_embedding, trajectory)

            # 3. predict model output
            # Predict the noise residual
            if self.local_cond_type == 'input':
                model_output = self.model(local_feature, t)[..., :self.action_dim]
            elif self.local_cond_type == 'input_global':
                model_output = self.model(local_feature, t, global_cond=global_cond)[..., :self.action_dim]
            elif self.local_cond_type == 'film':
                model_output = self.model(trajectory, t, local_cond=local_feature)

            # 4. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        # Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # condition through spacial_embedding
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        spacial_embedding, global_cond = self.obs_encoder(this_nobs)
        if global_cond is not None:
            # reshape back to B, Do
            global_cond = global_cond.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            spacial_embedding,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs,
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        spacial_embedding, global_cond = self.obs_encoder(this_nobs)
        if global_cond is not None:
            # reshape back to B, Do
            global_cond = global_cond.reshape(batch_size, -1)

        # # Visualizing pushT trajectory
        # img, state = this_nobs['image'], this_nobs['agent_pos']
        # plt.figure()
        # plt.imshow(img[0].permute(1, 2, 0).clone().detach().cpu().numpy())
        # state = state[0:2].clone().detach().cpu().numpy()
        # pos = trajectory[0].clone().detach().cpu().numpy()
        # pos = np.concatenate([state, pos], axis=0)
        # # pos[1] = -pos[1]
        # pos += 1
        # pos /= 2
        # pos *= 95
        # plt.scatter(pos[2:, 0], pos[2:, 1], c='r')
        # plt.scatter(pos[:2, 0], pos[:2, 1], c='pink')
        # plt.show()

        # generate inpainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        for i in range(100):
            noise = torch.randn(trajectory.shape, device=trajectory.device)
            if -1 <= noise.min() and noise.max() <= 1:
                break
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # 2. extract local features at the trajectory
        local_feature, _ = self.obs_encoder.extrac_feature(spacial_embedding, noisy_trajectory)

        # Predict the noise residual
        if self.local_cond_type == 'input':
            pred = self.model(local_feature, timesteps)[..., :self.action_dim]
        elif self.local_cond_type == 'input_global':
            pred = self.model(local_feature, timesteps, global_cond=global_cond)[..., :self.action_dim]
        elif self.local_cond_type == 'film':
            pred = self.model(trajectory, timesteps, local_cond=local_feature)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    nn = ResUNet(n_input_channel=1, n_output_channel=1, n_hidden=16)
    # nn.to('cuda:0')
    nn.eval()
    ones = torch.ones((1, 1, 96, 96)) * 100
    state = torch.tensor([47, 47])
    img = ones.clone()
    img[..., :state[0], state[1]:] = -100.
    img[..., :, :state[1] - 10] = 0.
    # img = |1 -1|
    #       |1  1|
    obs = {}
    obs['image'], obs['agent_pos'] = img, state.unsqueeze(0).float()
    traj = torch.linspace(-0.4, 0.4, 10)
    traj = torch.stack([traj, -traj], dim=1).unsqueeze(0)  # in shape [b k 2], in coordinate [x -y]
    traj[:, :, 1][traj[:, :, 1] > 0] = 0
    # traj = |_ /|
    #        |   |
    space_embedding = nn(obs)
    h, idx = nn.extrac_feature(space_embedding, traj)
    plt.figure()
    plt.imshow(space_embedding[0, 0].detach())
    plt.colorbar()
    # ptraj = traj.clone()
    # # ptraj[..., 1] = -ptraj[..., 1]
    # ptraj += 1
    # ptraj /= 2
    # ptraj *= 95 - 1
    plt.scatter(idx[..., 0], idx[..., 1], c='r')
    plt.figure()
    plt.plot(h.squeeze().detach())

    g = torch.tensor([0, 10])  # in image coordinate (h w)
    gstate = state + g
    gimg = ones.clone()
    gimg[..., :gstate[0], gstate[1]:] = -100.
    gimg[..., :, :gstate[1] - 10] = 0.
    gobs = {}
    gobs['image'], gobs['agent_pos'] = gimg, gstate.flip(0).unsqueeze(0).float()
    gtraj = traj.clone()
    gtraj[...,0] += g[1].unsqueeze(0).unsqueeze(0) * 2 / 95
    gtraj[...,1] += g[0].unsqueeze(0).unsqueeze(0) * 2 / 95
    gspace_embedding = nn(gobs)
    gh, gidx = nn.extrac_feature(gspace_embedding, gtraj)
    plt.figure()
    plt.imshow(gspace_embedding[0, 0].detach())
    plt.colorbar()
    # # gtraj[..., 1] = -gtraj[..., 1]
    # gtraj += 1
    # gtraj /= 2
    # gtraj *= 95 - 1
    plt.scatter(gidx[..., 0], gidx[..., 1], c='r')
    plt.figure()
    plt.plot(gh.squeeze().detach())
    plt.show()

    g = torch.tensor([10, 0])  # in image coordinate (h w)
    gstate = state + g
    gimg = ones.clone()
    gimg[..., :gstate[0], gstate[1]:] = -100.
    gimg[..., :, :gstate[1] - 10] = 0.
    gobs = {}
    gobs['image'], gobs['agent_pos'] = gimg, gstate.flip(0).unsqueeze(0).float()
    gtraj = traj.clone()
    gtraj[...,0] += g[1].unsqueeze(0).unsqueeze(0) * 2 / 95
    gtraj[...,1] += g[0].unsqueeze(0).unsqueeze(0) * 2 / 95
    gspace_embedding = nn(gobs)
    gh, gidx = nn.extrac_feature(gspace_embedding, gtraj)
    plt.figure()
    plt.imshow(gspace_embedding[0, 0].detach())
    plt.colorbar()
    # # gtraj[..., 1] = -gtraj[..., 1]
    # gtraj += 1
    # gtraj /= 2
    # gtraj *= 95 - 1
    plt.scatter(gidx[..., 0], gidx[..., 1], c='r')
    plt.figure()
    plt.plot(gh.squeeze().detach())
    plt.show()
