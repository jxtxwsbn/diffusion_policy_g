import copy
from typing import Dict
import math
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.policy.diffuser_actor_utils.layers import (
    FFWRelativeSelfAttentionModule,
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfCrossAttentionModule
)
from diffusion_policy.policy.diffuser_actor_utils.encoder import Encoder
from diffusion_policy.policy.diffuser_actor_utils.layers import ParallelAttention
from diffusion_policy.policy.diffuser_actor_utils.position_encodings import (
    RotaryPositionEncoding3D,
    SinusoidalPosEmb
)
from diffusion_policy.policy.diffuser_actor_utils.step_obs_encoder import StepObsEncoder
from diffusion_policy.policy.diffuser_actor_utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)


class StepDiffusionHead(nn.Module):

    def __init__(self,
                 embedding_dim=192,
                 num_attn_heads=8,
                 use_instruction=False,
                 # rotation_parametrization='quat',
                 nhist=2,
                 lang_enhanced=False):
        super().__init__()
        self.use_instruction = use_instruction
        self.lang_enhanced = lang_enhanced
        # if '6D' in rotation_parametrization:
        #     rotation_dim = 6  # continuous 6D
        # else:
        #     rotation_dim = 4  # quaternion

        # Encoders
        # self.traj_encoder = nn.Linear(9, embedding_dim)
        # self.traj_encoder = nn.Linear(3, embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Linear(embedding_dim * nhist, embedding_dim)
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # # Attention from trajectory queries to language
        # self.traj_lang_attention = nn.ModuleList([
        #     ParallelAttention(
        #         num_layers=1,
        #         d_model=embedding_dim, n_heads=num_attn_heads,
        #         self_attention1=False, self_attention2=False,
        #         cross_attention1=True, cross_attention2=False,
        #         rotary_pe=False, apply_ffn=False
        #     )
        # ])

        # # Estimate attends to context (no subsampling)
        # self.cross_attn = FFWRelativeCrossAttentionModule(
        #     embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        # )

        # # Shared attention layers
        # if not self.lang_enhanced:
        #     self.self_attn = FFWRelativeSelfAttentionModule(
        #         embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
        #     )
        # else:  # interleave cross-attention to language
        #     self.self_attn = FFWRelativeSelfCrossAttentionModule(
        #         embedding_dim, num_attn_heads,
        #         num_self_attn_layers=4,
        #         num_cross_attn_layers=3,
        #         use_adaln=True
        #     )

        # # Specific (non-shared) Output layers:
        # # 1. Rotation
        # self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        # if not self.lang_enhanced:
        #     self.rotation_self_attn = FFWRelativeSelfAttentionModule(
        #         embedding_dim, num_attn_heads, 2, use_adaln=True
        #     )
        # else:  # interleave cross-attention to language
        #     self.rotation_self_attn = FFWRelativeSelfCrossAttentionModule(
        #         embedding_dim, num_attn_heads, 2, 1, use_adaln=True
        #     )
        # self.rotation_predictor = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, rotation_dim)
        # )

        # 2. Position
        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )
        # self.position_self_attn = FFWRelativeSelfAttentionModule(embedding_dim,
        #                                                          num_attn_heads,
        #                                                          12, use_adaln=True)
        self.position_self_attn = FFWRelativeSelfCrossAttentionModule(
            embedding_dim, num_attn_heads,
            num_self_attn_layers=6,
            num_cross_attn_layers=5,
            use_adaln=True
        )
        self.position_predictor = nn.Sequential(
                                                nn.Linear(embedding_dim, embedding_dim),
                                                nn.SiLU(),
                                                nn.Linear(embedding_dim, 2)
                                            )

        # # 3. Openess
        # self.openess_predictor = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, 1)
        # )

    def forward(self, noisy_trajectory, timesteps,
                local_cond, global_cond):
        """
        Arguments:
            noisy_trajectory: (B, trajectory_length, 2)
            timestep: (B, 1)
            local_cond:
                local_context: (B, (F To))
                local_coordinate: (B, 1, 3)
            global_cond:
                global_context: (B, (F To))
                global_coordinate: (B, 1, 3)
        """
        # Trajectory features
        trajectory = torch.cat([noisy_trajectory, torch.zeros_like(noisy_trajectory[..., :1])], dim=-1)

        traj_time_pos = self.traj_time_emb(
            torch.arange(0, trajectory.size(1), device=trajectory.device)
        )[None].repeat(len(trajectory), 1, 1)
        traj_feats = traj_time_pos

        traj_feats = einops.rearrange(traj_feats, 'b l c -> l b c')

        global_context, global_coordinate = global_cond
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(timesteps, global_context)

        # Positional embeddings
        rel_trajectory_pos = self.relative_pe_layer(trajectory)
        rel_global_pos = self.relative_pe_layer(global_coordinate)

        # Position head
        # hidden_traj_feats = self.cross_attn(
        #     query=traj_feats,
        #     value=time_embs.unsqueeze(0),
        #     query_pos=rel_trajectory_pos,
        #     value_pos=rel_global_pos,
        #     diff_ts=time_embs
        # )[-1]
        hidden_traj_feats = self.position_self_attn(
            query=traj_feats,
            query_pos=rel_trajectory_pos,
            context=torch.ones_like(time_embs.unsqueeze(0)),
            context_pos=rel_global_pos,
            diff_ts=time_embs,
        )[-1]
        hidden_traj_feats = einops.rearrange(
            hidden_traj_feats, "npts b c -> b npts c"
        )
        position = self.position_predictor(hidden_traj_feats)

        # # Openess head from position head
        # openess = self.openess_predictor(position_features)

        return position

    def encode_denoising_timestep(self, timestep, curr_gripper_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)
            - curr_gripper_features: (B, (To F))

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features)
        return time_feats + curr_gripper_feats

    # def predict_pos(self, features, rel_pos, time_embs, num_gripper,
    #                 instr_feats):
    #     position_features = self.position_self_attn(
    #         query=features,
    #         query_pos=rel_pos,
    #         diff_ts=time_embs,
    #         context=instr_feats,
    #         context_pos=None
    #     )[-1]
    #     position_features = einops.rearrange(
    #         position_features[:num_gripper], "npts b c -> b npts c"
    #     )
    #     position = self.position_predictor(position_features)
    #     return position, position_features

    # def predict_rot(self, features, rel_pos, time_embs, num_gripper,
    #                 instr_feats):
    #     rotation_features = self.rotation_self_attn(
    #         query=features,
    #         query_pos=rel_pos,
    #         diff_ts=time_embs,
    #         context=instr_feats,
    #         context_pos=None
    #     )[-1]
    #     rotation_features = einops.rearrange(
    #         rotation_features[:num_gripper], "npts b c -> b npts c"
    #     )
    #     rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
    #     rotation = self.rotation_predictor(rotation_features)
    #     return rotation


class Diffusion2DSTEP(BaseImagePolicy):
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 obs_as_global_cond=True,
                 crop_shape=(84, 84),
                 da_encoder=False,
                 da_unet=True,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 cond_predict_scale=True,
                 obs_encoder_group_norm=False,
                 eval_fixed_crop=False,
                 fixed_crop=False,
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

        self.da_encoder = da_encoder
        self.da_unet = da_unet

        embedding_dim = 480
        num_attn_heads = 8

        if not da_encoder:
            # get raw robomimic config
            config = get_robomimic_config(
                algo_name='bc_rnn',
                hdf5_type='image',
                task_name='square',
                dataset_type='ph')

            with config.unlocked():
                # set config with shape_meta
                config.observation.modalities.obs = obs_config

                if crop_shape is None:
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

            # load model
            policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

            obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

            if obs_encoder_group_norm:
                # replace batch norm with group norm
                replace_submodules(
                    root_module=obs_encoder,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=x.num_features // 16,
                        num_channels=x.num_features)
                )
                # obs_encoder.obs_nets['agentview_image'].nets[0].nets

            # obs_encoder.obs_randomizers['agentview_image']
            if eval_fixed_crop or fixed_crop:
                replace_submodules(
                    root_module=obs_encoder,
                    predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                    func=lambda x: dmvc.CropRandomizer(
                        input_shape=x.input_shape,
                        crop_height=x.crop_height,
                        crop_width=x.crop_width,
                        num_crops=x.num_crops,
                        pos_enc=x.pos_enc,
                        fixed_crop=fixed_crop
                    )
                )

            self.obs_encoder = obs_encoder
            self.lin = nn.Linear(obs_encoder.output_shape()[0] * n_obs_steps, embedding_dim * n_obs_steps)

        if not da_unet:
            # create diffusion model
            obs_feature_dim = obs_encoder.output_shape()[0]
            input_dim = action_dim + obs_feature_dim * n_obs_steps
            global_cond_dim = None
            if obs_as_global_cond:
                input_dim = action_dim
                global_cond_dim = obs_feature_dim * n_obs_steps

            model = ConditionalUnet1D(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            )
            self.model = model

        if da_encoder:
            self.obs_encoder = StepObsEncoder(input_shape=shape,
                                              image_size=crop_shape,
                                              embedding_dim=embedding_dim,
                                              num_attn_heads=num_attn_heads,
                                              num_sampling_level=1,
                                              To=n_obs_steps)
        if da_unet:
            self.model = StepDiffusionHead(embedding_dim=embedding_dim,
                                           num_attn_heads=num_attn_heads,
                                           nhist=n_obs_steps)

        self.noise_scheduler = noise_scheduler
        # self.mask_generator = LowdimMaskGenerator(
        #     action_dim=action_dim,
        #     obs_dim=0 if obs_as_global_cond else obs_feature_dim * n_obs_steps,
        #     max_n_obs_steps=n_obs_steps,
        #     fix_obs_steps=True,
        #     action_visible=False
        # )
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
                           condition_data,
                           local_cond=None, global_cond=None,
                           # keyword arguments to scheduler.step
                           **kwargs
                           ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 2. predict model output
            model_output = model(trajectory, t.unsqueeze(0).to(trajectory.device),
                                 local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                **kwargs
            ).prev_sample

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict  # not implemented yet
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

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        global_cond = self.obs_encoder(this_nobs)
        if self.da_encoder:
            global_context = global_cond[0].reshape(B, -1)
            global_coordinate = global_cond[1][::To]
        else:
            global_context = self.lin(global_cond.reshape(B, -1))
            state = this_nobs['agent_pos'][:1, :]  # state in shape [b 2]
            global_coordinate = torch.cat([state, torch.zeros_like(state[:, :1])],
                                          dim=1).reshape(-1, 1, 3)  # [b 1 3]
        global_cond = (global_context, global_coordinate)

        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

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
                               lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        global_cond = self.obs_encoder(this_nobs)
        if self.da_encoder:
            global_context = global_cond[0].reshape(batch_size, -1)
            global_coordinate = global_cond[1][::self.n_obs_steps]
        else:
            global_context = self.lin(global_cond.reshape(batch_size, -1))
            state = this_nobs['agent_pos'][:1, :]  # state in shape [b 2]
            global_coordinate = torch.cat([state, torch.zeros_like(state[:, :1])],
                                          dim=1).reshape(-1, 1, 3)  # [b 1 3]
        global_cond = (global_context, global_coordinate)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
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

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps,
                          local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
