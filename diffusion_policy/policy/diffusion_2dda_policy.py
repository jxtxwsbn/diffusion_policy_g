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
from diffusion_policy.policy.diffuser_actor_utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)


class DiffuserActor(nn.Module):

    def __init__(self,
                 # backbone="clip",
                 image_size=(84, 84),
                 embedding_dim=384,
                 num_vis_ins_attn_layers=2,
                 use_instruction=False,
                 fps_subsampling_factor=5,
                 # gripper_loc_bounds=None,
                 # rotation_parametrization='6D',
                 # quaternion_format='xyzw',
                 shape=(3, 96, 96),
                 diffusion_timesteps=100,
                 nhist=2,
                 relative=False,
                 lang_enhanced=False):
        super().__init__()
        # self._rotation_parametrization = rotation_parametrization
        # self._quaternion_format = quaternion_format
        self._relative = relative
        self.use_instruction = use_instruction
        self.nhist = nhist
        self.encoder = Encoder(
            # backbone=backbone,
            input_shape=shape,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_sampling_level=1,
            nhist=nhist,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor
        )
        self.prediction_head = DiffusionHead(
            embedding_dim=embedding_dim,
            use_instruction=use_instruction,
            # rotation_parametrization=rotation_parametrization,
            nhist=nhist,
            lang_enhanced=lang_enhanced
        )
        # self.position_noise_scheduler = DDPMScheduler(
        #     num_train_timesteps=diffusion_timesteps,
        #     beta_schedule="scaled_linear",
        #     prediction_type="epsilon"
        # )
        # self.rotation_noise_scheduler = DDPMScheduler(
        #     num_train_timesteps=diffusion_timesteps,
        #     beta_schedule="squaredcos_cap_v2",
        #     prediction_type="epsilon"
        # )
        # self.n_steps = diffusion_timesteps
        # self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

    def encode_inputs(self, obs):
        img, state = obs['image'], obs['agent_pos']
        # img in shape [(b To) c h w]; state in shape [(b To) 2]
        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid = self.encoder.encode_images(img[::self.nhist])
        # # Keep only low-res scale
        # context_feats = einops.rearrange(
        #     rgb_feats_pyramid[0],
        #     "b ncam c h w -> b (ncam h w) c"
        # )
        context_feats, context = rgb_feats_pyramid[0], pcd_pyramid[0]
        curr_gripper = torch.cat([state, torch.zeros_like(state[:, :1])],
                                 dim=1).reshape(-1, self.nhist, 3)  # [(b To) 1 3]

        # Encode instruction (B, 53, F)
        instr_feats = None
        # if self.use_instruction:
        #     instr_feats, _ = self.encoder.encode_instruction(instruction)

        # # Cross-attention vision to language
        # if self.use_instruction:
        #     # Attention from vision to language
        #     context_feats = self.encoder.vision_language_attention(
        #         context_feats, instr_feats
        #     )

        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats, _ = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats, context
        )

        # FPS on visual features (N, B, F) and (B, N, F, 2)
        fps_feats, fps_pos = self.encoder.run_fps(
            context_feats.transpose(0, 1),
            self.encoder.relative_pe_layer(context)
        )
        return (
            context_feats, context,  # contextualized visual features
            instr_feats,  # language features
            adaln_gripper_feats,  # gripper history features
            fps_feats, fps_pos  # sampled visual features
        )

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            context_feats,
            context,
            instr_feats,
            adaln_gripper_feats,
            fps_feats,
            fps_pos
        ) = fixed_inputs

        return self.prediction_head(
            trajectory,
            timestep,
            context_feats=context_feats,
            context=context,
            instr_feats=instr_feats,
            adaln_gripper_feats=adaln_gripper_feats,
            fps_feats=fps_feats,
            fps_pos=fps_pos
        )

    # def conditional_sample(self, condition_data, condition_mask, fixed_inputs):
    #     self.position_noise_scheduler.set_timesteps(self.n_steps)
    #     self.rotation_noise_scheduler.set_timesteps(self.n_steps)
    #
    #     # Random trajectory, conditioned on start-end
    #     noise = torch.randn(
    #         size=condition_data.shape,
    #         dtype=condition_data.dtype,
    #         device=condition_data.device
    #     )
    #     # Noisy condition data
    #     noise_t = torch.ones(
    #         (len(condition_data),), device=condition_data.device
    #     ).long().mul(self.position_noise_scheduler.timesteps[0])
    #     noise_pos = self.position_noise_scheduler.add_noise(
    #         condition_data[..., :3], noise[..., :3], noise_t
    #     )
    #     noise_rot = self.rotation_noise_scheduler.add_noise(
    #         condition_data[..., 3:9], noise[..., 3:9], noise_t
    #     )
    #     noisy_condition_data = torch.cat((noise_pos, noise_rot), -1)
    #     trajectory = torch.where(
    #         condition_mask, noisy_condition_data, noise
    #     )
    #
    #     # Iterative denoising
    #     timesteps = self.position_noise_scheduler.timesteps
    #     for t in timesteps:
    #         out = self.policy_forward_pass(
    #             trajectory,
    #             t * torch.ones(len(trajectory)).to(trajectory.device).long(),
    #             fixed_inputs
    #         )
    #         out = out[-1]  # keep only last layer's output
    #         pos = self.position_noise_scheduler.step(
    #             out[..., :3], t, trajectory[..., :3]
    #         ).prev_sample
    #         rot = self.rotation_noise_scheduler.step(
    #             out[..., 3:9], t, trajectory[..., 3:9]
    #         ).prev_sample
    #         trajectory = torch.cat((pos, rot), -1)
    #
    #     trajectory = torch.cat((trajectory, out[..., 9:]), -1)
    #
    #     return trajectory

    # def compute_trajectory(
    #     self,
    #     trajectory_mask,
    #     rgb_obs,
    #     pcd_obs,
    #     instruction,
    #     curr_gripper
    # ):
    #     # Normalize all pos
    #     pcd_obs = pcd_obs.clone()
    #     curr_gripper = curr_gripper.clone()
    #     pcd_obs = torch.permute(self.normalize_pos(
    #         torch.permute(pcd_obs, [0, 1, 3, 4, 2])
    #     ), [0, 1, 4, 2, 3])
    #     curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
    #     curr_gripper = self.convert_rot(curr_gripper)
    #
    #     # Prepare inputs
    #     fixed_inputs = self.encode_inputs(
    #         rgb_obs, pcd_obs, instruction, curr_gripper
    #     )
    #
    #     # Condition on start-end pose
    #     B, nhist, D = curr_gripper.shape
    #     cond_data = torch.zeros(
    #         (B, trajectory_mask.size(1), D),
    #         device=rgb_obs.device
    #     )
    #     cond_mask = torch.zeros_like(cond_data)
    #     cond_mask = cond_mask.bool()
    #
    #     # Sample
    #     trajectory = self.conditional_sample(
    #         cond_data,
    #         cond_mask,
    #         fixed_inputs
    #     )
    #
    #     # Normalize quaternion
    #     if self._rotation_parametrization != '6D':
    #         trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
    #     # Back to quaternion
    #     trajectory = self.unconvert_rot(trajectory)
    #     # unnormalize position
    #     trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])
    #     # Convert gripper status to probaility
    #     if trajectory.shape[-1] > 7:
    #         trajectory[..., 7] = trajectory[..., 7].sigmoid()
    #
    #     return trajectory

    # def normalize_pos(self, pos):
    #     pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
    #     pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
    #     return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0
    #
    # def unnormalize_pos(self, pos):
    #     pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
    #     pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
    #     return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min
    #
    # def convert_rot(self, signal):
    #     signal[..., 3:7] = normalise_quat(signal[..., 3:7])
    #     if self._rotation_parametrization == '6D':
    #         # The following code expects wxyz quaternion format!
    #         if self._quaternion_format == 'xyzw':
    #             signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
    #         rot = quaternion_to_matrix(signal[..., 3:7])
    #         res = signal[..., 7:] if signal.size(-1) > 7 else None
    #         if len(rot.shape) == 4:
    #             B, L, D1, D2 = rot.shape
    #             rot = rot.reshape(B * L, D1, D2)
    #             rot_6d = get_ortho6d_from_rotation_matrix(rot)
    #             rot_6d = rot_6d.reshape(B, L, 6)
    #         else:
    #             rot_6d = get_ortho6d_from_rotation_matrix(rot)
    #         signal = torch.cat([signal[..., :3], rot_6d], dim=-1)
    #         if res is not None:
    #             signal = torch.cat((signal, res), -1)
    #     return signal
    #
    # def unconvert_rot(self, signal):
    #     if self._rotation_parametrization == '6D':
    #         res = signal[..., 9:] if signal.size(-1) > 9 else None
    #         if len(signal.shape) == 3:
    #             B, L, _ = signal.shape
    #             rot = signal[..., 3:9].reshape(B * L, 6)
    #             mat = compute_rotation_matrix_from_ortho6d(rot)
    #             quat = matrix_to_quaternion(mat)
    #             quat = quat.reshape(B, L, 4)
    #         else:
    #             rot = signal[..., 3:9]
    #             mat = compute_rotation_matrix_from_ortho6d(rot)
    #             quat = matrix_to_quaternion(mat)
    #         signal = torch.cat([signal[..., :3], quat], dim=-1)
    #         if res is not None:
    #             signal = torch.cat((signal, res), -1)
    #         # The above code handled wxyz quaternion format!
    #         if self._quaternion_format == 'xyzw':
    #             signal[..., 3:7] = signal[..., (4, 5, 6, 3)]
    #     return signal
    #
    # def convert2rel(self, pcd, curr_gripper):
    #     """Convert coordinate system relaative to current gripper."""
    #     center = curr_gripper[:, -1, :3]  # (batch_size, 3)
    #     bs = center.shape[0]
    #     pcd = pcd - center.view(bs, 1, 3, 1, 1)
    #     curr_gripper = curr_gripper.clone()
    #     curr_gripper[..., :3] = curr_gripper[..., :3] - center.view(bs, 1, 3)
    #     return pcd, curr_gripper
    #
    # def forward(
    #     self,
    #     gt_trajectory,
    #     trajectory_mask,
    #     rgb_obs,
    #     pcd_obs,
    #     instruction,
    #     curr_gripper,
    #     run_inference=False
    # ):
    #     """
    #     Arguments:
    #         gt_trajectory: (B, trajectory_length, 3+4+X)
    #         trajectory_mask: (B, trajectory_length)
    #         timestep: (B, 1)
    #         rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
    #         pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
    #         instruction: (B, max_instruction_length, 512)
    #         curr_gripper: (B, nhist, 3+4+X)
    #
    #     Note:
    #         Regardless of rotation parametrization, the input rotation
    #         is ALWAYS expressed as a quaternion form.
    #         The model converts it to 6D internally if needed.
    #     """
    #     if self._relative:
    #         pcd_obs, curr_gripper = self.convert2rel(pcd_obs, curr_gripper)
    #     if gt_trajectory is not None:
    #         gt_openess = gt_trajectory[..., 7:]
    #         gt_trajectory = gt_trajectory[..., :7]
    #     curr_gripper = curr_gripper[..., :7]
    #
    #     # gt_trajectory is expected to be in the quaternion format
    #     if run_inference:
    #         return self.compute_trajectory(
    #             trajectory_mask,
    #             rgb_obs,
    #             pcd_obs,
    #             instruction,
    #             curr_gripper
    #         )
    #     # Normalize all pos
    #     gt_trajectory = gt_trajectory.clone()
    #     pcd_obs = pcd_obs.clone()
    #     curr_gripper = curr_gripper.clone()
    #     gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
    #     pcd_obs = torch.permute(self.normalize_pos(
    #         torch.permute(pcd_obs, [0, 1, 3, 4, 2])
    #     ), [0, 1, 4, 2, 3])
    #     curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
    #
    #     # Convert rotation parametrization
    #     gt_trajectory = self.convert_rot(gt_trajectory)
    #     curr_gripper = self.convert_rot(curr_gripper)
    #
    #     # Prepare inputs
    #     fixed_inputs = self.encode_inputs(
    #         rgb_obs, pcd_obs, instruction, curr_gripper
    #     )
    #
    #     # Condition on start-end pose
    #     cond_data = torch.zeros_like(gt_trajectory)
    #     cond_mask = torch.zeros_like(cond_data)
    #     cond_mask = cond_mask.bool()
    #
    #     # Sample noise
    #     noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)
    #
    #     # Sample a random timestep
    #     timesteps = torch.randint(
    #         0,
    #         self.position_noise_scheduler.config.num_train_timesteps,
    #         (len(noise),), device=noise.device
    #     ).long()
    #
    #     # Add noise to the clean trajectories
    #     pos = self.position_noise_scheduler.add_noise(
    #         gt_trajectory[..., :3], noise[..., :3],
    #         timesteps
    #     )
    #     rot = self.rotation_noise_scheduler.add_noise(
    #         gt_trajectory[..., 3:9], noise[..., 3:9],
    #         timesteps
    #     )
    #     noisy_trajectory = torch.cat((pos, rot), -1)
    #     noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition
    #     assert not cond_mask.any()
    #
    #     # Predict the noise residual
    #     pred = self.policy_forward_pass(
    #         noisy_trajectory, timesteps, fixed_inputs
    #     )
    #
    #     # Compute loss
    #     total_loss = 0
    #     for layer_pred in pred:
    #         trans = layer_pred[..., :3]
    #         rot = layer_pred[..., 3:9]
    #         loss = (
    #             30 * F.l1_loss(trans, noise[..., :3], reduction='mean')
    #             + 10 * F.l1_loss(rot, noise[..., 3:9], reduction='mean')
    #         )
    #         if torch.numel(gt_openess) > 0:
    #             openess = layer_pred[..., 9:]
    #             loss += F.binary_cross_entropy_with_logits(openess, gt_openess)
    #         total_loss = total_loss + loss
    #     return total_loss


class DiffusionHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
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
        self.traj_encoder = nn.Linear(3, embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = nn.ModuleList([
            ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
        ])

        # Estimate attends to context (no subsampling)
        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )

        # Shared attention layers
        if not self.lang_enhanced:
            self.self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads,
                num_self_attn_layers=4,
                num_cross_attn_layers=3,
                use_adaln=True
            )

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
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.position_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.position_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

        # # 3. Openess
        # self.openess_predictor = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, 1)
        # )

    def forward(self, trajectory, timestep,
                context_feats, context, instr_feats, adaln_gripper_feats,
                fps_feats, fps_pos):
        """
        Arguments:
            trajectory: (B, trajectory_length, 2)
            timestep: (B, 1)
            context_feats: (B, N, F)
            context: (B, N, F, 2)
            instr_feats: (B, max_instruction_length, F)
            adaln_gripper_feats: (B, nhist, F)
            fps_feats: (N, B, F), N < context_feats.size(1)
            fps_pos: (B, N, F, 2)
        """
        # Trajectory features
        trajectory = torch.cat([trajectory, torch.zeros_like(trajectory[..., :1])], dim=-1)
        traj_feats = self.traj_encoder(trajectory)  # (B, L, F)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        if self.use_instruction:
            traj_feats, _ = self.traj_lang_attention[0](
                seq1=traj_feats, seq1_key_padding_mask=None,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        traj_feats = einops.rearrange(traj_feats, 'b l c -> l b c')
        context_feats = einops.rearrange(context_feats, 'b l c -> l b c')
        adaln_gripper_feats = einops.rearrange(
            adaln_gripper_feats, 'b l c -> l b c'
        )
        pos_pred, rot_pred, openess_pred = self.prediction_head(
            trajectory[..., :3], traj_feats,
            context[..., :3], context_feats,
            timestep, adaln_gripper_feats,
            fps_feats, fps_pos,
            instr_feats
        )
        # return [torch.cat((pos_pred, rot_pred, openess_pred), -1)]
        return pos_pred[..., :2]

    def prediction_head(self,
                        gripper_pcd, gripper_features,
                        context_pcd, context_features,
                        timesteps, curr_gripper_features,
                        sampled_context_features, sampled_rel_context_pos,
                        instr_feats):
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, F)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, F)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            curr_gripper_features: A tensor of shape (M, B, F)
            sampled_context_features: A tensor of shape (K, B, F)
            sampled_rel_context_pos: A tensor of shape (B, K, F, 2)
            instr_feats: (B, max_instruction_length, F)
        """
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(
            timesteps, curr_gripper_features
        )

        # Positional embeddings
        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        rel_context_pos = self.relative_pe_layer(context_pcd)

        # Cross attention from gripper to full context
        gripper_features = self.cross_attn(
            query=gripper_features,
            value=context_features,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs
        )[-1]

        # Self attention among gripper and sampled context
        features = torch.cat([gripper_features, sampled_context_features], 0)
        rel_pos = torch.cat([rel_gripper_pos, sampled_rel_context_pos], 1)
        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]

        num_gripper = gripper_features.shape[0]

        # # Rotation head
        # rotation = self.predict_rot(
        #     features, rel_pos, time_embs, num_gripper, instr_feats
        # )
        rotation = None

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )

        # # Openess head from position head
        # openess = self.openess_predictor(position_features)
        openess = None

        return position, rotation, openess

    def encode_denoising_timestep(self, timestep, curr_gripper_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)

        curr_gripper_features = einops.rearrange(
            curr_gripper_features, "npts b c -> b npts c"
        )
        curr_gripper_features = curr_gripper_features.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features)
        return time_feats + curr_gripper_feats

    def predict_pos(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        position_features = self.position_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        position_features = einops.rearrange(
            position_features[:num_gripper], "npts b c -> b npts c"
        )
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        rotation_features = self.rotation_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        rotation_features = einops.rearrange(
            rotation_features[:num_gripper], "npts b c -> b npts c"
        )
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation


class Diffusion2DDAPolicy(BaseImagePolicy):
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 obs_as_global_cond=True,
                 crop_shape=(84, 84),
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 cond_predict_scale=True,
                 obs_encoder_group_norm=False,
                 eval_fixed_crop=False,
                 fixed_crop=False,
                 canonicalize=False,
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

        # # get raw robomimic config
        # config = get_robomimic_config(
        #     algo_name='bc_rnn',
        #     hdf5_type='image',
        #     task_name='square',
        #     dataset_type='ph')
        #
        # with config.unlocked():
        #     # set config with shape_meta
        #     config.observation.modalities.obs = obs_config
        #
        #     if crop_shape is None:
        #         for key, modality in config.observation.encoder.items():
        #             if modality.obs_randomizer_class == 'CropRandomizer':
        #                 modality['obs_randomizer_class'] = None
        #     else:
        #         # set random crop parameter
        #         ch, cw = crop_shape
        #         for key, modality in config.observation.encoder.items():
        #             if modality.obs_randomizer_class == 'CropRandomizer':
        #                 modality.obs_randomizer_kwargs.crop_height = ch
        #                 modality.obs_randomizer_kwargs.crop_width = cw
        #
        # # init global state
        # ObsUtils.initialize_obs_utils_with_config(config)
        #
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
        #
        # if obs_encoder_group_norm:
        #     # replace batch norm with group norm
        #     replace_submodules(
        #         root_module=obs_encoder,
        #         predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        #         func=lambda x: nn.GroupNorm(
        #             num_groups=x.num_features//16,
        #             num_channels=x.num_features)
        #     )
        #     # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        #
        # # obs_encoder.obs_randomizers['agentview_image']
        # if eval_fixed_crop or fixed_crop:
        #     replace_submodules(
        #         root_module=obs_encoder,
        #         predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
        #         func=lambda x: dmvc.CropRandomizer(
        #             input_shape=x.input_shape,
        #             crop_height=x.crop_height,
        #             crop_width=x.crop_width,
        #             num_crops=x.num_crops,
        #             pos_enc=x.pos_enc,
        #             fixed_crop=fixed_crop
        #         )
        #     )

        # # create diffusion model
        # obs_feature_dim = obs_encoder.output_shape()[0]
        # input_dim = action_dim + obs_feature_dim * n_obs_steps
        # global_cond_dim = None
        # if obs_as_global_cond:
        #     input_dim = action_dim
        #     global_cond_dim = obs_feature_dim * n_obs_steps
        #
        # model = ConditionalUnet1D(
        #     input_dim=input_dim,
        #     local_cond_dim=None,
        #     global_cond_dim=global_cond_dim,
        #     diffusion_step_embed_dim=diffusion_step_embed_dim,
        #     down_dims=down_dims,
        #     kernel_size=kernel_size,
        #     n_groups=n_groups,
        #     cond_predict_scale=cond_predict_scale
        # )

        self.da = DiffuserActor(
            image_size=crop_shape,
            shape=shape,
            nhist=n_obs_steps
        )
        # self.obs_encoder = obs_encoder
        # self.model = model
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

        print("Diffusion params: %e" % sum(p.numel() for p in self.da.encoder.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.da.prediction_head.parameters()))

    # ========= inference  ============
    def conditional_sample(self,
                           condition_data,
                           nobs_features,
                           # keyword arguments to scheduler.step
                           **kwargs
                           ):
        # model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:

            # 2. predict model output
            model_output = self.da.policy_forward_pass(trajectory, t.unsqueeze(0).to(trajectory.device), nobs_features)

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
        nobs_features = self.da.encode_inputs(this_nobs)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            nobs_features,
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
        nobs_features = self.da.encode_inputs(this_nobs)

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
        pred = self.da.policy_forward_pass(noisy_trajectory, timesteps, nobs_features)

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
