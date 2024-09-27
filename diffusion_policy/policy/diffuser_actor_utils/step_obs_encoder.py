import dgl.geometry as dgl_geo
import einops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork

from .position_encodings import RotaryPositionEncoding3D
from .layers import FFWRelativeCrossAttentionModule, ParallelAttention, FFWRelativeSelfAttentionModule
from .resnet import load_resnet50, load_resnet18
# from .clip import load_clip
import diffusion_policy.model.vision.crop_randomizer as dmvc


def normalize_img(img):
    return img * 2 - 1.0


class StepObsEncoder(nn.Module):

    def __init__(self,
                 backbone="clip",
                 input_shape=(3, 96, 96),
                 image_size=(84, 84),
                 embedding_dim=192,
                 num_sampling_level=3,
                 To=2,
                 num_attn_heads=8,
                 num_vis_ins_attn_layers=2,
                 fps_subsampling_factor=5):
        super().__init__()
        # assert backbone in ["resnet50", "resnet18", "clip"]
        # assert image_size in [(128, 128), (256, 256)]
        # assert num_sampling_level in [1, 2, 3, 4]

        self.image_size = image_size
        self.num_sampling_level = num_sampling_level
        self.To = To
        self.fps_subsampling_factor = fps_subsampling_factor
        self.fps_shape = (image_size[0] // fps_subsampling_factor, image_size[1] // fps_subsampling_factor)
        self.crop = dmvc.CropRandomizer(
            input_shape=input_shape,
            crop_height=image_size[0],
            crop_width=image_size[1]
        )

        # build grid for image
        h, w = (np.linspace(-image_size[0] / input_shape[1], image_size[0] / input_shape[1], self.fps_shape[0]),
                np.linspace(-image_size[1] / input_shape[2], image_size[1] / input_shape[2], self.fps_shape[1]))
        h, w = np.meshgrid(h, w)
        h, w = torch.tensor(h), torch.tensor(w)
        h, w = h.reshape(-1).float(), w.reshape(-1).float()
        zeros = torch.zeros_like(h)
        self.register_buffer('grid', torch.stack([h, w, zeros], dim=1).unsqueeze(0))  # 1 n 3

        # Frozen backbone
        if backbone == "resnet50":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "resnet18":
            self.backbone, self.normalize = load_resnet18()
        # elif backbone == "clip":
        #     self.backbone, self.normalize = load_clip()
        else:
            self.backbone, self.normalize = None, normalize_img
            self.rgb_lin = nn.Linear(3, embedding_dim)
            self.rgb_encoder = FFWRelativeSelfAttentionModule(embedding_dim,
                                                              num_attn_heads,
                                                              3, use_adaln=False)

        if self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # # Semantic visual features at different scales
        # self.feature_pyramid = FeaturePyramidNetwork(
        #     [64, 256, 512, 1024, 2048], embedding_dim
        # )
        # if self.image_size == (128, 128):
        #     # Coarse RGB features are the 2nd layer of the feature pyramid
        #     # at 1/4 resolution (32x32)
        #     # Fine RGB features are the 1st layer of the feature pyramid
        #     # at 1/2 resolution (64x64)
        #     self.coarse_feature_map = ['res2', 'res1', 'res1', 'res1']
        #     self.downscaling_factor_pyramid = [4, 2, 2, 2]
        # elif self.image_size == (256, 256):
        #     # Coarse RGB features are the 3rd layer of the feature pyramid
        #     # at 1/8 resolution (32x32)
        #     # Fine RGB features are the 1st layer of the feature pyramid
        #     # at 1/2 resolution (128x128)
        #     self.feature_map_pyramid = ['res3', 'res1', 'res1', 'res1']
        #     self.downscaling_factor_pyramid = [8, 2, 2, 2]

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(1, embedding_dim)
        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=3, use_adaln=False
        )

        # # Goal gripper learnable features
        # self.goal_gripper_embed = nn.Embedding(1, embedding_dim)
        #
        # # Instruction encoder
        # self.instruction_encoder = nn.Linear(512, embedding_dim)

        # Attention from vision to language
        layer = ParallelAttention(
            num_layers=num_vis_ins_attn_layers,
            d_model=embedding_dim, n_heads=num_attn_heads,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )
        self.vl_attention = nn.ModuleList([
            layer
            for _ in range(1)
            for _ in range(1)
        ])

    def forward(self, obs):
        img, state = obs['image'], obs['agent_pos']
        # img in shape [(b To) c h w]; state in shape [(b To) 2]
        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid = self.encode_images(img)
        # # Keep only low-res scale
        # context_feats = einops.rearrange(
        #     rgb_feats_pyramid[0],
        #     "b ncam c h w -> b (ncam h w) c"
        # )
        context_feats, context = rgb_feats_pyramid[0], pcd_pyramid[0]
        curr_gripper = torch.cat([state, torch.zeros_like(state[:, :1])],
                                 dim=1).reshape(-1, 1, 3)  # [(b To) 1 3]

        # Encode instruction (B, 53, F)
        # if self.use_instruction:
        #     instr_feats, _ = self.encoder.encode_instruction(instruction)

        # # Cross-attention vision to language
        # if self.use_instruction:
        #     # Attention from vision to language
        #     context_feats = self.encoder.vision_language_attention(
        #         context_feats, instr_feats
        #     )

        # Encode gripper history ((B, To), 1, F)
        adaln_gripper_feats, _ = self.encode_gripper(curr_gripper, self.curr_gripper_embed, context_feats, context)
        # adaln_gripper_feats = adaln_gripper_feats + context_feats.max(dim=1, keepdim=True)[0]

        # import matplotlib.pyplot as plt
        # img_x = context[0].cpu().detach()
        # img_rgb = context_feats[0][:, :3].cpu().detach()
        # img_rgb -= img_rgb.min()
        # img_rgb /= img_rgb.max()
        # pos = state[0].cpu().detach()
        # plt.figure()
        # plt.scatter(img_x[:, 0], -img_x[:, 1], c=img_rgb, marker='o')
        # plt.scatter(pos[0], -pos[1], c='r')
        # plt.axis('equal')
        # plt.show()

        # # FPS on visual features (N, B, F) and (B, N, F, 2)
        # fps_feats, fps_pos = self.encoder.run_fps(
        #     context_feats.transpose(0, 1),
        #     self.encoder.relative_pe_layer(context)
        # )
        # return (
        #     context_feats, context,  # contextualized visual features
        #     instr_feats,  # language features
        #     adaln_gripper_feats,  # gripper history features
        #     fps_feats, fps_pos  # sampled visual features
        # )

        return adaln_gripper_feats, curr_gripper

    # def encode_curr_gripper(self, curr_gripper, context_feats, context):
    #     """
    #     Compute current gripper position features and positional embeddings.
    #
    #     Args:
    #         - curr_gripper: ((B, To), 1, 3+)
    #
    #     Returns:
    #         - curr_gripper_feats: ((B, To), 1, F)
    #         - curr_gripper_pos: ((B, To), 1, F, 2)
    #     """
    #     return self._encode_gripper(curr_gripper, self.curr_gripper_embed,
    #                                 context_feats, context)

    # def encode_goal_gripper(self, goal_gripper, context_feats, context):
    #     """
    #     Compute goal gripper position features and positional embeddings.
    #
    #     Args:
    #         - goal_gripper: (B, 3+)
    #
    #     Returns:
    #         - goal_gripper_feats: (B, 1, F)
    #         - goal_gripper_pos: (B, 1, F, 2)
    #     """
    #     goal_gripper_feats, goal_gripper_pos = self._encode_gripper(
    #         goal_gripper[:, None], self.goal_gripper_embed,
    #         context_feats, context
    #     )
    #     return goal_gripper_feats, goal_gripper_pos

    def encode_gripper(self, gripper, gripper_embed, context_feats, context):
        """
        Compute gripper position features and positional embeddings.

        Args:
            - gripper: ((B, To), 1, 3+)
            - context_feats: (B, npt, C)
            - context: (B, npt, 3)

        Returns:
            - gripper_feats: ((B, npt), 1, F)
            - gripper_pos: ((B, npt), 1, F, 2)
        """
        # Learnable embedding for gripper
        gripper_feats = gripper_embed.weight.unsqueeze(0).repeat(
            len(gripper), 1, 1
        )

        # Rotary positional encoding
        gripper_pos = self.relative_pe_layer(gripper[..., :3])
        context_pos = self.relative_pe_layer(context)

        gripper_feats = einops.rearrange(
            gripper_feats, 'b npt c -> npt b c'
        )
        context_feats = einops.rearrange(
            context_feats, 'b npt c -> npt b c'
        )
        gripper_feats = self.gripper_context_head(
            query=gripper_feats, value=context_feats,
            query_pos=gripper_pos, value_pos=context_pos
        )[-1]
        gripper_feats = einops.rearrange(
            gripper_feats, 'npt b c -> b npt c'
        )

        return gripper_feats, gripper_pos

    def encode_images(self, rgb):
        """
        Compute visual features/pos embeddings at different scales.

        Args:
            - rgb: (B, 3, H, W), pixel intensities

        Returns:
            - rgb_feats_pyramid: [(B, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """

        # Pass each view independently through backbone
        rgb = self.crop(rgb)
        rgb = self.normalize(rgb)
        rgb = torch.nn.functional.interpolate(rgb, size=self.fps_shape, mode='bilinear', align_corners=False)
        # rgb_features = self.backbone(rgb)

        # Pass visual features through feature pyramid network
        # rgb_features = self.feature_pyramid(rgb_features)

        # # Treat different cameras separately
        # pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")

        pcd_pyramid = [self.grid.repeat(rgb.shape[0], 1, 1)]

        bs = rgb.shape[0]
        rgb = einops.rearrange(rgb, "b c h w -> (b h w) c")
        rgb = self.rgb_lin(rgb)
        rgb = einops.rearrange(rgb, "(b npts) c -> npts b c", b=bs)
        rgb_pos = pcd_pyramid[0]
        rgb_pos = self.relative_pe_layer(rgb_pos)
        rgb = self.rgb_encoder(
            query=rgb,
            query_pos=rgb_pos,
        )[0]
        rgb = einops.rearrange(rgb, "npts b c -> b npts c", b=bs)
        rgb_feats_pyramid = [rgb]

        # for i in range(self.num_sampling_level):
        #     # Isolate level's visual features
        #     rgb_features_i = rgb_features[self.feature_map_pyramid[i]]
        #
        #     # Interpolate xy-depth to get the locations for this level
        #     feat_h, feat_w = rgb_features_i.shape[-2:]
        #     pcd_i = F.interpolate(
        #         pcd,
        #         (feat_h, feat_w),
        #         mode='bilinear'
        #     )
        #
        #     # Merge different cameras for clouds, separate for rgb features
        #     h, w = pcd_i.shape[-2:]
        #     pcd_i = einops.rearrange(
        #         pcd_i,
        #         "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        #     )
        #     rgb_features_i = einops.rearrange(
        #         rgb_features_i,
        #         "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras
        #     )
        #
        #     rgb_feats_pyramid.append(rgb_features_i)
        #     pcd_pyramid.append(pcd_i)

        return rgb_feats_pyramid, pcd_pyramid

    # def encode_instruction(self, instruction):
    #     """
    #     Compute language features/pos embeddings on top of CLIP features.
    #
    #     Args:
    #         - instruction: (B, max_instruction_length, 512)
    #
    #     Returns:
    #         - instr_feats: (B, 53, F)
    #         - instr_dummy_pos: (B, 53, F, 2)
    #     """
    #     instr_feats = self.instruction_encoder(instruction)
    #     # Dummy positional embeddings, all 0s
    #     instr_dummy_pos = torch.zeros(
    #         len(instruction), instr_feats.shape[1], 3,
    #         device=instruction.device
    #     )
    #     instr_dummy_pos = self.relative_pe_layer(instr_dummy_pos)
    #     return instr_feats, instr_dummy_pos

    def run_fps(self, context_features, context_pos):
        # context_features (Np, B, F)
        # context_pos (B, Np, F, 2)
        # outputs of analogous shape, with smaller Np
        npts, bs, ch = context_features.shape

        # Sample points with FPS
        sampled_inds = dgl_geo.farthest_point_sampler(
            einops.rearrange(
                context_features,
                "npts b c -> b npts c"
            ).to(torch.float64),
            max(npts // self.fps_subsampling_factor, 1), 0
        ).long()

        # Sample features
        expanded_sampled_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_context_features = torch.gather(
            context_features,
            0,
            einops.rearrange(expanded_sampled_inds, "b npts c -> npts b c")
        )

        # Sample positional embeddings
        _, _, ch, npos = context_pos.shape
        expanded_sampled_inds = (
            sampled_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ch, npos)
        )
        sampled_context_pos = torch.gather(
            context_pos, 1, expanded_sampled_inds
        )
        return sampled_context_features, sampled_context_pos

    # def vision_language_attention(self, feats, instr_feats):
    #     feats, _ = self.vl_attention[0](
    #         seq1=feats, seq1_key_padding_mask=None,
    #         seq2=instr_feats, seq2_key_padding_mask=None,
    #         seq1_pos=None, seq2_pos=None,
    #         seq1_sem_pos=None, seq2_sem_pos=None
    #     )
    #     return feats
