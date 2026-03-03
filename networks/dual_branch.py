import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as T
from pytorch3d.transforms import axis_angle_to_quaternion
from networks.modules import (
    DoubleConv,
    Down,
    Up,
    FourierEncoding3D
)

class DetailBlock(nn.Module):
    """
    A small convolutional block to enhance local high-frequency details.
    """
    def __init__(self, in_ch, expr_dim, hidden_ch=64):
        super().__init__()
        self.expr_proj = nn.Sequential(
            nn.Linear(expr_dim, hidden_ch),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(in_ch + in_ch + hidden_ch, hidden_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(hidden_ch, hidden_ch, 1)

    def forward(self, global_feat, disp_map, expr, mask):
        """
        global_feat: [B, C, H, W]
        disp_map:    [B, C, H, W]
        expr:        [B, expr_dim]
        mask:        [1, 1, H, W]  (binary mask)
        """
        B, C, H, W = global_feat.shape

        e = self.expr_proj(expr).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        disp_map_masked = disp_map * mask
        global_feat_masked = global_feat * mask

        x = torch.cat([disp_map_masked, global_feat_masked, e], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv_out(x)

        return x * mask


class DualBranchUNet(nn.Module):
    def __init__(
        self,
        device,
        uv_sample_coords,
        uv_mask,
        reference_image=None,
        position_map=None,
        condition_dim=118,
        bilinear=True,
    ):
        super().__init__()
        self.device = device
        self.eye_mask = uv_mask["eye_region"]
        self.nose_mask = uv_mask["nose"]
        self.lips_mask = uv_mask["lips"]
        self.forehead_mask = uv_mask["forehead"]

        self.transform = T.Compose([T.Resize((256, 256), antialias=True), T.ToTensor()])

        self.resize = T.Resize((256, 256), antialias=True)

        if reference_image is not None:
            self.register_buffer("reference_image", self.transform(reference_image).unsqueeze(0).detach().to(device))
        else:
            self.register_buffer("reference_image", torch.empty(1, 3, 256, 256))

        if position_map is not None:
            self.register_buffer("position_map", position_map.detach().float().to(device))
        else:
            self.register_buffer("position_map", torch.empty(1, 3, 256, 256))

        self.uv_sample_coords = uv_sample_coords.detach().float().to(device)
        self.condition_dim = condition_dim
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        #=========== Encoder ===========
        self.inc = DoubleConv(3, 64, mid_channels=32)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)

        #=========== Process different features ===========
        self.fourier = FourierEncoding3D(num_bands=6, max_freq=10.0)
        self.uv_conv = nn.Sequential(nn.Conv2d(3 + 6 * 6, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.disp_conv = nn.Sequential(nn.Conv2d(3 + 6 * 6, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.expr_embed = nn.Sequential(nn.Linear(100, 96), nn.ReLU(inplace=True))
        self.pose_embed = nn.Sequential(nn.Linear(15, 16), nn.ReLU(inplace=True))
        self.tran_embed = nn.Sequential(nn.Linear(3, 16), nn.ReLU(inplace=True))

        #=========== Global Decoder ============
        self.global_decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        #=========== Local Decoder ============
        self.local_decoder = DetailBlock(in_ch=64, expr_dim=100, hidden_ch=64)

        #=========== Fuse Layer ===========
        self.final_fuse = nn.Sequential(
            nn.Conv2d(128 + 4 * 64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 13, kernel_size=1),
        )

        self._initialize_weights()

    def forward(
        self, flame_param, timestep, displacement_map
    ):
        # process condition
        expr = flame_param["expr"][timestep].unsqueeze(0)  # 1*100
        neck_pose = flame_param["neck_pose"][timestep].unsqueeze(0)  # 1*3
        jaw_pose = flame_param["jaw_pose"][timestep].unsqueeze(0)  # 1*3
        eyes_pose = flame_param["eyes_pose"][timestep].unsqueeze(0)  # 1*6
        rotation = flame_param["rotation"][timestep].unsqueeze(0)  # 1*3
        pose = torch.cat((neck_pose, jaw_pose, eyes_pose, rotation), dim=1)
        translation = flame_param["translation"][timestep].unsqueeze(0)  # 1*3

        #========== Encoder ==========
        x1 = self.inc(self.reference_image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)

        #========== Global Decoder ==========
        pos_fourier = self.fourier(self.position_map.float())
        uv_feat = self.uv_conv(pos_fourier)
        expr_emb = self.expr_embed(expr)
        expr_feat = (expr_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d3.size(2), d3.size(3)))
        pose_emb = self.pose_embed(pose)
        pose_feat = (pose_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d3.size(2), d3.size(3)))
        trans_emb = self.tran_embed(translation)
        trans_feat = (trans_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d3.size(2), d3.size(3)))
        global_feat = torch.cat([d3, uv_feat, expr_feat, pose_feat, trans_feat], dim=1)
        offset_global = self.global_decoder(global_feat)

        #========== Local Decoder ==========
        disp_f = self.fourier(displacement_map.float())
        disp = self.disp_conv(disp_f)

        local_eye = self.local_decoder(d3, disp, expr, self.eye_mask)
        local_nose = self.local_decoder(d3, disp, expr, self.nose_mask)
        local_lips = self.local_decoder(d3, disp, expr, self.lips_mask)
        local_forehead = self.local_decoder(d3, disp, expr, self.forehead_mask)

        #========= Fuse Layer =========
        unified_feat = torch.concat(
            [offset_global, local_eye, local_nose, local_lips, local_forehead], dim=1
        )
        offset_out = self.final_fuse(unified_feat)
        offset_final = self._offset_attr_process(offset_out, self.uv_sample_coords)

        return offset_final

    def forward_once(self):
        # for testing, only calculate d3 once
        x1 = self.inc(self.reference_image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        self.d3 = self.up3(d2, x1)
        pos_fourier = self.fourier(self.position_map.float())
        self.uv_feat = self.uv_conv(pos_fourier)


    def _decode(self, flame_param, timestep, displacement_map):
        expr = flame_param["expr"][timestep].unsqueeze(0)  # 1*100
        neck_pose = flame_param["neck_pose"][timestep].unsqueeze(0)  # 1*3
        jaw_pose = flame_param["jaw_pose"][timestep].unsqueeze(0)  # 1*3
        eyes_pose = flame_param["eyes_pose"][timestep].unsqueeze(0)  # 1*6
        rotation = flame_param["rotation"][timestep].unsqueeze(0)  # 1*3
        pose = torch.cat((neck_pose, jaw_pose, eyes_pose, rotation), dim=1)
        translation = flame_param["translation"][timestep].unsqueeze(0)  # 1*3

        #========= Global Decoder ===========
        H, W = self.d3.shape[2], self.d3.shape[3]
        pos_fourier = self.fourier(self.position_map.float())
        uv_feat = self.uv_conv(pos_fourier)
        expr_emb = self.expr_embed(expr)
        expr_feat = (
            expr_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        )
        pose_emb = self.pose_embed(pose)
        pose_feat = (
            pose_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        )
        trans_emb = self.tran_embed(translation)
        trans_feat = (
            trans_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        )
        global_feat = torch.cat([self.d3, uv_feat, expr_feat, pose_feat, trans_feat], dim=1)
        offset_global = self.global_decoder(global_feat)

        #========= Local Decoder ==========
        disp_f = self.fourier(displacement_map.float())
        disp = self.disp_conv(disp_f)

        local_eye = self.local_decoder(self.d3, disp, expr, self.eye_mask)
        local_nose = self.local_decoder(self.d3, disp, expr, self.nose_mask)
        local_lips = self.local_decoder(self.d3, disp, expr, self.lips_mask)
        local_forehead = self.local_decoder(self.d3, disp, expr, self.forehead_mask)

        #========= Fuse Layer ==========
        unified_feat = torch.concat(
            [offset_global, local_eye, local_nose, local_lips, local_forehead], dim=1
        )
        offset_out = self.final_fuse(unified_feat)
        offset_final = self._offset_attr_process(offset_out, self.uv_sample_coords)

        return offset_final

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def update_uv_coords(self, uv_sample_coords):
        self.uv_sample_coords = uv_sample_coords.detach().float().to(self.device)

    def _offset_attr_process(self, offset_map, uv_coords):
        """
        Args:
            offset_map: [B, 13, 512, 512]
            uv_coords:  [B, N, 2]
        Returns:
            offset:     [B, N, 13]
        """
        # sample per-point offset from the map
        offset = self._uv_look_up(texture=offset_map, uv_coords=uv_coords)

        # apply activations to individual channels
        return torch.cat(
            [
                self._offset_position_activation(offset[:, :, 0:3]),
                self._offset_scaling_activation(offset[:, :, 3:6]),
                self._rotation_activation(offset[:, :, 6:9]),  # (B, N, 4)
                self._offset_color_activation(offset[:, :, 9:12]),
                self._offset_opacity_activation(offset[:, :, 12:13]),
            ],
            dim=-1,
        )

    @staticmethod
    def _offset_position_activation(input):
        """
        Force the NN output serves as position to be between [-0.1, 0.1]
        """
        return torch.tanh(input) * 0.1

    @staticmethod
    def _offset_color_activation(input):
        """
        Force the NN output serves as color to be between [-0.7, 0.7].
        """
        return torch.tanh(input) * 0.7

    @staticmethod
    def _offset_scaling_activation(input):
        """
        Force the NN output serves as scaling within proper range [0, +inf].
        """
        return torch.exp(input)

    @staticmethod
    def _offset_opacity_activation(input):
        """
        Force the NN output serves as scaling within proper range [-0.5, 0.5].
        """
        return torch.tanh(input) * 0.5

    def _rotation_activation(self, input: torch.Tensor) -> torch.Tensor:
        """
        Convert NN output to quaternion rotation [r, x, y, z] with proper normalization.
        Supports only [B, N, 3]input.

        Args:
            input: Tensor of shape [B, N, 3] — axis-angle representation.
        Returns:
            Tensor of shape [B, N, 4] — quaternion in [r, x, y, z] format.
        """
        input = torch.tanh(input) * (torch.pi)  # scale to [-π, π]
        quat = axis_angle_to_quaternion(input)  # [B, N, 4] wxyz

        return quat.contiguous()

    def _uv_look_up(
        self, texture: torch.Tensor, uv_coords: torch.Tensor
    ) -> torch.Tensor:
        """Use sampling to get gs value
        Args:
            texture:    [B, C, H, W] texture color
            uv_coords:  [B, N, 2] texture uv
        Returns:
            values:     [B, N, C] texture sampling
        """
        grid = uv_coords * 2.0 - 1.0  # [B, N, 2]
        grid = grid.unsqueeze(2)  # [B, N, 1, 2]

        sampled = F.grid_sample(
            texture, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        sampled_value = sampled.squeeze(3).permute(0, 2, 1)  # [B, N, C]

        return sampled_value