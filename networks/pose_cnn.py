from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
import timm

class PoseCNN(nn.Module):
    """Relative pose prediction network.
    From SfM-Learner (https://arxiv.org/abs/1704.07813)
    """
    def __init__(self, num_input_frames, enc_name: str = 'resnet18', pretrained: bool = False):
        super().__init__()
        self.enc_name = enc_name
        self.pretrained = pretrained

        self.n_imgs = num_input_frames
        self.encoder = timm.create_model(enc_name, in_chans=3 * self.n_imgs, features_only=True, pretrained=pretrained)
        state_dict = torch.load('/home/MonoViT/eval/resnet18-f37072fd.pth') 
        modified_state_dict = {}
        for k, v in state_dict.items():
            if k == 'conv1.weight':
            # Duplicate the weights across channels and normalize
                modified_weight = torch.cat([v] * self.n_imgs, dim=1) / self.n_imgs
                modified_state_dict[k] = modified_weight
            else:
                modified_state_dict[k] = v
        self.encoder.load_state_dict(modified_state_dict, strict=False)
        self.n_chenc = self.encoder.feature_info.channels()

        self.squeeze = self.block(self.n_chenc[-1], 256, kernel_size=1)

        self.decoder = nn.Sequential(
            self.block(256, 256, kernel_size=3, stride=1, padding=1),
            self.block(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 6 * (self.n_imgs-1), kernel_size=1),
        )

    @staticmethod
    def block(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0) -> nn.Module:
        #Conv + ReLU.
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        """Pose network forward pass.
        """
        feat = self.encoder(x)
        out = self.decoder(self.squeeze(feat[-1]))
        out = 0.01 * out.mean(dim=(2, 3)).view(-1, self.n_imgs-1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

