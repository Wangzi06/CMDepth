import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from mamba_ssm import Mamba
class MambaEncoder(nn.Module):
    def __init__(self, d_model,bimamba=False, layer_norm_epsilon = 1e-6):
        super(MambaEncoder, self).__init__()
        inner_dim = 8 * d_model
        self.norm_mamba = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.mamba_blocks = Mamba(d_model=d_model, bimamba=bimamba)
        self.ln_2 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2) #for 192x640 kernel_size = 3, padding = 1
        self.mlp_channels = nn.Sequential(
            nn.Linear(d_model, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model)
        )
    def forward(self, x):
        local = self.conv1d(x.permute(0, 2, 1))
        x = self.mamba_blocks(self.norm_mamba(x)) 
        x = x + local.permute(0,2,1)
        x = x + self.mlp_channels(self.ln_2(x))
        return x

