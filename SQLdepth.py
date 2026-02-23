import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import hub
import argparse
import os
import networks


class SQLdepth(nn.Module):
    def __init__(self, opt):
        super(SQLdepth, self).__init__()
        self.opt = opt


        self.encoder = networks.ResnetEncoderDecoder(num_layers=self.opt.num_layers, num_features=self.opt.num_features, model_dim=self.opt.model_dim)


        self.depth_decoder = networks.Depth_Decoder_QueryTr(in_channels=self.opt.model_dim, patch_size=self.opt.patch_size, dim_out=self.opt.dim_out, embedding_dim=self.opt.model_dim, 
                                                        query_nums=self.opt.query_nums, num_heads=4, min_val=self.opt.min_depth, max_val=self.opt.max_depth)

        if self.opt.load_pretrained_model:
            self.load_pretrained_model()

    def load_pretrained_model(self):
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        print("-> Loading pretrained encoder from ", self.opt.load_pt_folder)
        encoder_path = os.path.join(self.opt.load_pt_folder, "encoder.pth")
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        print("-> Loading pretrained depth decoder from ", self.opt.load_pt_folder)
        depth_decoder_path = os.path.join(self.opt.load_pt_folder, "depth.pth")
        loaded_dict_enc = torch.load(depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict_enc)


    def forward(self, x):
        x = self.encoder(x)
        return self.depth_decoder(x)["disp", 0]

