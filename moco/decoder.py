#! /user/bin/env python
# encoding: utf-8
'''
@author: sakura
@contact: saruka@mail.ustc.edu.cn
@file: decoder.py
@time: 1/13/23 1:42 PM
'''
from moco.GCN_Transformer_v2_masked import Transformer
import torch


class Decoder(torch.nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.GCN_Tran = Transformer(dim=opt.hidden_dim, n_heads=opt.heads, dim_ff=opt.d_ff, blocks=opt.blocks,
                                    dropout=opt.dropout)

    def forward(self, refine_feat, vid_len=None):
        mask = torch.ones((refine_feat.shape[0], refine_feat.shape[1]), device=refine_feat.device)
        if vid_len is not None:
            mask = mask * vid_len
        pose_feat_refine = self.GCN_Tran(refine_feat, mask)
        return pose_feat_refine


class Config:
    hidden_dim = 1536
    hidden_dim_face = 512
    proj_dropout = 0.0
    heads = 8
    d_ff = 2048
    blocks = 2
    dropout = 0.1
    input_size = 32
    temporal_pad = 0
    in_channels = 2
    out_channels = 3
    strategy = 'spatial'
    layout_encoder = 'stb'
    num_class = 128
    input_dim = 512
    inter_dist = False