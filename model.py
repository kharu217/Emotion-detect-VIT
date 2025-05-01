import torch
import torch.nn as nn

import numpy as np

from utils import *

class PatchEmbedding(nn.Module) :
    def __init__(self, d_model, image_size, patch_size, n_channels):
        super().__init__()

        self.d_model = d_model # Dimensionality of Model
        self.image_size = image_size # Image size
        self.patch_size = patch_size # Patch size
        self.n_channels = n_channels # Number of channels

        # The kernel size is same as the stride so this will make each local patches
        # with linear projection.
        self.linear_project = nn.Conv2d(self.n_channels, self.d_patch, kernel_size=self.patch_size, stride=self.patch_size)

    # B : Batch Size
    # C : Image Channels
    # H : Image height
    # W : Image Width
    # P_col : Patch column
    # P_row : Pathch Row

    def forward(self, x) :
        x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)
        
        x = x.flatten(2) # (B, d_model, P_Col, P_row) -> (B, d_model, P)

        x = x.transpose(1, 2) # (B, d_model, P) ->      (B, P, d_model)

class PositionalEncoding(nn.Module) :
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        pe = torch.zeros(max_seq_length, d_model)

        # Creating positional encoding
        for pos in range(max_seq_length) :
            for i in range(d_model) :
                if i % 2 == 0 :
                    pe[pos][i] = np.sin(pos/(10000**(i/d_model)))
                else :
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x) :
        # Expand to have class token for every image in batch
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)

        # Adding class tokens to beginning of each embedding
        x = torch.cat((tokens_batch, x), dim=1)

        # Add positional encoding to embeddings
        x = x + self.pe

        return x