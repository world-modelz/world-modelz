import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import conv1x1, conv3x3, Residual


class SimpleDiffusionModel(nn.Module):
    def __init__(self, d_model=128, dropout=.1, num_layers=10, d_pos=32):
        super().__init__()

        self.d_model = d_model

        def normalize(num_channels):
            return nn.GroupNorm(num_groups=32, num_channels=num_channels)
            #return nn.BatchNorm2d(num_channels)

        def nonlinearity(inplace=True):
            return nn.SiLU(inplace=inplace)
            #return nn.LeakyReLU(inplace=inplace)

        # increase dimensionality of input form 2 to target d_model
        self.init_block = nn.Sequential(
            # conv1x1(in_planes=2, out_planes=d_model*2, stride=1, bias=True),
            # normalize(num_channels=d_model*2),
            # nonlinearity(),
            # conv3x3(in_planes=d_model*2, out_planes=d_model*2, stride=1, bias=True),
            conv3x3(in_planes=2, out_planes=d_model*2, stride=1, bias=True),
            normalize(num_channels=d_model*2),
            nonlinearity(),
            conv1x1(d_model*2, d_model),
            normalize(num_channels=d_model),
            nonlinearity()
        )

        self.pos_dim = d_pos

        d_model2 = d_model + self.pos_dim

        # add some res layers
        layers = []
        for _ in range(num_layers):
            layers.append(Residual(d_model2, d_model2 * 2, 1, normalize, nonlinearity))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        self.res_stack = nn.Sequential(*layers)

        # decoder projection
        self.decoder_block = nn.Sequential(
            conv3x3(in_planes=d_model2, out_planes=d_model, stride=1),
            normalize(num_channels=d_model),
            nonlinearity(inplace=True),
            conv1x1(d_model, 2, bias=True)
        )

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Linear) 
                or isinstance(m, nn.Embedding)):
                pass
                #nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.orthogonal_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_timestep_embedding(self, batch_size, dim, t, stretch=50*math.pi, max_period=100.0):
        div_term = torch.exp(torch.arange(0, dim, 2, device=t.device) * -(math.log(max_period) / dim)) * stretch
        pe = torch.zeros(t.size(0), dim, device=t.device)
        pe[:, 0::2] = torch.sin(t * stretch * div_term)
        pe[:, 1::2] = torch.cos(t * stretch * div_term)
        return pe

    def forward(self, x, t):
        
        x = self.init_block(x)
        
        # add positional embedding for timestep
        time_emb = self.get_timestep_embedding(x.size(1), self.pos_dim, t).unsqueeze(-1).unsqueeze(-1)
        time_emb = time_emb.repeat(1, 1, x.shape[-2], x.shape[-1])

        # add time
        x = torch.cat([x, time_emb], dim=-3)
        x = self.res_stack(x)
        
        # add some res-layers 
        x = self.decoder_block(x)
        
        return x
