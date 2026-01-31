import math
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, model_dim, sequence_pos_encoder):
        super().__init__()
        self.model_dim = model_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.model_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def _positional_embedding(self, timesteps):
        device = timesteps.device
        dtype = self.sequence_pos_encoder.pe.dtype
        timesteps_fp32 = timesteps.to(dtype=torch.float32)
        position = timesteps_fp32.unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.model_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.model_dim)
        )
        emb = torch.zeros((timesteps.shape[0], self.model_dim), device=device, dtype=torch.float32)
        emb[:, 0::2] = torch.sin(position * div_term)
        emb[:, 1::2] = torch.cos(position * div_term)
        return emb.to(dtype=dtype).unsqueeze(1)

    def forward(self, timesteps):
        if timesteps.is_floating_point():
            pos_emb = self._positional_embedding(timesteps)
        else:
            pos_emb = self.sequence_pos_encoder.pe[timesteps]
        return self.time_embed(pos_emb).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, model_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.model_dim = model_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.model_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.model_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, model_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.model_dim = model_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.model_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.model_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
            output = output.reshape(nframes, bs, self.njoints, self.nfeats)
            output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
            return output
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)
            output = output.reshape(nframes, bs, self.njoints, self.nfeats)
            output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
            return output
        else:
            raise ValueError
