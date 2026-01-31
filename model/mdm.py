import torch
import torch.nn as nn
from model.rotation2xyz import Rotation2xyz
from model.BERT.BERT_encoder import load_bert
from model.mdm_modules import (
    PositionalEncoding,
    TimestepEmbedder,
    InputProcess,
    OutputProcess,
)


class MDM(nn.Module):
    def __init__(
        self,
        modeltype,
        njoints,
        nfeats,
        translation,
        pose_rep,
        glob,
        glob_rot,
        model_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        ablation=None,
        activation="gelu",
        legacy=False,
        data_rep="rot6d",
        dataset="humanml",
        clip_dim=512,
        arch="trans_dec",
        emb_trans_dec=False,
        clip_version="ViT-B/32",
        **kargs,
    ):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.model_dim = model_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get("normalize_encoder_output", False)

        self.cond_mode = kargs.get("cond_mode", "no_cond")
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.1)
        self.mask_frames = kargs.get("mask_frames", False)
        self.arch = arch
        if self.arch != "trans_dec":
            raise ValueError("Only 'trans_dec' architecture is supported in this minimal package.")
        if self.cond_mode != "text":
            raise ValueError("Only text conditioning is supported in this minimal package.")

        self.input_process = InputProcess(self.data_rep, self.input_feats, self.model_dim)
        self.emb_policy = kargs.get("emb_policy", "add")

        self.sequence_pos_encoder = PositionalEncoding(
            self.model_dim, self.dropout, max_len=kargs.get("pos_embed_max_len", 5000)
        )
        self.emb_trans_dec = emb_trans_dec

        self._init_backbone()
        self.embed_timestep = TimestepEmbedder(self.model_dim, self.sequence_pos_encoder)
        self._init_conditioning(kargs, clip_version)

        self.output_process = OutputProcess(
            self.data_rep, self.input_feats, self.model_dim, self.njoints, self.nfeats
        )

        self.rot2xyz = Rotation2xyz(device="cpu", dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith("clip_model.")]

    def _init_backbone(self):
        print("TRANS_DEC init")
        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)

    def _init_conditioning(self, kargs, clip_version):
        if self.cond_mode == "no_cond":
            return
        if "text" in self.cond_mode:
            print("EMBED TEXT")
            self.text_encoder_type = kargs.get("text_encoder_type", "bert")
            if self.text_encoder_type != "bert":
                raise ValueError("Only BERT text encoder is supported in this minimal package.")
            print("Loading BERT...")
            bert_model_path = "language_models/distilbert-base-uncased"
            self.clip_model = load_bert(bert_model_path)
            self.encode_text = self.bert_encode_text
            self.clip_dim = 768
            self.embed_text = nn.Linear(self.clip_dim, self.model_dim)

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[-2]
        if force_mask:
            return torch.zeros_like(cond)
        if self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(1, bs, 1)
            return cond * (1.0 - mask)
        return cond

    def bert_encode_text(self, raw_text):
        enc_text, mask = self.clip_model(raw_text)
        enc_text = enc_text.permute(1, 0, 2)
        mask = ~mask
        return enc_text, mask

    def _build_cond_emb(self, time_emb, y, bs):
        force_mask = y.get("uncond", False)
        text_mask = None
        if "text_embed" in y.keys():
            enc_text = y["text_embed"]
        else:
            enc_text = self.encode_text(y["text"])
        if isinstance(enc_text, tuple):
            enc_text, text_mask = enc_text
            if text_mask.shape[0] == 1 and bs > 1:
                text_mask = torch.repeat_interleave(text_mask, bs, dim=0)
        text_emb = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if self.emb_policy == "add":
            cond_emb = text_emb + time_emb
        else:
            cond_emb = torch.cat([time_emb, text_emb], dim=0)
            text_mask = torch.cat([torch.zeros_like(text_mask[:, 0:1]), text_mask], dim=1)
        return cond_emb, text_mask

    def _build_frames_mask(self, y, x, bs):
        frames_mask = None
        is_valid_mask = y["mask"].shape[-1] > 1
        if self.mask_frames and is_valid_mask:
            frames_mask = torch.logical_not(
                y["mask"][..., : x.shape[0]].squeeze(1).squeeze(1)
            ).to(device=x.device)
            if self.emb_trans_dec:
                step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device)
                frames_mask = torch.cat([step_mask, frames_mask], dim=1)
        return frames_mask

    def _forward_trans_dec(self, x, time_emb, cond_emb, frames_mask, text_mask):
        if self.emb_trans_dec:
            xseq = torch.cat((time_emb, x), axis=0)
        else:
            xseq = x
        xseq = self.sequence_pos_encoder(xseq)

        output = self.seqTransDecoder(
            tgt=xseq,
            memory=cond_emb,
            memory_key_padding_mask=text_mask,
            tgt_key_padding_mask=frames_mask,
        )

        if self.emb_trans_dec:
            output = output[1:]
        return output

    def forward(self, x, timesteps, y=None):
        bs, njoints, nfeats, nframes = x.shape
        time_emb = self.embed_timestep(timesteps)

        cond_emb, text_mask = self._build_cond_emb(time_emb, y, bs)

        x = self.input_process(x)
        frames_mask = self._build_frames_mask(y, x, bs)

        output = self._forward_trans_dec(x, time_emb, cond_emb, frames_mask, text_mask)

        output = self.output_process(output)
        return output

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self.rot2xyz, "smpl_model"):
            self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if hasattr(self.rot2xyz, "smpl_model"):
            self.rot2xyz.smpl_model.train(*args, **kwargs)
