import logging
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.TransGAN import Config


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        # Swap b,s,e -> s,b,e
        x = x.permute(1, 0, 2)

        x = x + self.pe[: x.size(0)]
        x = self.dropout(x)

        # Swap s,b,e -> b,s,e
        x = x.permute(1, 0, 2)
        return x


class TransGANEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super(TransGANEncoder, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = skip = self.layer_norm1(attn_output + x)
        x = self.linear(x)
        x = self.layer_norm2(x + skip)
        return x


class MaskSelfAttention(nn.Module):

    def __init__(
        self,
        d,
        H,
        T,
        bias=False,
        dropout=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        assert d % H == 0

        # key, query, value projections for all heads, but in a batch
        # output is 3X the dimension because it includes key, query and value
        self.c_attn = nn.Linear(d, 3 * d, bias=bias)

        # projection of concatenated attention head outputs
        self.c_proj = nn.Linear(d, d, bias=bias)

        # dropout modules
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.H = H
        self.d = d

        # causal mask to ensure that attention is only applied to
        # the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))

    def forward(self, x):
        B, T, _ = x.size()  # batch size, sequence length, embedding dimensionality

        # compute query, key, and value vectors for all heads in batch
        # split the output into separate query, key, and value tensors
        q, k, v = self.c_attn(x).split(self.d, dim=2)  # [B, T, d]

        # reshape tensor into sequences of smaller token vectors for each head
        k = k.view(B, T, self.H, self.d // self.H).transpose(1, 2)  # [B, H, T, d // H]
        q = q.view(B, T, self.H, self.d // self.H).transpose(1, 2)
        v = v.view(B, T, self.H, self.d // self.H).transpose(1, 2)

        # compute the attention matrix, perform masking, and apply dropout
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # [B, H, T, T]
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # compute output vectors for each token
        y = att @ v  # [B, H, T, d // H]

        # concatenate outputs from each attention head and linearly project
        y = y.transpose(1, 2).contiguous().view(B, T, self.d)
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransGANDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super(TransGANDecoder, self).__init__()
        self.multihead_attn1 = MaskSelfAttention(
            d=embed_dim,
            H=num_heads,
            T=512,
            dropout=dropout,
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.multihead_attn2 = MaskSelfAttention(
            d=embed_dim,
            H=num_heads,
            T=512,
            dropout=dropout,
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output = self.multihead_attn1(x)
        x = skip1 = self.layer_norm1(attn_output + x)

        attn_output = self.multihead_attn2(x)
        x = skip2 = self.layer_norm2(attn_output + skip1)

        x = self.linear(x)
        x = self.layer_norm3(x + skip2)
        return x


class TransGAN(nn.Module):
    def __init__(self, cfg: Config):
        super(TransGAN, self).__init__()

        self.projection = nn.Linear(4, cfg.transgan_input_embedding_dimension)
        self.pe = PositionalEncoding(d_model=cfg.transgan_input_embedding_dimension, dropout=cfg.transgan_dropout_pe)

        self.encoder = nn.Sequential(
            *[
                TransGANEncoder(
                    embed_dim=cfg.transgan_input_embedding_dimension,
                    num_heads=cfg.transgan_num_attention_heads,
                    dropout=cfg.transgan_dropout_attention,
                )
                for _ in range(cfg.transgan_num_encoder_layers)
            ]
        )

        self.decoder = nn.Sequential(
            *[
                TransGANDecoder(
                    embed_dim=cfg.transgan_input_embedding_dimension,
                    num_heads=cfg.transgan_num_attention_heads,
                    dropout=cfg.transgan_dropout_attention,
                )
                for _ in range(cfg.transgan_num_decoder_layers)
            ]
        )

        self.hidden = nn.Linear(cfg.transgan_input_embedding_dimension, cfg.transgan_hidden_layers)
        self.linear = nn.Linear(cfg.transgan_hidden_layers, 3)

    def create_time_series_data(self, inputs: Dict[str, torch.Tensor]):
        """
        Create time series data from the raw data.
        :param inputs: Dictionary containing raw data tensors
        """
        normed_current = inputs["normed_current"]
        normed_soc = inputs["normed_soc"]
        normed_min_cell_temperature = inputs["normed_min_cell_temperature"]
        normed_max_cell_temperature = inputs["normed_max_cell_temperature"]

        time_series_data = torch.stack(
            [
                normed_soc,
                normed_current,
                normed_min_cell_temperature,
                normed_max_cell_temperature,
            ],
            dim=-1,
        )  # B x L x H_in

        return time_series_data

    def forward(self, inputs):
        # -------- Begin time series
        time_series_data = self.create_time_series_data(inputs)
        projection = self.projection(time_series_data)
        projection_pe = self.pe(projection)
        encoder_feature = self.encoder(projection_pe)
        decoder_feature = self.decoder(encoder_feature)
        hidden = self.hidden(decoder_feature)
        # logits_reconstruction = torch.sigmoid(self.linear(hidden))
        logits_reconstruction = self.linear(hidden)

        return {
            "logits_rec": logits_reconstruction,
        }

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class Discriminator(TransGAN):
    def __init__(self, cfg: Config):
        super(Discriminator, self).__init__(cfg)
        self.projection = nn.Linear(3, cfg.transgan_input_embedding_dimension)
        self.linear = nn.Linear(cfg.transgan_hidden_layers, 1)

    def forward(self, inputs):
        projection = self.projection(inputs)
        logging.debug(f"projection: {projection.shape}")

        projection_pe = self.pe(projection)
        encoder_feature = self.encoder(projection_pe)
        logging.debug(f"encoder_feature: {encoder_feature.shape}")

        decoder_feature = self.decoder(encoder_feature)
        logging.debug(f"decoder_feature: {decoder_feature.shape}")

        hidden = self.hidden(decoder_feature)
        logits = self.linear(hidden)
        logging.debug(f"logits: {logits.shape}")

        return logits

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
