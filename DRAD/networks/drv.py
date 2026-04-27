import torch
import torch.nn as nn
from torch import nn

from configs.base import Config


class DRV(nn.Module):

    def __init__(
        self,
        cfg: Config,
    ):
        super().__init__()
        self.latent_size = cfg.dyad_latent_size
        self.bidirectional = cfg.dyad_bidirectional
        self.num_layers = cfg.dyad_num_layers
        self.hidden_size = cfg.dyad_hidden_size
        self.variable_length = cfg.dyad_variable_length
        rnn = eval("nn." + cfg.dyad_rnn_type.upper())

        self.rnn = rnn(
            cfg.dyad_encoder_embedding_size,
            cfg.dyad_hidden_size,
            num_layers=cfg.dyad_num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        self.outputs2embedding = nn.Linear(cfg.dyad_hidden_size * (2 if self.bidirectional else 1), cfg.dyad_output_embedding_size)

        self.encoder_filter = lambda x: x[:, :, : cfg.dyad_encoder_embedding_size]

    def forward(self, batch, noise_scale=1.0):
        input_sequence = batch["preprocess_inputs"].float().cuda()
        if len(input_sequence.shape) == 2:
            input_sequence = input_sequence.unsqueeze(0)
        en_input_sequence = self.encoder_filter(input_sequence)
        en_input_embedding = en_input_sequence.to(torch.float32)

        outputs, _ = self.rnn(en_input_embedding)

        log_p = self.outputs2embedding(outputs)
        return {
            "log_p": log_p,
            "mean": None,
            "log_v": None,
            "z": None,
            "mean_pred": None,
        }
