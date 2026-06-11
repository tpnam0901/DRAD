import torch
import torch.nn as nn
from configs.LSTM import Config


class LSTM(nn.Module):
    def __init__(self, cfg: Config):
        super(LSTM, self).__init__()

        # --------- Time series - VAE
        self.fusion_embed_dim = cfg.fusion_embed_dim
        self.bidirectional = cfg.rnn_bidirectional
        self.num_layers = cfg.rnn_num_layers

        self.rnn = nn.LSTM(
            input_size=cfg.rnn_input_size,
            hidden_size=cfg.fusion_embed_dim // 2 if cfg.rnn_bidirectional else cfg.fusion_embed_dim,
            num_layers=cfg.rnn_num_layers,
            bidirectional=cfg.rnn_bidirectional,
            batch_first=True,
        )

        # -------- Output layers
        # Volt, current, soc, max_single_volt, min_single_volt, max_temp, min_temp
        self.regression = nn.Linear(cfg.fusion_embed_dim, 5)
        # Mileage
        self.regression_mileage = nn.Sequential(
            nn.Linear(128 * cfg.fusion_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, inputs):
        # normed_time_series = torch.stack(
        #     [
        #         inputs["normed_soc"],
        #         inputs["normed_current"],
        #         inputs["normed_min_cell_temperature"],
        #         inputs["normed_max_cell_temperature"],
        #         # inputs["normed_voltage"],
        #         # inputs["normed_max_cell_voltage"],
        #         # inputs["normed_min_cell_voltage"],
        #     ],
        #     dim=-1,
        # )
        normed_time_series = inputs["preprocess_inputs"].float().cuda()
        # -------- Begin Time series features
        feat_series_encoder, _ = self.rnn(normed_time_series)
        # -------- End Time series features

        # -------- Begin output layers
        # Reconstruction for volt, current, soc, max_single_volt, min_single_volt, max_temp, min_temp
        logits_reconstruction = self.regression(feat_series_encoder)  # B x L x H_in

        # Mileage regression
        logits_mileage = self.regression_mileage(feat_series_encoder.flatten(1))  # B x 1
        logits_mileage = logits_mileage.squeeze()
        # -------- End output layers

        return {
            "log_p": logits_reconstruction,
            "logits_mileage": logits_mileage,
        }
