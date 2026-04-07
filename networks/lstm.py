import torch.nn as nn

from configs.LSTM import Config


class LSTM(nn.Module):
    def __init__(self, cfg: Config):
        super(LSTM, self).__init__()

        # --------- Time series - VAE
        self.fusion_embed_dim = 100
        self.bidirectional = True
        self.num_layers = 3

        self.rnn = nn.LSTM(
            input_size=7,
            hidden_size=self.fusion_embed_dim // 2 if self.bidirectional else self.fusion_embed_dim,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        # -------- Output layers
        # Volt, current, soc, max_single_volt, min_single_volt, max_temp, min_temp
        self.regression = nn.Linear(self.fusion_embed_dim, 1)
        # Mileage
        self.regression_mileage = nn.Sequential(
            nn.Linear(128 * self.fusion_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, inputs):

        # -------- Begin Time series features
        feat_series_encoder, _ = self.rnn(inputs["normed_time_series"])
        # -------- End Time series features

        # -------- Begin output layers
        # Reconstruction for volt, current, soc, max_single_volt, min_single_volt, max_temp, min_temp
        logits_reconstruction = self.regression(feat_series_encoder)  # B x L x H_in

        # Mileage regression
        logits_mileage = self.regression_mileage(feat_series_encoder.flatten(1))  # B x 1
        logits_mileage = logits_mileage.squeeze()
        # -------- End output layers

        return {
            "logits_rec": logits_reconstruction,
            "logits_mileage": logits_mileage,
        }
