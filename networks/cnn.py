import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(4, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=7, stride=1, padding=3)
        self.selu = nn.SELU()

        self.rnn = nn.GRU(
            input_size=32,
            hidden_size=90,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.5,
        )

        # -------- Output layers
        # Volt, current, soc, max_single_volt, min_single_volt, max_temp, min_temp
        self.regression = nn.Linear(90 * 2, 3)
        # Mileage
        self.regression_mileage = nn.Sequential(
            nn.Linear(128 * 90 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, inputs):
        # Swap b,l,c to b,c,l for Conv1d
        normed_time_series = torch.stack(
            [
                inputs["normed_soc"],
                inputs["normed_current"],
                inputs["normed_min_cell_temperature"],
                inputs["normed_max_cell_temperature"],
                # inputs["normed_voltage"],
                # inputs["normed_max_cell_voltage"],
                # inputs["normed_min_cell_voltage"],
            ],
            dim=-1,
        )

        x = normed_time_series.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.maxpool(x)
        x = self.selu(x)

        # Swap back to b,l,c for RNN
        rnn_input = x.permute(0, 2, 1)

        # -------- Begin Time series features
        feat_series_encoder, _ = self.rnn(rnn_input)
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
