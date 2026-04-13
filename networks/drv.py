import torch
import torch.nn as nn

from configs.DRV import Config


# DRV2
class DRV(nn.Module):
    def __init__(self, cfg: Config):
        super(DRV, self).__init__()
        self.input_features = cfg.input_features
        self.output_features = cfg.output_features

        # --------- Time series - AE
        self.rnn = nn.GRU(
            input_size=len(self.input_features),
            hidden_size=cfg.rnn_embed_dim,
            num_layers=cfg.rnn_num_layers,
            bidirectional=cfg.rnn_bidirectional,
            batch_first=True,
        )

        # -------- Output layers
        # Volt, current, soc, max_single_volt, min_single_volt, max_temp, min_temp
        self.regression = nn.Linear(cfg.rnn_embed_dim * 2 if cfg.rnn_bidirectional else 1, len(self.output_features))
        # # Mileage
        # self.regression_mileage = nn.Sequential(
        #     nn.Linear(128 * cfg.rnn_embed_dim * 2 if cfg.rnn_bidirectional else 1, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1),
        # )

    def forward(self, inputs):
        input_features = []
        for feature in self.input_features:
            input_features.append(inputs[feature])
        input_features = torch.stack(input_features, dim=2)

        # -------- Begin Time series features
        feat_series_encoder, _ = self.rnn(input_features)

        # Reconstruction for volt, current, soc, max_single_volt, min_single_volt, max_temp, min_temp
        logits_reconstruction = self.regression(feat_series_encoder)  # B x L x H_in

        # Mileage regression
        # logits_mileage = self.regression_mileage(feat_series_encoder.flatten(1))  # B x 1
        # logits_mileage = logits_mileage.squeeze()
        # -------- End output layers

        return {
            "logits_rec": logits_reconstruction,
            # "logits_mileage": logits_mileage,
        }


# class DRV(nn.Module):
#     def __init__(self, cfg: Config):
#         super(DRV, self).__init__()
#         self.input_features = cfg.input_features
#         self.output_features = cfg.output_features

#         # --------- Time series - AE
#         self.rnn = nn.GRU(
#             input_size=len(self.input_features),
#             hidden_size=cfg.rnn_embed_dim,
#             num_layers=cfg.rnn_num_layers,
#             bidirectional=cfg.rnn_bidirectional,
#             batch_first=True,
#         )

#         # -------- Output layers
#         # Volt, current, soc, max_single_volt, min_single_volt, max_temp, min_temp
#         self.regression = nn.Linear(cfg.rnn_embed_dim * 2 if cfg.rnn_bidirectional else 1, len(self.output_features))
#         # Mileage
#         self.regression_mileage = nn.Sequential(
#             nn.Linear(128 * cfg.rnn_embed_dim * 2 if cfg.rnn_bidirectional else 1, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1),
#         )

#     def forward(self, inputs):
#         input_features = []
#         for feature in self.input_features:
#             input_features.append(inputs[feature])
#         input_features = torch.stack(input_features, dim=2)

#         # -------- Begin Time series features
#         feat_series_encoder, _ = self.rnn(input_features)

#         # Reconstruction for volt, current, soc, max_single_volt, min_single_volt, max_temp, min_temp
#         logits_reconstruction = self.regression(feat_series_encoder)  # B x L x H_in

#         # Mileage regression
#         logits_mileage = self.regression_mileage(feat_series_encoder.flatten(1))  # B x 1
#         logits_mileage = logits_mileage.squeeze()
#         # -------- End output layers

#         return {
#             "logits_rec": logits_reconstruction,
#             "logits_mileage": logits_mileage,
#         }


# # EvalDistributedTestRepo
# class DRV(nn.Module):
#     def __init__(self, cfg: Config):
#         super(DRV, self).__init__()
#         self.input_features = cfg.input_features
#         self.output_features = cfg.output_features

#         # --------- Time series - AE
#         self.rnn = nn.GRU(
#             input_size=len(self.input_features),
#             hidden_size=cfg.rnn_embed_dim,
#             num_layers=cfg.rnn_num_layers,
#             bidirectional=cfg.rnn_bidirectional,
#             batch_first=True,
#         )

#         # -------- Output layers
#         # Volt, current, soc, max_single_volt, min_single_volt, max_temp, min_temp
#         self.outputs2embedding = nn.Linear(cfg.rnn_embed_dim * 2 if cfg.rnn_bidirectional else 1, len(self.output_features))
#         # # Mileage
#         # self.regression_mileage = nn.Sequential(
#         #     nn.Linear(128 * cfg.rnn_embed_dim * 2 if cfg.rnn_bidirectional else 1, 512),
#         #     nn.ReLU(),
#         #     nn.Linear(512, 1),
#         # )

#     def forward(self, inputs):
#         input_features = []
#         for feature in self.input_features:
#             input_features.append(inputs[feature])
#         input_features = torch.stack(input_features, dim=2)

#         # -------- Begin Time series features
#         feat_series_encoder, _ = self.rnn(input_features)

#         # Reconstruction for volt, current, soc, max_single_volt, min_single_volt, max_temp, min_temp
#         logits_reconstruction = self.outputs2embedding(feat_series_encoder)  # B x L x H_in

#         # Mileage regression
#         # logits_mileage = self.regression_mileage(feat_series_encoder.flatten(1))  # B x 1
#         # logits_mileage = logits_mileage.squeeze()
#         # -------- End output layers

#         return {
#             "logits_rec": logits_reconstruction,
#             # "logits_mileage": logits_mileage,
#         }
