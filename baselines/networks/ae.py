import torch.nn as nn

from configs.AE import Config


class AE(nn.Module):
    def __init__(self, cfg: Config):
        super(AE, self).__init__()

        # --------- Time series - VAE
        self.fusion_embed_dim = cfg.fusion_embed_dim
        self.rnn_input_size = cfg.rnn_input_size
        self.projection = nn.Linear(cfg.rnn_input_size, cfg.fusion_embed_dim)
        self.flatten = nn.Flatten()

        # Encoder
        self.l1_enc = nn.Linear(cfg.fusion_embed_dim * 128, 1024)
        self.l1_enc_bn = nn.BatchNorm1d(1024)
        self.l2_enc = nn.Linear(1024, 256)
        self.l2_enc_bn = nn.BatchNorm1d(256)
        self.l3_enc = nn.Linear(256, 64)
        self.l3_enc_bn = nn.BatchNorm1d(64)
        self.l4_enc = nn.Linear(64, 16)
        self.l4_enc_bn = nn.BatchNorm1d(16)

        # Decoder
        self.l1_dec = nn.Linear(16, 64)
        self.l1_dec_bn = nn.BatchNorm1d(64)
        self.l2_dec = nn.Linear(64, 256)
        self.l2_dec_bn = nn.BatchNorm1d(256)
        self.l3_dec = nn.Linear(256, 1024)
        self.l3_dec_bn = nn.BatchNorm1d(1024)
        self.l4_dec = nn.Linear(1024, cfg.rnn_input_size * 128)
        self.l4_dec_bn = nn.BatchNorm1d(cfg.rnn_input_size * 128)

        # -------- Output layers
        # Mileage
        self.regression_mileage = nn.Sequential(
            nn.Linear(cfg.rnn_input_size * 128, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, inputs):

        x = inputs["normed_time_series"]
        x = self.projection(x)  # B x L x H_fusion
        x = self.flatten(x)  # B x (L * H_fusion)

        # Encoder
        x = self.l1_enc(x)
        x = self.l1_enc_bn(x)
        x = self.l2_enc(x)
        x = self.l2_enc_bn(x)
        x = self.l3_enc(x)
        x = self.l3_enc_bn(x)
        x = self.l4_enc(x)
        x = self.l4_enc_bn(x)

        # Decoder
        x = self.l1_dec(x)
        x = self.l1_dec_bn(x)
        x = self.l2_dec(x)
        x = self.l2_dec_bn(x)
        x = self.l3_dec(x)
        x = self.l3_dec_bn(x)
        x = self.l4_dec(x)
        x = self.l4_dec_bn(x)

        logits_reconstruction = x.view(-1, 128, self.rnn_input_size)  # B x L x H_fusion

        # Mileage regression
        logits_mileage = self.regression_mileage(x)  # B x 1
        logits_mileage = logits_mileage.squeeze()
        # -------- End output layers

        return {
            "logits_rec": logits_reconstruction,
            "logits_mileage": logits_mileage,
        }
