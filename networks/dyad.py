import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from configs.DyAD import Config


def to_var(x):
    """
    var to gpu
    :param x: data or model
    :return: x
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


class DyAD(nn.Module):

    def __init__(
        self,
        cfg: Config,
    ):
        super().__init__()
        rnn_type = cfg.dyad_rnn_type
        hidden_size = cfg.dyad_hidden_size
        latent_size = cfg.dyad_latent_size
        num_layers = cfg.dyad_num_layers
        bidirectional = cfg.dyad_bidirectional
        variable_length = cfg.dyad_variable_length
        encoder_embedding_size = cfg.dyad_encoder_embedding_size
        decoder_embedding_size = cfg.dyad_decoder_embedding_size
        output_embedding_size = cfg.dyad_output_embedding_size

        self.feature_split = cfg.dyad_decoder_embedding_size
        self.latent_size = latent_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.variable_length = variable_length
        self.noise_scale = cfg.dyad_noise_scale

        rnn = eval("nn." + rnn_type.upper())

        self.encoder_rnn = rnn(
            encoder_embedding_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.decoder_rnn = rnn(
            decoder_embedding_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2log_v = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2embedding = nn.Linear(hidden_size * (2 if bidirectional else 1), output_embedding_size)
        self.mean2latent = nn.Sequential(nn.Linear(latent_size, int(hidden_size / 2)), nn.ReLU(), nn.Linear(int(hidden_size / 2), 1))

    # def forward(self, input_sequence, encoder_filter, decoder_filter, seq_lengths, noise_scale=1.0):
    def forward(self, inputs):  # type: ignore
        input_sequence = inputs["normed_time_series"]
        seq_lengths = None  # This value is always None in official code

        batch_size = input_sequence.size(0)
        en_input_embedding = input_sequence
        if self.variable_length:
            en_input_embedding = pack_padded_sequence(en_input_embedding, seq_lengths, batch_first=True)
        output, hidden = self.encoder_rnn(en_input_embedding)
        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        mean = self.hidden2mean(hidden)
        log_v = self.hidden2log_v(hidden)
        std = torch.exp(0.5 * log_v)
        mean_pred = self.mean2latent(mean)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        if self.training:
            z = z * std * self.noise_scale + mean
        else:
            z = mean
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        de_input_sequence = input_sequence[:, :, : self.feature_split]
        de_input_embedding = de_input_sequence.to(torch.float32)
        if self.variable_length:
            de_input_embedding = pack_padded_sequence(de_input_embedding, seq_lengths, batch_first=True)

            outputs, _ = self.decoder_rnn(de_input_embedding, hidden)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, _ = self.decoder_rnn(de_input_embedding, hidden)
        log_p = self.outputs2embedding(outputs)
        return {
            "log_p": log_p,
            "mean": mean,
            "log_v": log_v,
            "z": z,
            "mean_pred": mean_pred,
        }
