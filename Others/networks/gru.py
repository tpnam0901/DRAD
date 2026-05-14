import torch
import torch.nn as nn
from configs.LSTM import Config


class GRU(nn.Module):

    def __init__(
        self,
        cfg: Config,
    ):
        super().__init__()
        self.bidirectional = True
        self.num_layers = 1
        self.hidden_size = 256
        rnn = eval("nn." + "gru".upper())

        self.rnn = rnn(
            2,
            256,
            num_layers=1,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        self.outputs2embedding = nn.Linear(256 * (2 if self.bidirectional else 1), 5)

        self.encoder_filter = lambda x: x[:, :, :2]

    def forward(self, batch):
        input_sequence = batch["normed_time_series"].float().cuda()
        if len(input_sequence.shape) == 2:
            input_sequence = input_sequence.unsqueeze(0)
        en_input_sequence = self.encoder_filter(input_sequence)
        en_input_embedding = en_input_sequence.to(torch.float32)

        outputs, _ = self.rnn(en_input_embedding)

        log_p = self.outputs2embedding(outputs)
        return {
            "log_p": log_p,
        }
