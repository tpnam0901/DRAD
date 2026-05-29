from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
        # Seed used in the paper: 980, 2025, 3189, 6315, 8455
        self.brand_num = 3
        self.seed = 980

        self.model_type = "DyAD"
        self.name = "DyAD_{}_{}".format(self.brand_num, self.seed)

        if self.model_type == "DyAD":
            self.batch_size = 128
            self.optimizer: str = "adamw"
            self.weight_decay = 1e-6
            if self.brand_num == 2:
                self.learning_rate = 0.0001
            else:
                self.learning_rate = 0.005
            self.lr_scheduler = "StepLR"
            self.lr_T_max: int = self.num_epochs
            cosine_factor = 0.1  # 1.0 for all and fivefold settings
            self.lr_eta_min: float = self.learning_rate * cosine_factor

        self.unlock()
        # Add more configuration parameters as needed

        self.dyad_rnn_type: str = "gru"  ########### Brand 1 | Brand 2 | Brand 3 |  All  | five folds

        self.dyad_hidden_size: int = 64  ###########   128   |   1024  | 256     |  64   |     64
        if self.model_type == "DyAD":
            temp_list = [128, 1024, 256]
            self.dyad_hidden_size = temp_list[self.brand_num - 1]

        self.dyad_latent_size: int = 32  ###########   8     |   24    | 16      |  32   |     32
        if self.model_type == "DyAD":
            temp_list = [8, 24, 16]
            self.dyad_latent_size = temp_list[self.brand_num - 1]

        self.dyad_num_layers: int = 2  #############   2     |   1     | 1       |  1    |     1
        if self.model_type == "DyAD":
            temp_list = [2, 1, 1]
            self.dyad_num_layers = temp_list[self.brand_num - 1]

        self.dyad_encoder_embedding_size: int = 7  #   7     |   7     | 7       |  6    |     6

        self.dyad_decoder_embedding_size: int = 7  #   2     |   4     | 2       |  2    |     2
        if self.model_type == "DyAD":
            temp_list = [2, 4, 2]
            self.dyad_decoder_embedding_size = temp_list[self.brand_num - 1]

        self.dyad_output_embedding_size: int = 7  ##   5     |   3     | 5       |  4    |     4
        if self.model_type == "DyAD":
            temp_list = [5, 3, 5]
            self.dyad_output_embedding_size = temp_list[self.brand_num - 1]

        self.dyad_anneal0: float = 0.1
        if self.model_type == "DyAD" and self.brand_num == 1:
            self.dyad_anneal0 = 0.01

        self.dyad_nll_weight: float = 10
        if self.model_type == "DyAD" and self.brand_num == 2:
            self.dyad_nll_weight = 5

        self.dyad_latent_label_weight: float = 0.001
        if self.model_type == "DyAD" and self.brand_num == 2:
            self.dyad_latent_label_weight = 1
        self.dyad_noise_scale: float = 0.01
        if self.model_type == "DyAD" and self.brand_num == 1:
            self.dyad_noise_scale = 1

        self.dyad_k: float = 0.0025
        self.dyad_x0: int = 500

        self.dyad_variable_length: bool = False
        self.dyad_dim_feedforward: int = 2048
        self.dyad_nhead: int = 2
        self.dyad_kernel_size: int = 3
        self.dyad_bidirectional: bool = True
        self.dyad_anneal_function: str = "linear"

        if self.model_type == "DyAD":
            self.dyad_sample_length: int = 50

        # Lock the config to prevent further modifications
        self.lock()
