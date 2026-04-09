from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.num_epochs = 100

        self.brand_num: int = 2

        self.model_type = "TransGAN"
        self.name = "TransGAN_brand{}".format(self.brand_num)

        self.unlock()
        # Add more configuration parameters as needed

        self.transgan_gen_warm_up: int = -1
        self.transgan_num_encoder_layers: int = 3
        self.transgan_num_decoder_layers: int = 3

        self.transgan_num_attention_heads: int = 8

        self.transgan_input_embedding_dimension: int = 128
        self.transgan_hidden_layers: int = 512
        self.transgan_dropout_attention: float = 0.1
        self.transgan_dropout_pe: float = 0.1

        # Lock the config to prevent further modifications
        self.lock()
