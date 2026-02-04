class Config:
    vocab_size = 30000
    max_len = 128

    hidden_size = 256
    num_heads = 4
    num_layers = 2
    ffn_hidden = hidden_size * 4

    dropout = 0.1
    device = "cuda"