class Config:
    vocab_size = 30000
    max_len = 128
    batch_size = 16

    hidden_size = 256
    num_layers = 4
    num_heads = 4
    ffn_hidden = 1024
    dropout = 0.1

    lr = 5e-4
    epochs = 3