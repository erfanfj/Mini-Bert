import torch.nn as nn
import torch
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)
    

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_hidden):
        super().__init__()

        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
    
class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0)

        token_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(positions)

        embeddings = token_emb + pos_emb
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)
    
    
class MiniBERT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = BertEmbeddings(
            config.vocab_size,
            config.hidden_size,
            config.max_len
        )

        self.layers = nn.ModuleList([
            EncoderLayer(
                config.hidden_size,
                config.num_heads,
                config.ffn_hidden
            )
            for _ in range(config.num_layers)
        ])

        self.mlm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        logits = self.mlm_head(x)   
        return logits
