import torch
import random
from torch.utils.data import Dataset
import os
from .tokenizer import tokenize

def read_in_chunks(file, chunk_size=10000):
    chunk = []
    for line in file:
        chunk.append(line)
        if len(chunk) == chunk_size:
            yield ''.join(chunk)
            chunk = []
    if chunk:
        yield ''.join(chunk)
class MLMDataset(Dataset):
    def __init__(
        self,
        file_paths,
        tokenizer,
        max_len=128,
        mlm_prob=0.15,
        chunk_size=10000,
        min_tokens=5
    ):
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.samples = []

        self._build_samples(
            file_paths=file_paths,
            chunk_size=chunk_size,
            min_tokens=min_tokens
        )

    def _build_samples(self, file_paths, chunk_size, min_tokens):
        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                for text_chunk in read_in_chunks(f, chunk_size):
                    tokens = tokenize(text_chunk)

                    for i in range(0, len(tokens), self.max_len - 2):
                        chunk = tokens[i:i + self.max_len - 2]

                        if len(chunk) < min_tokens:
                            continue

                        self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)
    
    def _mask_tokens(self, token_ids):
        labels = [-100] * len(token_ids)
        for i in range(1, len(token_ids) - 1):  
            if random.random() < self.mlm_prob:
                labels[i] = token_ids[i]
                prob = random.random()

                if prob < 0.8:
                    token_ids[i] = self.tokenizer.word2id["[MASK]"]

                elif prob < 0.9:
                    token_ids[i] = random.randint(
                        0, len(self.tokenizer.word2id) - 1
                    )


        return token_ids, labels
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]

        input_ids = [self.tokenizer.word2id["[CLS]"]]
        input_ids += [
            self.tokenizer.word2id.get(t, self.tokenizer.word2id["[UNK]"])
            for t in tokens
        ]
        input_ids.append(self.tokenizer.word2id["[SEP]"])

        input_ids, labels = self._mask_tokens(input_ids)

        attention_mask = [1] * len(input_ids)

        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids += [self.tokenizer.word2id["[PAD]"]] * pad_len
            labels += [-100] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }