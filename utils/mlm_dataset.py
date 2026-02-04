import torch
import random
from torch.utils.data import Dataset
import os
from .tokenizer import tokenize

def load_txt_files(data_dir):
    texts = []

    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            path = os.path.join(data_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())

    return texts

class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob

        self.samples = []
        self._build_samples(texts)

    def _build_samples(self, texts):
        for text in texts:
            tokens = tokenize(text)

            for i in range(0, len(tokens), self.max_len - 2):
                chunk = tokens[i:i + self.max_len - 2]
                if len(chunk) < 5:
                    continue
                self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)
    
    def _mask_tokens(self, token_ids):
        labels = [-100] * len(token_ids)

        for i in range(1, len(token_ids) - 1):  # CLS و SEP دست نخورند
            if random.random() < self.mlm_prob:
                labels[i] = token_ids[i]
                prob = random.random()

                if prob < 0.8:
                    token_ids[i] = self.tokenizer.word2id["[MASK]"]
                elif prob < 0.9:
                    token_ids[i] = random.randint(
                        len(self.tokenizer.word2id),
                        len(self.tokenizer.word2id)
                    )
                # else: unchanged

        return token_ids, labels
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]

        ids = [self.tokenizer.word2id["[CLS]"]]
        ids += [self.tokenizer.word2id.get(t, self.tokenizer.word2id["[UNK]"]) for t in tokens]
        ids.append(self.tokenizer.word2id["[SEP]"])

        ids, labels = self._mask_tokens(ids)

        attention_mask = [1] * len(ids)

        if len(ids) < self.max_len:
            pad_len = self.max_len - len(ids)
            ids += [self.tokenizer.word2id["[PAD]"]] * pad_len
            labels += [-100] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }