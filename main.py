from torch.utils.data import DataLoader
from utils.mlm_dataset import MLMDataset ,load_txt_files
from utils.tokenizer import Tokenizer

texts = load_txt_files("data")

tokenizer = Tokenizer(vocab_size=30000)
tokenizer.build_vocab(texts)
print("-------------------")

dataset = MLMDataset(
    texts=texts,
    tokenizer=tokenizer,
    max_len=128
)
print("-------------------")


loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)
print("-------------------")


batch = next(iter(loader))

print(batch["input_ids"].shape)
print(batch["labels"].shape)
print(batch["attention_mask"].shape)