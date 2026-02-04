from torch.utils.data import DataLoader
from utils.mlm_dataset import MLMDataset 
from utils.tokenizer import Tokenizer
import glob

files = glob.glob("data/Persian-WikiText-*.txt")

tokenizer = Tokenizer(vocab_size=30000)

with open(files[0], encoding="utf-8") as f:
    tokenizer.build_vocab([f.read()[:500_000]])

dataset = MLMDataset(
    file_paths=files,
    tokenizer=tokenizer,
    max_len=128
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)


batch = next(iter(loader))
print(batch["input_ids"].shape)
print(batch["labels"].shape)
print(batch["attention_mask"].shape)