import torch
import random
import json

from utils.tokenizer import Tokenizer
from utils.bert import MiniBERT
from utils.config import Config
from utils.tokenizer import tokenize



# ---------- Load vocab ----------
def load_vocab(tokenizer, vocab_path):
    with open(vocab_path, encoding="utf-8") as f:
        tokenizer.word2id = json.load(f)

    # کلیدهای json استرینگ هستند → باید int شوند
    tokenizer.id2word = {int(v): k for k, v in tokenizer.word2id.items()}


# ---------- MLM Prediction ----------
def predict_mask_by_word(text, target_word, tokenizer, model, device):
    model.eval()

    # Encode sentence
    ids = tokenizer.encode(text)
    input_ids = torch.tensor([ids]).to(device)
    attention_mask = (input_ids != tokenizer.word2id["[PAD]"]).long()

    # Tokenize target word (خیلی مهم)
    target_tokens = tokenize(target_word)
    if len(target_tokens) != 1:
        raise ValueError("Target word must map to exactly one token.")

    target_token = target_tokens[0]
    target_id = tokenizer.word2id.get(target_token, tokenizer.word2id["[UNK]"])

    # پیدا کردن موقعیت توکن در جمله
    try:
        mask_pos = ids.index(target_id)
    except ValueError:
        raise ValueError(f"Token '{target_token}' not found in sentence.")

    original_id = input_ids[0, mask_pos].item()

    # Mask
    input_ids[0, mask_pos] = tokenizer.word2id["[MASK]"]

    # Forward
    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    pred_id = torch.argmax(logits[0, mask_pos]).item()

    return {
        "masked_word": target_word,
        "original_token": tokenizer.id2word.get(original_id, "[UNK]"),
        "predicted_token": tokenizer.id2word.get(pred_id, "[UNK]")
    }


# ---------- Main ----------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer + vocab
    tokenizer = Tokenizer(vocab_size=Config.vocab_size)
    load_vocab(tokenizer, "artifacts/vocab.json")

    # Load model
    model = MiniBERT(Config).to(device)
    state_dict = torch.load(
        "models/mini_bert_epoch_3.pt",
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)

    print("✅ Model and vocab loaded successfully.\n")

    # Interactive input
    while True:
        text = input("Enter a sentence (or 'exit'): ")
        if text.lower() == "exit":
            break

        word = input("Which word should be masked? ")

        try:
            result = predict_mask_by_word(text, word, tokenizer, model, device)

            print("\n--- MLM Prediction ---")
            print("Masked word     :", result["masked_word"])
            print("Original token  :", result["original_token"])
            print("Predicted token :", result["predicted_token"])
            print("----------------------\n")

        except Exception as e:
            print("❌ Error:", e)
