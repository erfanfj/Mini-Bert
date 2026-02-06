import re
from hazm import Normalizer, WordTokenizer
from collections import Counter

fa_normalizer = Normalizer()
fa_tokenizer = WordTokenizer()

SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4
}


class Tokenizer:
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size

        # 🔴 NEW: global counter for chunk-based vocab
        self.counter = Counter()

        self.word2id = dict(SPECIAL_TOKENS)
        self.id2word = {v: k for k, v in SPECIAL_TOKENS.items()}

    # 🔴 NEW: instead of build_vocab
    def add_texts(self, texts):
        for text in texts:
            tokens = tokenize(text)
            self.counter.update(tokens)

    # 🔴 NEW: finalize vocab after all data seen
    def finalize_vocab(self):
        most_common = self.counter.most_common(
            self.vocab_size - len(self.word2id)
        )

        for word, _ in most_common:
            idx = len(self.word2id)
            self.word2id[word] = idx
            self.id2word[idx] = word

    def encode(self, text, max_len=128):
        tokens = tokenize(text)

        ids = [self.word2id["[CLS]"]]
        ids += [self.word2id.get(t, self.word2id["[UNK]"]) for t in tokens]
        ids.append(self.word2id["[SEP]"])

        if len(ids) < max_len:
            ids += [self.word2id["[PAD]"]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        return ids
    

def tokenize(text: str):
    text = normalize_text(text)

    tokens = []
    for word in text.split():
        if re.search(r"[\u0600-\u06FF]", word):
            tokens.extend(fa_tokenizer.tokenize(word))
        else:
            tokens.append(word)

    return tokens


def normalize_text(text: str) -> str:
    text = text.strip()
    text = fa_normalizer.normalize(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\u0600-\u06FF0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ############
# #   test   #
# ############
# texts = [
#     "BERT برای پردازش زبان طبیعی استفاده می‌شود",
#     "Deep Learning در NLP خیلی مهم است"
# ]
# print(normalize_text(texts[0]))
# print(tokenize(texts[0]))

# tokenizer = Tokenizer(vocab_size=10000)
# tokenizer.build_vocab(texts)

# print(tokenizer.encode("Deep Learning در NLP خیلی مهم است"))