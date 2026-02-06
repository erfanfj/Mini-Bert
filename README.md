# 🧠 Mini-BERT (Persian–English)

## Masked Language Modeling from Scratch

This project implements a **Mini-BERT** model **from scratch** using **PyTorch** and pretrains it with **Masked Language Modeling (MLM)** on large-scale Persian text corpora (e.g., Persian WikiText).

The pipeline fully supports **mixed Persian–English text**, uses **Hazm** for Persian normalization, and is designed to be **memory-efficient** via **chunk-based streaming**.

---

## 🚀 Features

- ✅ Custom **Tokenizer** (Persian + English)
- ✅ Persian normalization using **Hazm**
- ✅ Mixed-script (FA/EN) token handling
- ✅ Chunk-based vocabulary construction (OOM-safe)
- ✅ Custom **Mini-BERT** (Transformer Encoder)
- ✅ Dynamic **MLM masking**
- ✅ Full training, evaluation, and inference
- ✅ Interview & research demo-ready

---

## 📦 Project Structure

```text
Mini-Bert/
│
├── data/
│   └── Persian-WikiText-*.txt
│
├── utils/
│   ├── tokenizer.py        # Persian–English tokenizer
│   ├── mlm_dataset.py      # MLM dataset + masking
│   ├── bert.py             # Mini-BERT architecture
│   └── config.py           # Model & training config
│
├── results/
│   ├── train_loss.png
│   ├── metrics.txt
│
├── models/
│   ├── mini_bert_epoch_1.pt
│   ├── mini_bert_epoch_2.pt
│   ├── mini_bert_epoch_3.pt
│
├── Train.ipynb                 # Training script
├── test_infer.py           # MLM inference (MASK prediction)
└── README.md
```

---

## ⚙️ Installation

### 1. Download Dataset

Download the Persian Wikipedia dataset from Kaggle:

**🔗 [Persian Wikipedia Dataset](https://www.kaggle.com/datasets/miladfa7/persian-wikipedia-dataset)**

After downloading:
1. Extract the dataset files
2. Place them in a folder named `data/` in the project root

```text
Mini-Bert/
├── data/
│   └── Persian-WikiText-*.txt  ← Place dataset files here

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install torch hazm nltk tqdm scikit-learn matplotlib
```

### ⚠️ Windows Note

During development, set `num_workers=0` in `DataLoader` to avoid multiprocessing pickling issues.

---

## 🧠 Tokenizer Design

The tokenizer is custom-built and optimized for Persian NLP:

- **Persian text** normalized using **Hazm**
- **English tokens** lowercased
- **Unicode-based** mixed-script detection
- **Word-level** tokenization

### Special Tokens

- `[PAD]` — Padding token
- `[UNK]` — Unknown token
- `[CLS]` — Classification token
- `[SEP]` — Separator token
- `[MASK]` — Mask token

### 🔹 Vocabulary Construction (Chunk-Based)

To safely handle large corpora:

1. Text files are read **incrementally**
2. Token frequencies accumulated via a **global counter**
3. Vocabulary finalized after scanning the **full dataset**

**No full corpus is ever loaded into memory.**

---

## 🏋️ Training (Masked Language Modeling)

```bash
python main.py
```

### Training Details

- **Objective:** Masked Language Modeling (MLM)
- **Optimizer:** AdamW
- **Loss:** `CrossEntropyLoss(ignore_index=-100)`

### Masking Strategy

Masking follows **BERT conventions**:

- **80%** → `[MASK]`
- **10%** → random token
- **10%** → unchanged

### Sample Training Log

```text
Epoch 1 | Avg Loss: 4.54
Epoch 2 | Avg Loss: 3.64
Epoch 3 | Avg Loss: 3.39
```

Model checkpoints are saved in `models/`.

---

## 📊 Evaluation Metrics

Evaluation is performed **only on masked tokens**:

- **Loss**
- **Accuracy**
- **Macro F1-score**

### Outputs

- Metrics: `results/metrics.txt`
- Training loss plot: `results/train_loss.png`

---

## 🔍 Inference (MASK Prediction)

Inference is **token-based**, not string-based.

```bash
python test_infer.py
```

### Example

**Input sentence:**

```text
یادگیری ماشین در پردازش زبان طبیعی مهم است
```

**Masked & predicted:**

```text
Original token : طبیعی
Predicted token: زبان
```

Masking is applied at **token-id level**, ensuring consistency with normalization and vocabulary.

---

## 🧪 Reproducibility Notes

- **Attention masks** are broadcast to: `[batch_size, num_heads, seq_len, seq_len]`
- **MLM labels** use `-100` for ignored positions
- **Inference** never relies on raw string replacement
- **Tokenization & masking** are deterministic

---

## 🎯 Interview-Ready Talking Points

1. The tokenizer supports **mixed Persian–English text** using **Hazm**.
2. Vocabulary is built **incrementally** via **chunk-based streaming**.
3. MLM is implemented following **original BERT conventions**.
4. Evaluation is performed **only on masked tokens** using **macro-F1**.
5. **Masking and inference** are applied at **token-id level**.

---

## 📌 Author

**Erfan**  
Mini-BERT from scratch for Persian NLP 🚀

---