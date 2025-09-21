# Neural Translator: English ↔ German Seq2Seq Model

A GRU-based sequence-to-sequence neural machine translation model** built with PyTorch and TorchText for translating between English and German using the Multi30k dataset.

---

## 🛠 Tech Stack
- Frameworks & Libraries: PyTorch, TorchText, SpaCy, NumPy
- Languages: Python 3.x
- Tokenizer: SpaCy (de_core_news_sm, en_core_web_sm)
- GPU Support: CUDA (optional)

---

## ⚡ Features
- GRU-based encoder–decoder architecture for sequence-to-sequence translation
- Custom data preprocessing pipeline:
  - Tokenization
  - Vocabulary building with special tokens <BOS>, <EOS>, <PAD>, <UNK>
  - Batch padding for variable-length sentences
- Teacher forcing during training to improve convergence
- Supports batch training using PyTorch DataLoader and custom collate_fn

## 🔹 Dataset
This project uses the Multi30k dataset (English ↔ German).

- Download URLs updated in code:
- multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
- multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

- Dataset splits:
  - Train: 29,000 sentences
  - Validation: 1,014 sentences
  - Test: 1,000 sentences

---

## 🔹 How It Works
1. Preprocessing:
   - Tokenize German and English sentences using SpaCy
   - Build vocabularies with TorchText
   - Add special tokens: <BOS>, <EOS>, <PAD>, <UNK>
   - Pad sequences in each batch using collate_fn

2. Encoder:
   - Embeds source tokens → GRU → outputs hidden states
   - Returns final hidden state to initialize decoder

3. Decoder:
   - Embeds target tokens → ReLU → GRU initialized with encoder hidden
   - Linear + LogSoftmax → predicts next word probabilities

4. Training:
   - Teacher forcing: input t_{0...T-1} → predict t_{1...T}
   - Loss: nn.NLLLoss(ignore_index=PAD_IDX)
   - Optimizers update both encoder and decoder

5. Inference:
   - Start with <BOS> token
   - Iteratively predict next token until <EOS> is generated
