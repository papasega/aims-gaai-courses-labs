# Lab S1: From Char-LSTM to Attention

Module 0, Session 1. This lab trains a small character-level LSTM on a Wolof
corpus, generates text from several prompts, and asks you to observe where the
model loses long-range coherence. The optional extension replaces the recurrent
model with a minimal causal self-attention model.

## What You Will Produce

1. Three generated samples from `01_baseline.ipynb`.
2. A short annotation of drift, repetition, and context loss.
3. Optional: a side-by-side comparison between the LSTM and `TinyAttnLM` from
   `02_extension.ipynb`.

## Layout

```text
lab_s1/
├── README.md
├── requirements.txt
├── 01_baseline.ipynb       # required notebook
├── 02_extension.ipynb      # optional attention extension
├── src/
│   ├── char_lstm.py
│   └── utils.py
├── data/
│   ├── download_wolof.py
│   └── download_corpus.py
└── solutions/
    └── SOLUTIONS.md        # instructor reference only
```

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
cd lab_s1
python -m pip install -r requirements.txt
```

## Run

```bash
jupyter notebook 01_baseline.ipynb
```

Run the cells from top to bottom. The default corpus is Wolof. The first setup
cell creates `data/wolof.txt` automatically if it is missing:

```python
CORPUS_PATH = "data/wolof.txt"
```

To use English instead, download Tiny Shakespeare first:

```bash
python data/download_corpus.py
```

Then change the corpus path in the notebook:

```python
CORPUS_PATH = "data/tinyshakespeare.txt"
```

## Required Task

In Section 7 of `01_baseline.ipynb`, fill the TODO notes:

- drift offset;
- repetition pattern;
- context memory;
- long prompt vs short prompt;
- architectural fix idea.

## Optional Extension

Open `02_extension.ipynb` to train a tiny causal self-attention model on the
same corpus. The extension is not a full Transformer: it has one attention
layer, a causal mask, learned positional embeddings, and no multi-head attention,
feed-forward block, or LayerNorm.

## Submission

```bash
git add .
git commit -m "lab_s1: completed baseline + observations"
git push
```

Submit the completed notebook before the deadline.
