# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab S1, Extension: Minimal Dot-Product Attention
#
# **Goal.** Implement a tiny attention-augmented sequence model from scratch and
# compare it qualitatively with the LSTM trained in `01_baseline.py`.
#
# This is the bridge between Session 1 (RNN/LSTM) and Session 2 (Transformer).
# We do NOT yet build a Transformer. We build the smallest thing that uses
# attention so the contrast with the LSTM is visible.

# %% [markdown]
# ## 0. Setup — run this cell first

# %%
import os, sys, math, lzma, urllib.request
from pathlib import Path

# ── 0.1  Fix working directory ──────────────────────────────────────────────
def _find_lab_root(marker="src") -> Path:
    for p in [Path.cwd(), Path(__file__).parent if "__file__" in dir() else Path(".")]:
        if (p / marker).is_dir():
            return p
        for parent in p.parents:
            if (parent / marker).is_dir():
                return parent
    return Path.cwd()

LAB_ROOT = _find_lab_root()
os.chdir(LAB_ROOT)
sys.path.insert(0, str(LAB_ROOT))
print(f"working directory: {Path.cwd()}")

# ── 0.2  Auto-download Wolof corpus ─────────────────────────────────────────
CORPUS_PATH = Path("data/wolof.txt")
CORPUS_PATH.parent.mkdir(exist_ok=True)

if not CORPUS_PATH.exists():
    print("Corpus not found — downloading Wolof CC100 (~500 KB)…")
    try:
        URL = "https://data.statmt.org/cc-100/wo.txt.xz"
        TMP = CORPUS_PATH.parent / "wolof_cc100.txt.xz"
        urllib.request.urlretrieve(URL, TMP)
        with lzma.open(TMP, "rt", encoding="utf-8") as f:
            text_dl = f.read(500_000)
        TMP.unlink()
        CORPUS_PATH.write_text(text_dl, encoding="utf-8")
        print(f"✓ Saved {len(text_dl):,} chars → {CORPUS_PATH}")
    except Exception as e:
        print(f"CC100 failed ({e}). Using built-in Wolof sample…")
        WOLOF_SAMPLE = """Wolof ak Pulaar ak Seereer ak Joola ak Mandinka mooy làkk yi ñu gëna wax ci Senegaal.
Ndax sa yàgg-yàgg dafa am solo ci sa aduna.
Ku amul jom, amul dara.
Liggeey bi mooy xaalis mi.
Jàngalekat bi dafa nekk ci ekool bi.
Nit ku baax, mooy ku wóor ci lu mu wax.
Senegaal dafa rafet lool.
Suñu réew mi am na jamm.
"""
        CORPUS_PATH.write_text(WOLOF_SAMPLE * 400, encoding="utf-8")
        print(f"✓ Built-in sample saved → {CORPUS_PATH}")
else:
    print(f"✓ Corpus already present: {CORPUS_PATH} ({CORPUS_PATH.stat().st_size//1024} KB)")

# ── 0.3  Core imports ────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.utils import build_vocab, load_corpus, make_loaders, train_one_epoch, evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# %% [markdown]
# ## 1. Implement scaled dot-product attention from scratch
#
# Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

# %%
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (B, T, d_k)
    mask:    (T, T) with 0 where allowed, -inf where blocked.
    returns: (B, T, d_k), (B, T, T) attention weights.
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (B, T, T)
    if mask is not None:
        scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    out = weights @ V                                    # (B, T, d_k)
    return out, weights


# Sanity check
B, T, d = 2, 5, 8
Q = torch.randn(B, T, d)
K = torch.randn(B, T, d)
V = torch.randn(B, T, d)
out, w = scaled_dot_product_attention(Q, K, V)
print(f"out: {out.shape}  weights: {w.shape}  rows sum to 1? {torch.allclose(w.sum(-1), torch.ones(B, T))}")

# %% [markdown]
# ## 2. A tiny attention-only character model
#
# Architecture:
#   - char embeddings + learned positional embeddings,
#   - one self-attention layer (causal mask),
#   - a final linear head over the vocabulary.
#
# No multi-head, no LayerNorm, no FFN. Just attention.
# This is enough to see the qualitative shift vs the LSTM.

# %%
class TinyAttnLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(positions)
        Q, K, V = self.q(h), self.k(h), self.v(h)
        # Causal mask: position t can only see positions <= t
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        logits = self.head(attn_out)
        return logits, None  # (None) keeps the same signature as CharLSTM

    @torch.no_grad()
    def generate(self, seed, char_to_idx, idx_to_char, max_new_tokens=200, temperature=1.0, device="cpu"):
        self.eval()
        idx = torch.tensor([[char_to_idx[c] for c in seed if c in char_to_idx]], device=device)
        out = list(seed)
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -self.max_len :]
            logits, _ = self(idx_cropped)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
            out.append(idx_to_char[int(next_idx)])
        return "".join(out)


# %% [markdown]
# ## 3. Train the tiny attention model
#
# Same data, same training loop. Only the model differs.

# %%
# Use the same corpus as 01_baseline
CORPUS_PATH = "data/wolof.txt"   # or "data/tinyshakespeare.txt"
text = load_corpus(CORPUS_PATH)
chars, char_to_idx, idx_to_char = build_vocab(text)
seq_length = 128
batch_size = 64
max_chars = 200_000
train_loader, val_loader = make_loaders(
    text, char_to_idx, seq_length, batch_size,
    stride=4, max_chars=max_chars,
)

model = TinyAttnLM(vocab_size=len(chars), d_model=128, max_len=seq_length).to(device)
print(f"trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
epochs = 5
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss()

losses = []
for ep in range(1, epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    losses.append((train_loss, val_loss))
    print(f"epoch {ep}/{epochs} | train {train_loss:.3f} | val {val_loss:.3f}")

# %% [markdown]
# ## 4. Generate and compare with the LSTM samples
#
# Use the same prompts as in `01_baseline.py`.

# %%
print("=== short ===")
print(model.generate("Senegaal", char_to_idx, idx_to_char, max_new_tokens=400, temperature=0.8, device=device))

# %%
print("=== medium ===")
print(model.generate("Ku amul jom, amul", char_to_idx, idx_to_char, max_new_tokens=400, temperature=0.8, device=device))

# %%
seed_long = """Wolof ak Pulaar ak Seereer mooy làkk yi
ñu gëna wax ci Senegaal.
"""
print("=== long ===")
print(model.generate(seed_long, char_to_idx, idx_to_char, max_new_tokens=600, temperature=0.8, device=device))

# %% [markdown]
# ## 5. TODO: side-by-side qualitative comparison
#
# Fill the table below. Read 100, 300, 600 chars of each generation.
#
# | dimension                  | LSTM | TinyAttn |
# |----------------------------|------|----------|
# | local fluency              | ?    | ?        |
# | speaker name memory        | ?    | ?        |
# | repetition after 200 chars | ?    | ?        |
# | sentence completion        | ?    | ?        |
#
# Then write 3 lines explaining what attention buys you, in your own words.

# %% [markdown]
# ## What this extension proves
#
# A single attention layer (no LSTM cell, no recurrence) trained on the same
# corpus already shows different failure modes: less local fluency at the
# character level, but better recovery of the speaker context across long spans.
# That is the exact intuition Session 2 generalizes into the Transformer.
