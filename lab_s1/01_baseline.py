# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab S1, Baseline: Char-LSTM and its Limits
#
# **Goal.** Train a small character-level LSTM on a Wolof corpus, generate text from
# different prompts, and observe where the model loses long-range coherence.
#
# **Time.** ~20 minutes (most of it training).
#
# **Output you must produce.**
# - 3 generated samples (short / medium / long prompts)
# - 1 short paragraph annotating where the LSTM starts drifting

# %% [markdown]
# ## 0. Setup — run this cell first

# %%
import os, sys, lzma, urllib.request
from pathlib import Path

# ── 0.1  Fix working directory ──────────────────────────────────────────────
# Notebooks launched from any folder (VS Code, JupyterLab, Colab) may have a
# different cwd. We always want to be at the lab_s1/ root.
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
        print(f"CC100 failed ({e}). Using built-in Wolof proverbs sample…")
        WOLOF_SAMPLE = """Wolof ak Pulaar ak Seereer ak Joola ak Mandinka mooy làkk yi ñu gëna wax ci Senegaal.
Ndax sa yàgg-yàgg dafa am solo ci sa aduna.
Ku amul jom, amul dara.
Liggeey bi mooy xaalis mi.
Jàngalekat bi dafa nekk ci ekool bi.
Nit ku baax, mooy ku wóor ci lu mu wax.
Góor gi dem na ca dëkk ba.
Xale bi di na jàng ci daara ji.
Ndey ji dafa bëgg doom ji.
Fas wi dafa gaaw lool.
Maa ngi ñów fii ci ngoon si.
Suba si, dinaa dem ci liggeeyu bi.
Yow, ana sa waa-kër yi?
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

from src.char_lstm import CharLSTM
from src.utils import build_vocab, load_corpus, make_loaders, train_one_epoch, evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device : {device}")
torch.manual_seed(42)

# %% [markdown]
# ## 1. Load and inspect the corpus
#
# By default we use a **Wolof** corpus (`data/wolof.txt`).
# If you prefer English, switch to `data/tinyshakespeare.txt` (see README).

# %%
# --- CHOOSE YOUR CORPUS ---
# Option A (default): Wolof
CORPUS_PATH = "data/wolof.txt"

# Option B: English (uncomment the line below)
# CORPUS_PATH = "data/tinyshakespeare.txt"

text = load_corpus(CORPUS_PATH)
print(f"corpus length: {len(text):,} characters")
print(f"first 300 chars:\n{text[:300]}")

# %%
chars, char_to_idx, idx_to_char = build_vocab(text)
print(f"vocab size: {len(chars)}")
print(f"distinct chars: {''.join(chars)}")

# %% [markdown]
# ## 2. Build train / val loaders
#
# Each sample is a window of `seq_length` characters, and the target is the same
# window shifted by 1. We use `stride=4` (skip 4 chars between windows) and
# cap at `max_chars` for fast training on CPU.

# %%
seq_length = 128
batch_size = 64
max_chars = 200_000  # limit corpus size for fast CPU training (~3 min)

train_loader, val_loader = make_loaders(
    text, char_to_idx, seq_length, batch_size,
    stride=4, max_chars=max_chars,
)
print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

# %% [markdown]
# ## 3. Instantiate the model

# %%
model = CharLSTM(
    vocab_size=len(chars),
    embedding_dim=64,
    hidden_dim=256,
    num_layers=2,
    dropout=0.2,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"trainable parameters: {n_params:,}")

# %% [markdown]
# ## 4. Train
#
# On CPU with `max_chars=200_000` and `stride=4`: ~3 minutes for 5 epochs.
# On Colab T4: ~1 minute. Reduce `epochs` to 2 if you only want to see convergence.

# %%
epochs = 5
lr = 3e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
for ep in range(1, epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"epoch {ep}/{epochs} | train {train_loss:.3f} | val {val_loss:.3f}")

# %% [markdown]
# ## 5. Plot the loss curves

# %%
plt.figure(figsize=(6, 3.5))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("epoch")
plt.ylabel("cross-entropy loss")
plt.legend()
plt.title("Char-LSTM training")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Generate from three prompts (the heart of this lab)
#
# **What to look for:**
# 1. Local fluency: does the output look like real text at the word level?
# 2. Long-range coherence: does it stay on a topic, finish a sentence cleanly?
# 3. Repetition: do the same n-grams keep coming back?

# %% [markdown]
# ### 6.1 Short prompt

# %%
sample_short = model.generate(
    seed="Senegaal",
    char_to_idx=char_to_idx,
    idx_to_char=idx_to_char,
    max_new_tokens=400,
    temperature=0.8,
    device=device,
)
print(sample_short)

# %% [markdown]
# ### 6.2 Medium prompt (a partial sentence)

# %%
sample_medium = model.generate(
    seed="Ku amul jom, amul",
    char_to_idx=char_to_idx,
    idx_to_char=idx_to_char,
    max_new_tokens=400,
    temperature=0.8,
    device=device,
)
print(sample_medium)

# %% [markdown]
# ### 6.3 Long prompt (a multi-line setup)

# %%
seed_long = """Wolof ak Pulaar ak Seereer mooy làkk yi
ñu gëna wax ci Senegaal. Ndax sa yàgg-yàgg
"""
sample_long = model.generate(
    seed=seed_long,
    char_to_idx=char_to_idx,
    idx_to_char=idx_to_char,
    max_new_tokens=600,
    temperature=0.8,
    device=device,
)
print(sample_long)

# %% [markdown]
# ## 7. TODO: annotate what you see
#
# In the cell below, write 5 lines answering:
#
# 1. At which character offset (~) does the model start drifting?
# 2. What kind of repetition do you observe?
# 3. Does the model "forget" the initial context after a while?
# 4. Does the long prompt help or hurt long-range coherence?
# 5. What would you change in the architecture to fix this?

# %% [markdown]
# > YOUR NOTES HERE
#
# - drift offset:
# - repetition pattern:
# - context memory:
# - long prompt vs short prompt:
# - architectural fix idea:

# %% [markdown]
# ## 8. Quick sanity check (optional)
#
# Generate with two temperatures (0.5 conservative, 1.2 chaotic) and compare.

# %%
print("=== T = 0.5 ===")
print(model.generate("Liggeey", char_to_idx, idx_to_char, max_new_tokens=200, temperature=0.5, device=device))
print()
print("=== T = 1.2 ===")
print(model.generate("Liggeey", char_to_idx, idx_to_char, max_new_tokens=200, temperature=1.2, device=device))

# %% [markdown]
# ## What this lab proves
#
# The LSTM produces locally plausible text but loses the thread within
# 100-200 characters. Repetition appears, then drift. This is the empirical
# motivation for **attention** (Session 2): instead of compressing the whole
# past into a fixed hidden state, attention lets every position look back at
# every previous position directly.
