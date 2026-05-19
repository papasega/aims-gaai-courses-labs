# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab S2, Baseline: Visualizing Attention in a Pre-trained Transformer
#
# **Goal.** Load a small multilingual BERT, run it on three sentences (English,
# French, Wolof translit), generate attention heatmaps, and identify what
# different heads are doing.
#
# **Time.** ~30 minutes. CPU is enough.
#
# **Output you must produce.**
# - 3 heatmaps from different heads/layers on the canonical "it" sentence,
# - 1 heatmap on a French sentence,
# - 1 heatmap on a Wolof translit sentence,
# - one-line interpretation per heatmap.

# %% [markdown]
# ## 0. Setup

# %%
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.attention_viz import load_model, get_attentions, plot_head, head_importance

# %% [markdown]
# ## 1. Load the model
#
# `distilbert-base-multilingual-cased`: 6 layers, 12 heads, ~530 MB.
# First call downloads from the HF Hub; subsequent calls hit the local cache.

# %%
tokenizer, model = load_model("distilbert-base-multilingual-cased")
print(f"layers: {model.config.num_hidden_layers}, heads: {model.config.num_attention_heads}")

# %% [markdown]
# ## 2. The canonical "it" sentence
#
# This is the sentence Jay Alammar's "Illustrated Transformer" uses to show
# coreference. Watch how different heads handle "it".

# %%
sentence_en = "The animal didn't cross the street because it was too tired."
tokens_en, attn_en = get_attentions(sentence_en, tokenizer, model)
print(tokens_en)
print(f"layers={len(attn_en)}, head shape per layer = {attn_en[0].shape}")

# %% [markdown]
# ### 2.1 Plot a few heads to find an interesting one

# %%
plot_head(attn_en, tokens_en, layer=0, head=0, title="EN | layer 0, head 0")
plt.show()

# %%
plot_head(attn_en, tokens_en, layer=2, head=5, title="EN | layer 2, head 5")
plt.show()

# %%
plot_head(attn_en, tokens_en, layer=5, head=11, title="EN | layer 5, head 11")
plt.show()

# %% [markdown]
# ### 2.2 TODO: find one head that resolves "it"
#
# Loop over (layer, head) pairs, plot the row corresponding to the "it" token,
# and find a head where the strongest weight is on "animal" or "street".

# %%
# TODO: adapt this loop to find a head where it -> animal is highlighted.
# Find the index of "it" in tokens_en first.
it_idx = tokens_en.index("it")
print(f"'it' is at position {it_idx}")

# Try a few heads:
for L in [3, 4, 5]:
    for H in [0, 4, 8]:
        w = attn_en[L][0, H, it_idx].cpu().numpy()
        top = np.argsort(-w)[:3]
        top_tokens = [tokens_en[i] for i in top]
        print(f"L={L} H={H} | 'it' attends most to: {top_tokens} (weights {w[top].round(2)})")

# %% [markdown]
# ### 2.3 Optional extension: automatically rank heads for "it" -> "animal"
#
# The previous cell asks you to inspect a few heads manually. This extension
# scans every layer/head and ranks them by the attention weight from the query
# token `"it"` to the key token `"animal"`.

# %%
target_token = "animal"
target_idx = tokens_en.index(target_token)

candidates = []
for L, layer_attn in enumerate(attn_en):
    for H in range(layer_attn.shape[1]):
        w = layer_attn[0, H, it_idx].detach().cpu().numpy()
        top = np.argsort(-w)[:5]
        candidates.append(
            (
                float(w[target_idx]),
                L,
                H,
                [tokens_en[i] for i in top],
                w[top].round(2),
            )
        )

candidates = sorted(candidates, reverse=True)
for score, L, H, top_tokens, top_weights in candidates[:5]:
    print(
        f"L={L} H={H} | weight(it -> {target_token})={score:.3f} | "
        f"top={top_tokens} | weights={top_weights}"
    )

best_score, best_L, best_H, _, _ = candidates[0]
plot_head(
    attn_en,
    tokens_en,
    layer=best_L,
    head=best_H,
    title=f"Best candidate for it -> {target_token} | L{best_L} H{best_H}",
)
plt.show()

# %% [markdown]
# ## 3. The same model on a French sentence
#
# DistilBERT-multilingual covers ~100 languages including French. We expect
# similar behavior, with sub-word tokenization splitting some words.

# %%
sentence_fr = "L'animal n'a pas traverse la rue parce qu'il etait trop fatigue."
tokens_fr, attn_fr = get_attentions(sentence_fr, tokenizer, model)
print(tokens_fr)

# %%
plot_head(attn_fr, tokens_fr, layer=4, head=8, title="FR | layer 4, head 8")
plt.show()

# %% [markdown]
# ## 4. A Wolof translit sentence
#
# Wolof was likely seen during pre-training only marginally. We do not expect
# fluency, but the model still produces an attention pattern. Useful baseline
# for the bias discussion in Module 4.

# %%
sentence_wo = "Naka nga def Aminata bu jot fii ?"
tokens_wo, attn_wo = get_attentions(sentence_wo, tokenizer, model)
print(tokens_wo)

# %%
plot_head(attn_wo, tokens_wo, layer=4, head=8, title="Wolof | layer 4, head 8")
plt.show()

# %% [markdown]
# ## 5. Head importance map (which heads are sharp?)

# %%
ent = head_importance(attn_en)
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(ent, cmap="viridis", aspect="auto")
ax.set_xlabel("head index")
ax.set_ylabel("layer")
ax.set_title("Mean attention entropy per (layer, head). Lower = more peaked.")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. TODO: write your interpretations
#
# For each heatmap above, write 1 line answering:
# - what does the bright diagonal pattern mean?
# - which heads attend mostly to the previous token?
# - which heads attend across long distances?
# - is the Wolof sentence treated similarly to French?

# %% [markdown]
# > YOUR NOTES HERE
#
# - heatmap 1 (L0 H0):
# - heatmap 2 (L2 H5):
# - heatmap 3 (L5 H11):
# - French heatmap:
# - Wolof heatmap:
# - the head you found that resolves "it":

# %% [markdown]
# ## What this lab proves
#
# Different heads specialize without supervision: previous-token, coreference,
# punctuation, syntactic dependency. This is the empirical reason multi-head
# attention matters. The same model behaves consistently on French; on Wolof,
# patterns are noisier, hinting at the data-coverage problem we discuss in S12.
