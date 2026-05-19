"""Load a small multilingual Transformer, extract attention weights, plot heatmaps.

Default model: distilbert-base-multilingual-cased (~530 MB, 6 layers, 12 heads).
Pick something else by passing model_name to load_model().
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from typing import Optional


def load_model(model_name: str = "distilbert-base-multilingual-cased", device: Optional[str] = None):
    """Return (tokenizer, model) configured to return attention weights."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def get_attentions(text: str, tokenizer, model):
    """Run text through the model and return (tokens, attentions).

    tokens:     list of strings (sub-words) including special tokens.
    attentions: tuple of length n_layers, each shape (1, n_heads, T, T).
    """
    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt")
    enc = {key: value.to(device) for key, value in enc.items()}
    out = model(**enc)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return tokens, out.attentions


def plot_head(
    attentions,
    tokens,
    layer: int,
    head: int,
    title: str = None,
    cmap: str = "viridis",
    figsize=(7, 6),
):
    """Plot a heatmap of a specific (layer, head) attention pattern."""
    weights = attentions[layer][0, head].cpu().numpy()  # (T, T)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(weights, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=60, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_xlabel("attended-to (key) position")
    ax.set_ylabel("attending (query) position")
    if title is None:
        title = f"layer {layer}, head {head}"
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, ax


def head_importance(attentions):
    """For each (layer, head), return the mean entropy of attention.

    Lower entropy = sharper / more specialized head.
    Useful to spot "interesting" heads quickly.
    """
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    out = np.zeros((n_layers, n_heads))
    for L in range(n_layers):
        w = attentions[L][0]  # (n_heads, T, T)
        eps = 1e-12
        ent = -(w * (w + eps).log()).sum(dim=-1).mean(dim=-1)  # (n_heads,)
        out[L] = ent.cpu().numpy()
    return out
