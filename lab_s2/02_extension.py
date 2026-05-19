# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab S2, Extension: Compute Attention by Hand
#
# **Goal.** Compute scaled dot-product attention on a tiny 3-token example,
# step by step, with numpy. Verify against PyTorch. The Transformer becomes
# concrete.
#
# **Time.** ~15 minutes.

# %% [markdown]
# ## 0. Setup

# %%
import sys
sys.path.insert(0, ".")

import math
import numpy as np
import torch

from src.manual_attention import manual_attention, softmax_rows

np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## 1. Pick a 3-token example with simple integer matrices
#
# We choose Q, K, V with d_k = d_v = 4 so the math is tractable on paper.
# T = 3 tokens, each represented by a 4-dim Q vector, a 4-dim K vector, and a
# 4-dim V vector.

# %%
Q = np.array(
    [
        [1.0, 0.0, 1.0, 0.0],   # token 1 query
        [0.0, 2.0, 0.0, 2.0],   # token 2 query
        [1.0, 1.0, 1.0, 1.0],   # token 3 query
    ]
)
K = np.array(
    [
        [1.0, 0.0, 1.0, 0.0],   # token 1 key
        [0.0, 1.0, 0.0, 1.0],   # token 2 key
        [1.0, 1.0, 0.0, 0.0],   # token 3 key
    ]
)
V = np.array(
    [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]
)

print("shapes:", Q.shape, K.shape, V.shape)

# %% [markdown]
# ## 2. Hand calculation: every intermediate matrix is printed

# %%
out_np, weights_np = manual_attention(Q, K, V, verbose=True)

# %% [markdown]
# ## 3. Verify with PyTorch's built-in scaled dot-product attention

# %%
Qt = torch.tensor(Q, dtype=torch.float32).unsqueeze(0)   # (1, T, d_k)
Kt = torch.tensor(K, dtype=torch.float32).unsqueeze(0)
Vt = torch.tensor(V, dtype=torch.float32).unsqueeze(0)

# torch.nn.functional.scaled_dot_product_attention applies softmax internally.
out_torch = torch.nn.functional.scaled_dot_product_attention(Qt, Kt, Vt)
print(f"torch output:\n{out_torch.numpy()[0]}")
print(f"\nnumpy output:\n{out_np}")
print(f"\nallclose: {np.allclose(out_torch.numpy()[0], out_np, atol=1e-5)}")

# %% [markdown]
# ## 4. TODO: do the same on paper
#
# Take Q, K, V from cell 1. Without running anything, on a blank sheet:
# 1. compute Q K^T (3x3 matrix),
# 2. divide by sqrt(d_k) = sqrt(4) = 2,
# 3. apply softmax row by row,
# 4. multiply by V.
#
# Then run the cell below and check that you got the same numbers.

# %%
print("Hand-vs-code comparison reference (run this only after you tried on paper):")
print(f"weights:\n{weights_np}")
print(f"output:\n{out_np}")

# %% [markdown]
# ## 5. Bonus: causal mask
#
# In a decoder, position t can only attend to positions <= t. Add the causal
# mask manually and recompute.

# %%
def manual_attention_causal(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)
    T = scores.shape[0]
    mask = np.triu(np.full((T, T), -np.inf), k=1)
    scores = scores + mask
    weights = softmax_rows(scores)
    return weights @ V, weights

out_causal, w_causal = manual_attention_causal(Q, K, V)
print(f"causal weights (upper triangle = 0):\n{w_causal}")
print(f"causal output:\n{out_causal}")

# %% [markdown]
# ## What this extension proves
#
# - The Transformer's "magic" reduces to four matrix operations.
# - The softmax forces every row of weights to sum to 1: each query allocates
#   probability mass across keys.
# - The causal mask is just adding -inf to the upper triangle before softmax.
#
# After this exercise, the multi-head version is conceptually trivial: same
# four ops, run h times in parallel on different projections, then concat.
