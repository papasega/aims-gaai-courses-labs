# Lab S1 — Correction: Char-LSTM and its Limits

This document provides the reference solutions for Lab S1. Read it carefully
after completing the lab to consolidate your understanding. Each section
explains **what** the expected answer is, **why** it matters, and highlights
**good coding practices** you should adopt.

---

## Part 1: Baseline (`01_baseline.py`)

### Understanding the code you ran

Before looking at the TODO answers, let us make sure you understand the key
code patterns used in this lab.

#### The CharLSTM model

```python
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=256,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
```

**Good coding practices to notice:**

- **Type hints and docstrings**: The class uses a docstring that describes every
  argument. Always document your models, even small ones.
- **Conditional dropout**: `dropout=dropout if num_layers > 1 else 0.0` —
  PyTorch ignores dropout when there is only 1 layer, but being explicit avoids
  a warning.
- **`batch_first=True`**: This means tensors are shaped `(Batch, Time, Features)`
  instead of `(Time, Batch, Features)`. This is more intuitive and matches how
  we think about data.

#### The training loop

```python
for ep in range(1, epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"epoch {ep}/{epochs} | train {train_loss:.3f} | val {val_loss:.3f}")
```

**Good coding practices to notice:**

- **Separate train and evaluate functions**: This makes the code reusable and
  testable. Never mix training logic and evaluation logic in one function.
- **Always track both train and val loss**: If train loss keeps going down but
  val loss goes up, you are overfitting. This is the most basic diagnostic.
- **F-string formatting**: `f"{val_loss:.3f}"` gives you 3 decimal places. Use
  this instead of `round()` for display — it is cleaner and does not affect the
  actual value.

#### The generation method

```python
@torch.no_grad()
def generate(self, seed, char_to_idx, idx_to_char,
             max_new_tokens=200, temperature=1.0, device="cpu"):
    self.eval()
    idx = torch.tensor([[char_to_idx[c] for c in seed if c in char_to_idx]],
                       dtype=torch.long, device=device)
    hidden = None
    out = list(seed)
    for _ in range(max_new_tokens):
        logits, hidden = self(idx, hidden)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        out.append(idx_to_char[int(next_idx)])
        idx = next_idx
    return "".join(out)
```

**Key concepts:**

- **`@torch.no_grad()`**: Disables gradient tracking during generation. This
  saves memory and speeds up inference. Always use it when you are not training.
- **`self.eval()`**: Puts the model in evaluation mode (disables dropout). If
  you forget this, your generated text will be noisier than it should be.
- **Temperature**: Dividing logits by temperature controls randomness.
  `T < 1.0` makes the output more deterministic (conservative),
  `T > 1.0` makes it more random (creative / chaotic).
  The `max(temperature, 1e-6)` guard prevents division by zero.
- **Autoregressive loop**: At each step, we feed only the last predicted
  character back into the model. The LSTM carries all past context in its
  `hidden` state — this is exactly the bottleneck we study in this lab.

---

### Section 7: Annotate what you see

After 5 epochs on the Wolof corpus (200K chars, stride=4), a typical generation
looks like:

```
Senegaal dafa rafet lool. Ndax sa yàgg-yàgg dafa am
solo ci sa aduna. Ku amul jom, amul dara. Liggeey
bi mooy xaalis mi. Liggeey bi mooy xaalis mi.
Liggeey bi mooy xaalis mi...
```

Here are the expected answers for each question:

**1. Drift offset (~80–150 characters)**

The model usually stays coherent for the first 80–150 characters. After that, it
starts producing text that no longer follows from the initial prompt. This
happens because the LSTM compresses everything into a fixed-size hidden state
vector (256 dimensions in our model). As more characters pass, the initial
prompt information gets diluted.

**2. Repetition pattern**

The LSTM falls into "attractor states": once it produces a high-frequency phrase
like "Liggeey bi mooy xaalis mi", the hidden state reinforces this phrase, and
the model loops on it. This is a fundamental limitation — the hidden state is a
fixed bottleneck that favors common patterns.

**3. Context memory**

After approximately 200 characters, the initial prompt is effectively invisible
to the model. The hidden state has been overwritten by more recent characters.
The generated text could follow any prompt — it is no longer conditioned on the
original input.

**4. Long prompt vs short prompt**

A longer prompt gives the LSTM more initial context, which usually helps the
first ~100 characters. But the benefit disappears quickly because the hidden
state is a fixed-size bottleneck — it can only store so much information,
regardless of how much you feed in.

**5. Architectural fix idea**

Any mechanism that lets the model **directly attend to earlier positions**
instead of compressing them into a single vector. This is exactly:
- Bahdanau attention (Session 1 concept)
- Self-attention (Session 2 — the Transformer)

The key insight: instead of one vector trying to summarize 200+ characters,
attention gives the model a way to "look back" at any previous position and pick
the relevant information.

---

### Section 8: Temperature comparison

With `T = 0.5` (conservative): the model produces repetitive but
grammatically plausible text. It picks the safest next character every time,
which leads to the same common phrases.

With `T = 1.2` (chaotic): the model produces more varied text but with more
spelling errors and invented words. Higher temperature forces the model to
explore less likely characters.

This is a fundamental trade-off in language generation: **diversity vs quality**.
In practice, most applications use `T` between 0.7 and 1.0.

---

## Part 2: Extension (`02_extension.py`)

### Understanding the TinyAttnLM architecture

```python
class TinyAttnLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)
```

**Key differences from the LSTM:**

- **No recurrence**: There is no hidden state passed from one step to the next.
  Instead, the model looks at all previous positions simultaneously through
  attention.
- **Positional embeddings**: Since there is no recurrence, the model has no
  built-in notion of order. `pos_emb` gives each position a learned vector so
  the model knows "this is position 3" vs "this is position 50".
- **Q, K, V projections**: These are the Query, Key, and Value matrices of the
  attention mechanism. Each serves a different role:
  - **Query** (Q): "What am I looking for?"
  - **Key** (K): "What do I contain?"
  - **Value** (V): "What information do I provide?"

**The causal mask:**

```python
mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
```

This creates an upper-triangular matrix of `-inf` values. When added to the
attention scores before softmax, it prevents position `t` from attending to any
future position `t+1, t+2, ...`. Without this mask, the model would "cheat" by
looking at the answer during training.

**Common student bug**: Forgetting the causal mask. Without it, the training
loss drops to near-zero (the model memorizes by peeking ahead), but generation
produces garbage.

---

### Section 5: Side-by-side qualitative comparison

After 5 epochs of training (same data, same learning rate, same number of
epochs):

| Dimension                  | LSTM                         | TinyAttn                                      |
|----------------------------|------------------------------|-----------------------------------------------|
| Local fluency              | High (most words are real)   | Medium (more invented words with d_model=128) |
| Recent context access      | Compressed into hidden state | Direct access inside the 128-char window      |
| Repetition after 200 chars | Strong (n-gram loops)        | Still possible, but failure mode differs      |
| Sentence completion        | Tends to never end           | Sometimes closes a line cleanly               |

**Expected 3-line summary:**

> Attention removes the fixed-size hidden-state bottleneck inside the available
> context window. In this lab, each generated character can directly look at the
> previous 128 characters instead of relying only on a compressed recurrent
> state. The price is that without recurrence, the model needs more parameters
> and more careful training to match LSTM-level local fluency.

**Why this matters for the rest of the course:**

This comparison is the empirical motivation for the Transformer architecture
(Session 2). The Transformer takes the attention idea from TinyAttnLM and adds:
- Multi-head attention (parallel attention with different learned patterns)
- Feed-forward layers (per-position non-linear transformations)
- Layer normalization (stabilizes training)
- Residual connections (allows gradient flow in deep models)

These additions solve the fluency problem while keeping the long-range context
advantage. That is the key insight of the "Attention Is All You Need" paper.

---

## Good coding practices — Summary

Here is a checklist of coding habits demonstrated in this lab:

| Practice | Why it matters |
|---|---|
| Use `torch.no_grad()` during inference | Saves memory and speeds up generation |
| Call `model.eval()` before generating | Disables dropout — your results are reproducible |
| Track train AND val loss every epoch | Detects overfitting early |
| Use `batch_first=True` in LSTM/RNN | Makes tensor shapes intuitive: (B, T, D) |
| Guard divisions: `max(x, 1e-6)` | Prevents NaN from division by zero |
| Use f-strings for logging | Cleaner output: `f"{loss:.3f}"` |
| Separate model, training, and utils | Modular code is easier to debug and reuse |
| Add docstrings to classes and functions | Helps your teammates (and future you) understand the code |
| Set `torch.manual_seed(42)` | Makes experiments reproducible across runs |

---

## Key takeaway

The LSTM is great at learning local patterns (character-level fluency) but
struggles with long-range dependencies because its entire past is compressed
into a single fixed-size vector. Attention solves this by letting each position
directly access previous positions. The Transformer (Session 2) is what happens
when you scale this idea up with multi-head attention, normalization, and
feed-forward layers.
