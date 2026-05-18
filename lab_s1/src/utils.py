"""Tokenizer, batching, training, and evaluation helpers."""

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------------
# Tokenization (character-level)
# ----------------------------------------------------------------------------
def build_vocab(text: str):
    """Return (chars, char_to_idx, idx_to_char) for a character-level model."""
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return chars, char_to_idx, idx_to_char


def encode(text: str, char_to_idx: dict) -> torch.Tensor:
    return torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)


def decode(ids, idx_to_char: dict) -> str:
    return "".join(idx_to_char[int(i)] for i in ids)


# ----------------------------------------------------------------------------
# Dataset of overlapping windows (next-character prediction)
# ----------------------------------------------------------------------------
class CharWindowDataset(Dataset):
    """Each item is a (input, target) pair where target is input shifted by 1.

    Args:
        data: encoded character tensor.
        seq_length: window length.
        stride: step between consecutive windows. stride=1 gives maximum
            overlap (slow), stride=seq_length gives no overlap (fast).
            Default 4 is a good speed/coverage trade-off.
    """

    def __init__(self, data: torch.Tensor, seq_length: int, stride: int = 4):
        self.data = data
        self.seq_length = seq_length
        self.stride = max(stride, 1)
        self._len = max((len(data) - seq_length - 1) // self.stride, 0)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start : start + self.seq_length]
        y = self.data[start + 1 : start + 1 + self.seq_length]
        return x, y


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def make_loaders(
    text: str,
    char_to_idx: dict,
    seq_length: int,
    batch_size: int,
    train_frac: float = 0.9,
    stride: int = 4,
    max_chars: int | None = None,
):
    """Build train/val DataLoaders.

    Args:
        stride: step between windows (higher = faster, default 4).
        max_chars: cap the corpus at this many characters (None = use all).
    """
    if max_chars is not None:
        text = text[:max_chars]
    data = encode(text, char_to_idx)
    split = int(len(data) * train_frac)
    train_ds = CharWindowDataset(data[:split], seq_length, stride=stride)
    val_ds = CharWindowDataset(data[split:], seq_length, stride=stride)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader


# ----------------------------------------------------------------------------
# Corpus loading
# ----------------------------------------------------------------------------
def load_corpus(path: str = "data/tinyshakespeare.txt") -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}. Run: python data/download_corpus.py"
        )
    return p.read_text(encoding="utf-8")
