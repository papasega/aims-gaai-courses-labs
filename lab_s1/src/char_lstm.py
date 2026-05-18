"""Character-level LSTM for text generation.

Reference: Karpathy's char-rnn (2015), simplified for pedagogy.
"""

import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    """A 2-layer LSTM operating on character embeddings.

    Args:
        vocab_size: number of distinct characters in the corpus.
        embedding_dim: size of each character vector.
        hidden_dim: size of the LSTM hidden state.
        num_layers: number of stacked LSTM layers.
        dropout: applied between LSTM layers.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.embedding(x)              # (B, T, E)
        out, hidden = self.lstm(emb, hidden) # (B, T, H)
        logits = self.fc(out)                # (B, T, V)
        return logits, hidden

    @torch.no_grad()
    def generate(
        self,
        seed: str,
        char_to_idx: dict,
        idx_to_char: dict,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> str:
        """Generate text autoregressively starting from `seed`.

        Temperature controls randomness:
          - 0.5: conservative (more deterministic)
          - 1.0: neutral
          - 1.5+: creative / chaotic
        """
        self.eval()
        idx = torch.tensor(
            [[char_to_idx[c] for c in seed if c in char_to_idx]],
            dtype=torch.long,
            device=device,
        )
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
