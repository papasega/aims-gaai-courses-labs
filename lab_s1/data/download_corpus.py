"""Download tiny shakespeare (~1.1 MB) for the character-level LSTM lab.

Run once before opening the notebook:
    python data/download_corpus.py
"""

import os
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = Path(__file__).parent / "tinyshakespeare.txt"


def main():
    if OUT.exists():
        size_kb = os.path.getsize(OUT) / 1024
        print(f"Already present: {OUT} ({size_kb:.1f} KB)")
        return
    print(f"Downloading from {URL} ...")
    urllib.request.urlretrieve(URL, OUT)
    size_kb = os.path.getsize(OUT) / 1024
    print(f"Saved to {OUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
