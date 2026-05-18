"""Download a Wolof text corpus for the character-level LSTM lab.

Sources explored:
  - CC100-Wolof (Common Crawl, monolingual Wolof text)
  - GalsenAI Wolof datasets on Hugging Face

Run once:
    python data/download_wolof.py
"""

import os
import urllib.request
from pathlib import Path

# CC100 Wolof sample (monolingual text from Common Crawl)
URL = "https://data.statmt.org/cc-100/wo.txt.xz"
OUT_XZ = Path(__file__).parent / "wolof_cc100.txt.xz"
OUT = Path(__file__).parent / "wolof.txt"

MAX_CHARS = 500_000  # keep ~500K chars (enough for char-LSTM, fast training)


def main():
    if OUT.exists():
        size_kb = os.path.getsize(OUT) / 1024
        print(f"Already present: {OUT} ({size_kb:.1f} KB)")
        return

    # Try CC100 first
    try:
        print(f"Downloading CC100 Wolof from {URL} ...")
        urllib.request.urlretrieve(URL, OUT_XZ)
        print("Decompressing...")
        import lzma
        with lzma.open(OUT_XZ, "rt", encoding="utf-8") as f:
            text = f.read(MAX_CHARS)
        OUT.write_text(text, encoding="utf-8")
        OUT_XZ.unlink()  # remove compressed file
        size_kb = os.path.getsize(OUT) / 1024
        print(f"Saved to {OUT} ({size_kb:.1f} KB, {len(text):,} chars)")
        return
    except Exception as e:
        print(f"CC100 download failed: {e}")
        print("Falling back to a small built-in Wolof sample...")

    # Fallback: a small built-in sample (proverbs + sentences)
    wolof_sample = """Wolof ak Pulaar ak Seereer ak Joola ak Mandinka mooy làkk yi ñu gëna wax ci Senegaal.
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
Benn nit mënul a def loxo.
Ku nekk fii, war na def lu baax.
Am naa jabar ak doom.
Ndox mi dafa sedd.
Jigéen ji dafa jàng ba pare.
"""
    # Repeat to get enough text for training
    text = (wolof_sample * 200)[:MAX_CHARS]
    OUT.write_text(text, encoding="utf-8")
    size_kb = os.path.getsize(OUT) / 1024
    print(f"Saved built-in Wolof sample to {OUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
