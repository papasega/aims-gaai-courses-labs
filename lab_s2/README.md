# Lab S2: Visualizing Attention in a Pre-trained Transformer

Module 0, Session 2. **Baseline:** load a small multilingual BERT, run it on three
sentences (English, French, Wolof translit), visualize attention weights, and observe
how different heads specialize. **Extension:** compute attention by hand on a 3-token
example and verify against PyTorch.

This lab follows the same structure as `lab_s1/` and the
[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) convention.

Note: this is the evening attention-inspection lab announced in the Session 2
slides. The in-class MiniViT/MNIST notebook is a separate guided lab.

## What you will produce

1. 3 attention heatmaps from different heads/layers on the canonical "it" sentence.
2. A French + Wolof transliteration sentence visualized to feel multilingual behavior.
3. (Bonus) A manually computed attention on 3 tokens, verified numerically.

## Time budget

- Baseline: 30 min (no training, just inference).
- Extension: +15-20 min.

## Layout

```
lab_s2/
├── README.md
├── requirements.txt
├── src/
│   ├── attention_viz.py      load model/device, extract attentions, heatmaps
│   └── manual_attention.py   step-by-step Q K V computation
├── 01_baseline.ipynb         main notebook for students
├── 01_baseline.py            jupytext source for the same notebook
├── 02_extension.ipynb        bonus manual attention notebook
├── 02_extension.py           jupytext source for the same notebook
└── solutions/
    └── SOLUTIONS.md
```

## Model and utilities

- Model used in the baseline: `distilbert-base-multilingual-cased`
  (`AutoTokenizer` + `AutoModel(output_attentions=True)`).
- Utility scripts used by the notebooks:
  - `src/attention_viz.py`: model loading, CPU/GPU device selection, attention
    extraction, heatmaps.
  - `src/manual_attention.py`: hand-written scaled dot-product attention.
- Keep the notebooks, `requirements.txt`, and the `src/` folder together. If you
  download or upload only the notebook, imports such as
  `from src.attention_viz import ...` will fail.

## Setup

```bash
cd labs/lab_s2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run will download `distilbert-base-multilingual-cased` (~530 MB) from
the Hugging Face Hub. Subsequent runs use the local cache.

## Run

```bash
jupyter notebook 01_baseline.ipynb
```

The `.ipynb` notebooks are already provided. If you edit the `.py` jupytext
sources and want to regenerate notebooks, run:

```bash
jupytext --to ipynb 01_baseline.py
jupytext --to ipynb 02_extension.py
```

Or open `01_baseline.ipynb` directly in VS Code, PyCharm, JupyterLab, or Colab.

## Colab / VS Code options

If you need a GPU, or if your local setup is slow, use Google Colab. This lab is
small and CPU is enough, but Colab can be more convenient because the environment
is already close to ready.

Recommended Colab workflow:

1. Download or upload the full `lab_s2/` folder, not only the notebook.
2. Make sure `01_baseline.ipynb`, `02_extension.ipynb`, `requirements.txt`, and
   `src/` are in the same folder.
3. In Colab, run:

```python
!pip install -q -r requirements.txt
```

4. If you want GPU in Colab: `Runtime` -> `Change runtime type` -> `GPU`.
   The helper `load_model()` automatically uses CUDA when it is available.

VS Code + Colab workflow:

1. Install a Google Colab extension in VS Code.
2. Open `01_baseline.ipynb` from the full `lab_s2/` folder.
3. Connect the notebook to a Colab runtime.
4. If needed, set the Colab runtime to GPU.
5. Keep the `src/` folder next to the notebook so the utility imports work.

## Hardware

- **CPU is enough.** Inference on a 12-token sentence is sub-second. No GPU needed.
- GPU is optional and mainly useful if you experiment with larger models. If a
  CUDA GPU is available, `src/attention_viz.py` will use it automatically.
- Colab works well, provided the notebook and utility scripts stay together.

## Deliverable

Push to GitHub Classroom: `lab-s2-attention/` containing:
- the completed notebook,
- 3 heatmaps with one-line interpretation each,
- the manual attention worksheet (extension, optional but recommended).

## What this lab proves

Different attention heads learn different roles without supervision: previous-word,
coreference, syntactic dependency. This is the empirical justification for
multi-head attention. The extension closes the loop: you can compute attention with
a calculator. The Transformer is not magic.
