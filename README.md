# AIMS Applied GenAI — Lab S1 Repository

**Course:** Applied Generative and Agentic AI (30h)
**Instructor:** Dr. Papa-Séga WADE — Orange Innovation / AIMS Senegal

This GitHub Classroom repository contains the Session 1 lab for Module 0:
**From Char-LSTM to Attention**. The lab is self-contained and uses a Wolof
character-level language modeling task to make the limits of recurrent memory
concrete before introducing attention.

---

## Lab

| Lab | Topic | Session |
|-----|-------|---------|
| [`lab_s1/`](./lab_s1/) | From Char-LSTM to Attention (Wolof corpus) | Session 1 |

---

## Quick Start

```bash
# 1. Accept the GitHub Classroom assignment link -> repo created for you
# 2. Clone your personal repo
git clone https://github.com/PSW-Generative-and-Agentic-AI-AIMS-classroom/lab-session-1-YOUR_USERNAME.git
cd lab-session-1-YOUR_USERNAME

# 3. Create and activate a local environment
python -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
cd lab_s1
python -m pip install -r requirements.txt

# 5. Open the baseline notebook
jupyter notebook 01_baseline.ipynb
```

The first notebook cell creates `lab_s1/data/wolof.txt` automatically. It tries
to download a small Wolof sample and falls back to a bundled Wolof proverb sample
if the network is unavailable.

---

## Submission

Complete the TODO section in `01_baseline.ipynb`:

- generate text from the short, medium, and long prompts;
- annotate when the LSTM starts drifting;
- describe repetition, context loss, and one architectural fix idea.

The extension notebook `02_extension.ipynb` is optional unless your instructor
explicitly asks for it.

Push your completed notebook before the deadline:

```bash
git add .
git commit -m "lab_s1: completed baseline + observations"
git push
```

---

*AIMS Senegal — Module 0: From Neurons to Transformers*
