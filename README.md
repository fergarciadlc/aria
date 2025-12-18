# ARIA Dataset

**Authentic vs Reconstructed Isolated Audio**

ARIA is a dataset designed for evaluating audio models on the task of distinguishing, mixing, and analyzing real (authentic) and AI-generated (reconstructed) isolated stems.
The name ARIA comes from the musical term meaning a solo vocal piece, reflecting the dataset's focus on isolated sources.

---

## Overview

ARIA provides a structured set of audio stems and their corresponding mixtures, enabling systematic experiments in:
- real vs synthetic audio detection
- hybrid real/AI mixture analysis
- source separation
- stem replacement evaluation
- generative audio benchmarking

Each song includes multiple isolated stems, both in their authentic form and in AI-reconstructed versions.
From these, ARIA generates all possible combinations of real and reconstructed stems, grouped by how many reconstructed stems appear in each mix.

---

## Folder Structure

Each song is stored in its own directory:

```
<song_name>/
│
├── stems/
│   ├── real/        # authentic stems
│   │     r0.wav
│   │     r1.wav
│   │     ...
│   │     r(n-1).wav
│   │
│   └── ai-gen/      # reconstructed (AI-generated) stems
│         f0.wav
│         f1.wav
│         ...
│         f(n-1).wav
│
└── mix/
    ├── U0/          # mixes with 0 fake stems (all real)
    ├── U1/          # mixes with exactly 1 fake stem
    ├── U2/          # mixes with exactly 2 fake stems
    ├── ...
    ├── U(n-1)/      # mixes with n−1 fake stems
    └── Un/          # mixes with all stems fake
```
The number of stems n may vary per song.

---

## Mix Naming Convention

Inside each `Uk/` folder, mixes are named according to which stem indices have been replaced by AI-generated versions.

**Examples:**

```
mix_S-empty.wav          # all stems real
mix_S-3.wav              # stem 3 replaced
mix_S-0-2.wav            # stems 0 and 2 replaced
mix_S-0-1-2-3.wav        # all stems replaced
```

This structure ensures reproducible and complete coverage of all possible hybrid combinations.

---

## Conceptual Definition

For each song:
1. There are n stems, indexed from 0 to n−1.
2. Each stem has two versions:
   - Authentic (r_i)
   - Reconstructed (f_i)
3. A mix is created by choosing any subset of stem indices to replace with their reconstructed counterparts.
4. All mixes with exactly k reconstructed stems are grouped in folder `Uk`.
5. This yields 2ⁿ total mixtures per song.

---

## Intended Use Cases

ARIA is suited for research in:
- audio forgery and authenticity detection
- hybrid generative modeling
- source replacement and editing methods
- separation and remix evaluation
- robustness testing for audio models
- training or benchmarking discriminators and classifiers

