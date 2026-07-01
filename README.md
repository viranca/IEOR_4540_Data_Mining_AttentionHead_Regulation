# Regularizing Attention-Head Diversity in Transformers via CKA

Does explicitly encouraging Transformer attention heads to be *different from one another*
improve translation quality? This project adds a **Centered Kernel Alignment (CKA)**
regularizer to a standard encoder–decoder Transformer, runs an ablation over its strength,
and measures the effect on a German→English translation task.

**Short answer:** on this task, enforcing head diversity did **not** improve BLEU (it slightly
reduced it). The value of the project is the method and the honest ablation, not a win.

> Course project for **IEOR 4540 (Data Mining)**, Columbia University, under the guidance of
> **Dr. Krzysztof Choromanski** (Research Scientist, Google DeepMind; co-inventor of the
> Performer linear-attention Transformer).

---

## Attribution

The **baseline Transformer** (encoder–decoder architecture, training loop, and Multi30k /
`torchtext` data pipeline) is adapted from Ben Trevett's excellent open-source tutorial
**["6 - Attention Is All You Need"](https://github.com/bentrevett/pytorch-seq2seq)**.

The **original contribution of this project** is:

- a **CKA/HSIC head-diversity regularizer** added to the training objective;
- instrumentation that tracks CKA **separately** across encoder self-attention, decoder
  self-attention, and encoder–decoder cross-attention;
- an **ablation** over regularization strength (λ = 0.01 → 2.0) against the vanilla baseline;
- follow-up analysis on training length, head count, and verification of the CKA implementation.

Please credit Ben Trevett's tutorial if you reuse the baseline code.

---

## Background

Multi-head attention lets a Transformer attend to information from different representation
subspaces. In practice, heads can become **redundant** — several heads learn to do the same
thing. This project asks whether *penalizing* that redundancy during training (pushing heads to
be mutually dissimilar) helps the model generalize.

Head similarity is measured with **Centered Kernel Alignment (CKA)**, computed via the
**Hilbert–Schmidt Independence Criterion (HSIC)** with a linear kernel. CKA gives a value in
[0, 1] for how similar two heads' representations are. The regularizer adds the mean pairwise
CKA across heads to the loss, scaled by λ, so that minimizing the loss also pushes heads apart.

---

## Method

- **Task:** Multi30k German→English translation.
- **Model:** encoder–decoder Transformer — 3 encoder layers, 3 decoder layers, 4 attention
  heads per layer, `d_model = 256`, feed-forward dim 512, dropout 0.1.
- **Training:** Adam (lr = 5e-4), batch size 128, 4 epochs.
- **Regularizer:** `loss = cross_entropy + λ · mean_pairwise_CKA(heads)`, with CKA tracked
  independently for the three attention types listed above.
- **Ablation:** λ ∈ {0.01, 0.1, 0.5, 1.0, 2.0}, each compared to the vanilla baseline.
- **Evaluation:** cross-entropy / perplexity during training; BLEU on the test set.

---

## Results

| Model                          | Test BLEU |
| ------------------------------ | --------- |
| Vanilla baseline (no CKA)      | ≈ 36.5    |
| CKA-regularized (best, λ=0.01) | ≈ 33.2    |
| CKA-regularized (λ = 0.1–2.0)  | flat / lower |
| Drop-head variant              | not completed¹ |

**Takeaway:** across every regularization strength tried, enforcing head diversity did not beat
the vanilla baseline and generally reduced BLEU slightly. This is reported as-is: a negative
result is still a result. Follow-up checks (training length, head count, and verification of the
CKA/HSIC computation) were done to rule out an implementation artifact behind the finding.

¹ A "drop-head" variant was attempted but abandoned due to tensor-dimension issues in the
head-masking step; it is not part of the completed ablation.

> Exact BLEU varies slightly by run; the accompanying course paper reports the vanilla baseline
> at 36.52.

---

## Repository structure

| File                              | What it is |
| --------------------------------- | ---------- |
| `Transformer_Vanilla.ipynb`       | Baseline Transformer (adapted from Trevett), no CKA. |
| `Transformer_CKA_Vanilla.ipynb`   | Baseline with CKA **measured** (instrumented) but not regularized. |
| `Transformer_CKA_Regulized_*.ipynb` | CKA-**regularized** training at λ = 0.01 / 0.1 / 0.5 / 1.0 / 2.0. |
| `Transformer_CKA_Drophead.ipynb`  | Drop-head experiment (incomplete — see note above). |
| `Plots.ipynb`                     | Results and per-head CKA heatmaps / plots. |
| `.data/multi30k/`                 | Multi30k dataset (also auto-downloaded by `torchtext`). |

Trained model checkpoints (`*.pt`) are **not** committed — they are large and regenerate by
running the notebooks. See `.gitignore`.

---

## Setup & run

```bash
# Python 3.8+ recommended
pip install torch torchtext==0.6.0 spacy numpy matplotlib jupyter
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

jupyter notebook
```

Then, to reproduce:

1. Run `Transformer_Vanilla.ipynb` for the baseline.
2. Run any `Transformer_CKA_Regulized_<λ>.ipynb` to train a regularized model.
3. Run `Plots.ipynb` to reproduce the CKA/attention figures.

> Note: this project uses the legacy `torchtext` data API (`Field`, `BucketIterator`), which was
> current at the time (2022). Newer `torchtext` versions remove it; pin `torchtext==0.6.0` to run
> the notebooks unchanged.

---

## References

- Vaswani et al., *Attention Is All You Need* (2017).
- Kornblith et al., *Similarity of Neural Network Representations Revisited* (2019) — CKA.
- Ben Trevett, *PyTorch Seq2Seq* tutorials — baseline Transformer implementation.

---

*Author: Viranca R. I. Balsingh — Columbia University, IEOR 4540 (Data Mining), 2022.*
