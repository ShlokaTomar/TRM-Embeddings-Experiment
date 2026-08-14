# Is the Latent Reasoning Entangled?


This repository contains the code and experiments behind our blog post investigating whether latent-embedding approaches to ARC-AGI (like the Tiny Recursive Model) genuinely disentangle "what transformation to apply" from "how to execute it" — or whether the two remain entangled in the model's parameters.

> 📖 Full write-up: *[AI Club, CFI's Substack](https://aiclubiitmadras.substack.com/)*

---

## Overview

Solving ARC-AGI's visual grid-reasoning tasks usually pushes SoTA models toward expensive step-by-step Chain-of-Thought (CoT) reasoning. A newer line of work takes a different approach: encode the puzzle's transformation logic as a point in a high-dimensional **latent embedding space**, with a small shared network acting purely as an *execution engine* that runs whatever rule the embedding encodes.

This is called the **neural hash map hypothesis** (following Roye-Azar et al.), and this project stress-tests it across three concrete instances from the literature:

- **TRM** — Tiny Recursive Model
- **VARC (ViT variant)** — Vision Transformer approach to ARC
- **Plain Transformer** — a per-task-embedding baseline without the recursive machinery

## The Three Hypotheses

| # | Hypothesis | Claim |
|---|------------|-------|
| 1 | **Clean Separation** | The puzzle embedding cleanly isolates the transformation "program"; the model is strictly an execution engine. |
| 2 | **Partial Hash Map** | Transformation logic is *shared* between the puzzle embedding and the model's parameters. |
| 3 | **Parameter-Dominant** | The neural-hash-map theory fails; logic lives dominantly in the model's parameters, not the embedding. |

**Our conclusion: none of the tested approaches cleanly satisfy Hypothesis 1. The evidence points to Hypothesis 2 — a partial hash map, where logic is entangled between the per-task embedding and the model trunk.**

## Background: ARC-AGI

[ARC-AGI](https://arcprize.org/) (Abstraction and Reasoning Corpus for Artificial General Intelligence) is a benchmark created by François Chollet to measure fluid intelligence and skill-acquisition efficiency rather than memorization. Humans score near 100%; SoTA AI models still struggle, and largely can't brute-force their way to a good score — solving it well is meant to require actually *inferring and applying* a rule.

TRM, briefly: instead of a long CoT trace, it refines a compact latent reasoning state `z` and answer state `y` through nested recursive loops using a single shared shallow Transformer `f` (~7M parameters):

```
For t = 1 ... T outer iterations:
    For k = 1 ... N inner reasoning steps:
        z := f(x, y, z)
    y := f(y, z)
```

TRM also learns a *separate embedding row per training puzzle (and per augmented variant of every puzzle)*. For ARC-AGI-1's fully augmented training set, that's roughly **876,000 embedding rows at 512 dimensions** — on the order of 449M additional parameters sitting next to the "7M-parameter" trunk. Whether that table is a legitimate index into shared reasoning, a genuine logic/execution mix, or disposable memorization is the question this repo investigates.

## Key Findings

### Experiment I — Invariance Against Augmentations
- Color-permuted inputs (fed with the correct puzzle ID) still solved correctly with **94.72% mean cell accuracy** (vs. 95.60% baseline) and **80% exact-match** (24/30).
- Un-permuting the model's output and comparing to ground truth held at **91.25%** accuracy — the model was tracking new colors through the rule, not outputting noise.
- **Averaging ~1,000 per-augmentation embeddings into one vector** (no fine-tuning) barely hurt performance: **100% → 99.56% cell accuracy** across 45 ARC-AGI-2 tasks — evidence the embeddings cluster around a shared "task center" rather than acting as ~1,000 unrelated hash entries.
- Naive cosine similarity between augmentations of the same task is misleadingly low (~0.1–0.6, depending on measurement), because 512-D spaces make even same-cluster points look orthogonal. Switching to **Gaussian Mixture Model (GMM)** clustering on 28,042 embeddings (30 tasks) gave an **Adjusted Rand Index of 0.757** and **84.69% clustering accuracy** — strong evidence that augmented variants of the same task form cohesive density clouds, not scattered lookup keys.

### Experiment II — Blank ID & Cross-Task Study
- Replacing a task's embedding with a **blank (never-trained) ID** collapses performance by **~65 points of cell accuracy and ~87 points of exact-match**.
- With no task ID at all, only 4/40 outputs reproduce the input exactly; mean cell accuracy vs. the raw input is 40.27%, and the model shows strong background-preserving priors — evidence of transformation biases baked into the trunk itself.
- **Fine-tuning only the task-ID embedding fails** (near-zero exact match on held-out tasks); full fine-tuning of the embedding *and* the trunk is required. If the embedding were a fully independent "program," tuning it alone should have been enough — it isn't.

### Plain Transformer & VARC (ViT)
- Removing per-task embeddings entirely from a vanilla transformer dropped ARC-AGI-1 accuracy from **44% → 24%** — task embeddings hold substantial, but not all, of the needed information.
- On an 18M-parameter ViT (VARC-style, 10 encoder layers, 512-dim embeddings): test-time fine-tuning **everything** gave 89.33% pixel / 46.88% exact-match accuracy, while fine-tuning **only the task embedding** (trunk frozen) gave just 74.4% pixel / 6.67% exact-match — again pointing to entangled, not cleanly separated, logic.

## Repository Structure

| File | Description |
|---|---|
| `TRM_embedding_experiment.ipynb` | Core TRM puzzle-ID embedding experiments — color permutation, geometric augmentation transfer, and embedding-averaging tests. |
| `trm_embedding_experiment_on_ARC1.ipynb` | Blank-ID, color-permutation, and augmentation experiments run on the ARC-AGI-1 train set. |
| `trm-arc-2.ipynb` | Averaged-embedding, cosine-similarity/PCA, and GMM clustering experiments on ARC-AGI-2 tasks. |
| `ttt_on_only_tasktoken.ipynb` | Test-time training with **only the task-ID embedding** unfrozen (trunk frozen). |
| `ttt_on_tasktoken_and_model.ipynb` | Test-time training with **both the task-ID embedding and the full model trunk** unfrozen. |
| `vit_training.ipynb` | Training and test-time fine-tuning of the VARC-style ViT model (18M params, 64×64 canvas, 2×2 patches, 10 encoder layers). |
| `README.md` | This file. |

## Future Work

- Is the neural hash map the *optimal* way to represent abstract-reasoning logic — should it be a single vector, a subspace, or a non-linear manifold?
- If disentanglement is the goal, what architecture/training encourages it? We suspect VAE-style architectures (à la Latent Program Search) may push harder toward the clean neural-hash-map theory.
- Do LLMs solving ARC tasks via CoT carry an analogous "puzzle embedding" in their residual stream — and is it represented linearly?

## Authors

Rishi Sujith · Shloka Tomar · Rohan Bohra · Smitali Bhandari
*Project Viveka 2.0, AI Club, Center for Innovation, IIT Madras*

## References

1. Chollet, F. — [Abstraction and Reasoning Corpus for Artificial General Intelligence](https://arcprize.org/)
2. Roye-Azar, A., Vargas-Naranjo, S., Ghai, D., Balamurugan, N., Amir, R. — [Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute](https://arxiv.org/abs/2512.11847)
3. Jolicoeur-Martineau, A. — [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
4. McGovern, R. K. — [Test-time Adaptation of Tiny Recursive Models](https://arxiv.org/abs/2511.02886)
5. Vakde, M. — [44% on ARC-AGI-1 in 67 cents](https://mvakde.github.io/blog/44-on-arc-1/)
6. Hu, K., Cy, A., Qiu, L., Ding, X. D., Wang, R., Zhu, Y. E., Andreas, J., He, K. — [ARC Is a Vision Problem!](https://arxiv.org/abs/2511.14761v1)

## Citation

If you find this work useful, please cite our blog post:

```
Rishi Sujith, Shloka Tomar, Rohan Bohra and Smitali Bhandari (2026). Is the Latent Reasoning Entangled?.
Project Viveka 2.0, AI Club, Centre for Innovation, IIT Madras.
```



