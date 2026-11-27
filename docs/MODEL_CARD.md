# Model card: Volition neuron-set transformer

## Overview

Volition decodes a mouse's binary wheel choice from the firing rates of recorded
neurons in the 0.1 second before the first wheel movement. The model is a
permutation-invariant set transformer: each recorded neuron is a token built from its
z-scored firing rate and a learned embedding of its Beryl brain region. Multi-head
self-attention pools the neuron set into one trial vector, and a small head predicts
the choice. Because region identities are shared across sessions, one set of weights
reads sessions with different and non-overlapping neurons.

- **Version:** 1.0.0
- **Task:** binary pre-movement choice decoding
- **Input:** a set of neurons, each with a z-scored firing rate and a Beryl region acronym
- **Output:** probability of choice = +1 (right), plus per-neuron pooling attention
- **Serialized formats:** `model.pt` (PyTorch), `decoder.onnx` (ONNX, probability and attention outputs)

## Training data

International Brain Laboratory Brain-Wide Map, frozen release `2023_12_bwm_release`
(CC-BY 4.0). Ten sessions from ten mice, at most two per lab, selected for unit
quality and minimum neuron, trial, error, and region counts. Trials are cleaned with
the official IBL inclusion criteria and aligned to the first wheel movement. Features
are firing rates in the window from -100 ms to movement onset (half-open, so a spike
at movement onset is excluded). Firing rates are z-scored per neuron using only that
session's trials, which uses no labels. See [docs/data_card.md](data_card.md).

## Training procedure

- Backbone: 2 transformer encoder layers, model width 64, 4 heads, GELU, pre-norm.
- Optional self-supervised pretraining by masked-neuron reconstruction (15% of tokens).
- Supervised fine-tuning with binary cross-entropy and early stopping on a held-out validation split.
- Optimizer AdamW, learning rate 1e-3, weight decay 1e-4, batch size 256.
- Trained on an NVIDIA RTX 4060 (CUDA). The dataset is small; training a fold takes well under a minute.

## Evaluation

Held-out trials, mean ± std over three seeds.

| Protocol | Metric | Value |
|---|---|---|
| Pooled held-out (shared model) | ROC AUC | 0.748 ± 0.026 |
| Pooled held-out (shared model) | Balanced accuracy | 0.685 ± 0.017 |
| Pooled held-out (shared model) | Expected calibration error | 0.044 ± 0.010 |
| Per-session logistic regression | ROC AUC | 0.888 ± 0.009 |
| Per-session random forest | ROC AUC | 0.930 ± 0.005 |
| Leave-one-session-out (transformer) | ROC AUC | 0.496 ± 0.064 |
| Leave-one-session-out (region-pooled logreg) | ROC AUC | 0.541 ± 0.096 |

Robustness (shipped model, cohort trials), balanced accuracy as neurons are removed:
0.70 at 0%, 0.67 at 25%, 0.67 at 50%, 0.62 at 75%, 0.56 at 90%; dropping the single
largest region per session gives 0.64.

## Intended use

Research and education: understanding how pre-movement neural population activity
relates to an animal's choice, comparing decoding approaches, and demonstrating
interpretable, in-browser neural decoding. The shipped weights are trained on all ten
sessions; every reported accuracy comes from held-out evaluation, never from the data
the shipped model trained on.

## Limitations

- This is an offline research decoder, not a live brain-computer interface.
- A single shared model is less accurate than a decoder retrained per animal. Region-level shared features lose the per-neuron information that per-session models exploit.
- The model does not transfer to an unseen animal: leave-one-session-out performance is near chance.
- Self-supervised pretraining did not produce a reliable gain on this dataset.
- Choice correlates with the stimulus and the animal's prior, so the decoder reads a mix of signals, not a pure motor intention.
- Conclusions hold for the retained trials and the chosen pre-movement window. The window uses the observed first-movement time, which limits claims about real-time prediction.

## Ethical considerations

The data is public animal electrophysiology released by the International Brain
Laboratory under CC-BY 4.0 and must credit IBL. The browser demo runs inference
locally, so trials loaded in the demo stay on the user's device. No human data is
involved.
