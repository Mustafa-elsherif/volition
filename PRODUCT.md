# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Inferred (no interview, per the project brief's autonomous directive): static HTML, CSS, and vanilla JavaScript, no build step, so it deploys directly to GitHub Pages. In-browser model inference uses ONNX Runtime Web from a CDN. A separate FastAPI backend (Dockerized, under `api/`) serves the same model for programmatic use.

## Users

Primary: neuroscience and machine-learning researchers, students, and technical reviewers evaluating how reliably neural population activity predicts a mouse's choice before it moves. They arrive to understand the method, run the decoder on a real sample, and judge the honesty of the results. Secondary: engineers integrating the decoding API.

## Product Purpose

Volition decodes a mouse's binary wheel choice from the firing rates of recorded neurons in the 0.1 second before movement, using one model that spans recording sessions from different animals. It exists to show that a single, interpretable, shareable decoder can read neural population activity across sessions, and to report honestly where that signal transfers and where it does not. Success: a visitor runs a real trial, sees the decoded choice with its per-neuron and per-region drivers, and trusts the reported numbers because the limits are stated plainly.

## Positioning

A single permutation-invariant set transformer that reads a variable, non-overlapping set of neurons (each tagged by brain region) rather than a fixed per-session feature vector. This is the mechanism a per-session linear decoder cannot copy: it runs on any neuron count, degrades gracefully when neurons or a region are missing, and exposes the pooling attention as a faithful per-neuron explanation.

## Operating Context

Evaluated in a browser with no server for the live demo (inference runs client-side), or against the FastAPI backend for integration. The underlying data is the public IBL Brain-Wide Map (CC-BY 4.0), ten sessions from ten mice. The decoder works on cleaned, offline research trials, not a live brain-computer interface.

## Capabilities and Constraints

- In-browser inference on bundled real sample trials and on user-supplied trials, with no data leaving the page.
- Per-prediction explanation: pooling attention summed per neuron and per brain region.
- Honest reported metrics from held-out evaluation, including a negative cross-subject transfer result.
- A grounded multilingual assistant: backend-only Gemini when a key is configured, otherwise a local retrieval-grounded fallback over the project knowledge base; no user-facing API key field.
- Constraint: single-trial requests send already z-scored firing rates, because per-neuron normalization needs a whole session.

## Brand Commitments

- Name: **Volition**. Custom inline-SVG logo used across the product.
- Voice: plain, human, professional. No em dashes in user-facing copy. No marketing hype. Numbers are never inflated; limits are stated.
- Must read as its own identity, distinct from unrelated sibling projects, and must not resemble generic AI-dashboard templates.

## Evidence on Hand

- Ten processed IBL sessions in `data/processed/` (not committed); real held-out metrics produced by `src/models/run_experiments.py` into `results/model/`.
- Shipped model artifact in `artifacts/` (`decoder.onnx`, `metadata.json`, `samples.json`).
- Knowledge base at `api/app/knowledge_base.json`.
- No customer testimonials, no third-party endorsements, no pricing. Future work must not fabricate any.

## Product Principles

1. Honest first: report held-out numbers and the negative results, never a flattering subset.
2. Explain every prediction; attention is shown because it is what the model actually pooled.
3. One shared model, any neuron set: robustness to missing inputs is a feature, not an afterthought.
4. The live demo runs without a server, so anyone can verify it on real data.

## Accessibility & Inclusion

Responsive from 320px, light and dark themes, keyboard operable, WCAG AA contrast, motion respects reduced-motion preference.
