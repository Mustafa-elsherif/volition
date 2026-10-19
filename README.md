# Brain Decoding: Reliable Pre-Movement Neural Decoding

## Summary

A mouse sees a picture on the left or right of a screen and turns a wheel to answer. Thin probes record its neurons. We take the firing rate of each neuron in the 0.1 s before the first wheel movement and train a simple model to predict the mouse's choice. Then we test how reliable the prediction stays when there is less training data, or when some neurons or a whole brain region are missing.

## Research question

How reliably can a simple model predict the mouse's choice from brain activity before movement, when (a) the training data is reduced, and (b) some neurons or a whole brain region are missing?

## Dataset

- **IBL Brain-Wide Map** (frozen release `2023_12_bwm_release`), 10 sessions from 10 different mice.
- Access: public, through ONE-api and `brainwidemap` (https://openalyx.internationalbrainlab.org).
- Licence: **CC-BY 4.0**. We must cite it (see References).

## Stages

| Stage | Days | What we make |
|---|---|---|
| 0. Setup | Oct 20 | the repo and the shared settings |
| 1. Data | Oct 21 - Oct 23 | 10 clean, ready session files |
| 2. Models | Oct 24 - Oct 26 | the predictions of the 6 experiments |
| 3. Evaluation | Oct 27 - Oct 28 | the 8 result tables and their notebooks |
| 4. Delivery | Oct 29 - Oct 30 | figures, dashboard, one command, report, v1.0 |

## Repo structure

```
brain-decoding/
├── .gitignore, requirements.txt, config.yaml, sessions.yaml, run_all.py, README.md
├── src/
│   ├── common/      settings.py, checks.py
│   ├── data/        sessions.py, clean_trials.py, neurons.py, features.py, build_dataset.py
│   ├── models/      models.py, splits.py, baselines.py, experiments.py, run_experiments.py
│   ├── evaluation/  calibration.py, metrics.py, statistics.py, make_tables.py
│   └── analysis/    eda.py, plots.py, controls.py, run_controls.py, make_figures.py
├── app/             dashboard.py
├── notebooks/       01_ ... 11_
├── tests/           test_<yourname>.py
├── data/raw/        IBL downloads (never uploaded)
├── data/processed/  10 ready sessions
├── results/         predictions/, tables/, figures/
└── docs/            data_card.md, literature_notes.md, report parts, final_report.md, slides.pdf
```

## Install

```
git clone https://github.com/Mustafa-elsherif/brain-decoding.git
cd brain-decoding
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Mac or Linux use `python3.12 -m venv .venv` and `source .venv/bin/activate`.

## Run commands

| # | Command | Reads -> writes |
|---|---|---|
| 1 | `python -m src.data.build_dataset` | sessions.yaml + IBL -> data/processed/ |
| 2 | `python -m src.models.run_experiments` | data/processed/ -> results/predictions/ |
| 3 | `python -m src.evaluation.make_tables` | predictions -> 5 tables |
| 4 | `python -m src.analysis.run_controls` | predictions -> 3 tables |
| 5 | `python -m src.analysis.make_figures` | data + 8 tables -> 8 figures |
| all | `python run_all.py` (`--quick` = first session only) | runs 1 -> 5 in order |
| app | `streamlit run app/dashboard.py` | opens the dashboard |

## Team

- Mustafa (team leader)
- Ahmed
- Mohamed
- Omar

## References

- IBL Brain-Wide Map data release (CC-BY 4.0), International Brain Laboratory: https://openalyx.internationalbrainlab.org
- IBL Nature (2025) paper: link to be added (see `docs/literature_notes.md`)
- BrainWideBench (2026) paper: link to be added (see `docs/literature_notes.md`)

*No results yet. They are added in Stage 4.*
