# Brain decoding data card

This project uses International Brain Laboratory (IBL) Brain-Wide Map trials and neural recordings to study how well pre-movement neural activity predicts a mouse's binary wheel choice. The configured frozen release is `2023_12_bwm_release`, with aggregate table tag `2024_Q2_IBL_et_al_BWM`.

## Cohort and neural features

The team configuration targets 10 sessions with at most two per lab. The following thresholds come from [config.yaml](../config.yaml). The final session manifest (`sessions.yaml`) and unit-selection results are pending.

| Setting | Configured value |
| --- | --- |
| Unit quality | QC label `1.0` |
| Brain regions | Beryl mapping, excluding `root` and `void` |
| Session thresholds | At least 60 neurons, 400 trials, 3 errors, and 3 regions, with 10 neurons per region |
| Fallback thresholds | Reduce the region minimum to 2 if fewer than 10 sessions qualify. Reduce the neuron minimum to 40 if fewer than 10 still qualify. |
| Feature window | −0.1 to 0.0 seconds relative to the first wheel movement |

## Trial cleaning

[S1.3](../src/data/clean_trials.py) loads the IBL trials and applies the official inclusion mask using `nan_exclude="default"` and `exclude_nochoice=True`. The audit assigns exclusions in order to no choice (`choice == 0`), missing required trial fields, reaction time outside the inclusive 0.08–2.0 second interval, and any additional IBL exclusions. The required fields are `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`.

Reaction time is `firstMovement_times - stimOn_times`. Zero-contrast trials can be retained. The direction audit uses only retained nonzero-contrast trials. It checks the consistency of the choice and stimulus signs against feedback and records the convention.

| Derived field | Definition |
| --- | --- |
| `trial_number` | Zero-based row position in the input table returned by IBL |
| `y` | `1` for `choice == +1` and `0` for `choice == -1` |
| `stim_side` | `+1` for a left stimulus and `-1` for a right stimulus |
| `contrast` | Left contrast is positive and right contrast is negative, with zero retained |
| `is_error` | True when `feedbackType == -1` |
| `is_zero_contrast` | True when signed contrast is zero |
| `prob_left` | Original `probabilityLeft` value |
| `reaction_time` | Seconds from stimulus onset to first movement |

## Verified example

[The S1.9 notebook](../notebooks/03_trial_cleaning.ipynb) records the run for `6713a4a7-faed-4df2-acab-ee4e63326f8d` (`NYU-11`, 2020-02-18). This session serves as a workflow check. Its membership in the final cohort is unconfirmed.

| Cleaning stage | Excluded | Remaining |
| --- | ---: | ---: |
| Trials returned by the loader | — | 565 |
| No choice | 9 | 556 |
| Missing required fields | 28 | 528 |
| Reaction time out of range | 134 | 394 |
| Additional IBL exclusions | 0 | 394 |

The IBL mask check passed. The choice-direction check reported the `same` convention with 100% agreement across 360 retained nonzero-contrast trials. Counts start with the table returned by IBL and exclude any trials the loader truncated earlier.

## Intended use and limits

Use the cleaned trials for binary choice decoding with neural features aligned to movement onset. Report the filtering criteria when interpreting results. Excluding no-choice, missing-event, and out-of-range trials limits conclusions to the retained trials. Movement alignment uses the observed first movement time, which limits what the resulting evaluation says about real-time prediction.

The single-session audit verifies the cleaning workflow on that session. Cohort-level coverage and exclusions still need the final manifest, per-session counts, and retained unit and region summaries. Record any fallback thresholds actually applied when those results are available.
