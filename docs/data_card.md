# Brain decoding data card

This project uses International Brain Laboratory (IBL) Brain-Wide Map trials and neural recordings to study how well pre-movement neural activity predicts a mouse's binary wheel choice. The configured frozen release is `2023_12_bwm_release`, with aggregate table tag `2024_Q2_IBL_et_al_BWM`.

| Item | Value |
| --- | --- |
| Dataset | IBL Brain-Wide Map ([official data release page](https://docs.internationalbrainlab.org/notebooks_external/data_release_brainwidemap.html)) |
| Data server | https://openalyx.internationalbrainlab.org (public read-only database, reached with ONE-api) |
| Release freeze | `2023_12_bwm_release` — 459 sessions, 699 probe insertions, 139 mice, 12 labs |
| Aggregate table tag | `2024_Q2_IBL_et_al_BWM` |
| Licence | CC-BY 4.0 — free to use and share; the International Brain Laboratory must be credited |
| Loading code | `ONE-api` 3.6.1 and IBL's `brainwidemap` package (commit `118fc36`), pinned in [requirements.txt](../requirements.txt) |

## Cohort and neural features

The team configuration targets 10 sessions with at most two per lab. The following thresholds come from [config.yaml](../config.yaml). The final session manifest is in [sessions.yaml](../sessions.yaml).

| Setting | Configured value |
| --- | --- |
| Unit quality | QC label `1.0` |
| Brain regions | Beryl mapping, excluding `root` and `void` |
| Session thresholds | At least 60 neurons, 400 trials, 3 errors, and 3 regions, with 10 neurons per region |
| Fallback thresholds | Reduce the region minimum to 2 if fewer than 10 sessions qualify. Reduce the neuron minimum to 40 if fewer than 10 still qualify. |
| Feature window | −0.1 to 0.0 seconds relative to the first wheel movement |

## Final cohort

Ten sessions passed the primary thresholds, so no fallback threshold was applied. Each session comes from a different mouse and no lab contributes more than two sessions. Fourteen further sessions are kept as reserves in [sessions.yaml](../sessions.yaml).

| Session | Mouse | Lab | Probes | Trials | Neurons | Regions |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `746d1902-fa59-4cab-b0aa-013be36060d5` | ZFM-01592 | mainenlab | 2 | 481 | 429 | 17 |
| `ee40aece-cffd-4edb-a4b6-155f158c666a` | ZM_2241 | mainenlab | 2 | 527 | 427 | 15 |
| `f10efe41-0dc0-44d0-8f26-5ff68dca23e9` | CSHL060 | churchlandlab | 2 | 429 | 396 | 9 |
| `56b57c38-2699-4091-90a8-aba35103155e` | SWC_054 | mrsicflogellab | 2 | 415 | 391 | 13 |
| `4d8c7767-981c-4347-8e5e-5d5fffe38534` | SWC_038 | mrsicflogellab | 2 | 562 | 389 | 10 |
| `6274dda8-3a59-4aa1-95f8-a8a549c46a26` | CSH_ZAD_025 | zadorlab | 2 | 591 | 349 | 10 |
| `2bdf206a-820f-402f-920a-9e86cd5388a4` | UCLA037 | churchlandlab_ucla | 2 | 427 | 346 | 9 |
| `8207abc6-6b23-4762-92b4-82e05bed5143` | CSH_ZAD_024 | zadorlab | 2 | 491 | 332 | 8 |
| `dfbe628d-365b-461c-a07f-8b9911ba83aa` | KS046 | cortexlab | 2 | 516 | 331 | 7 |
| `09394481-8dd2-4d5c-9327-f2753ede92d7` | ibl_witten_26 | wittenlab | 2 | 433 | 326 | 12 |

The cohort holds 4872 retained trials and 3716 neurons in 65 Beryl regions. Only 25 of those regions appear in more than one session; the other 40 appear in a single session, so a result about one region usually rests on one session.

Excluding `root` and `void` removed 295 of the 4011 good units, about 7 percent. The loss per session ranges from 0 to 12.6 percent. In the feature window the median neuron fires at 4.72 Hz and the mean is 11.05 Hz, and 59 neurons do not fire at all in the window.

[The S1.11 notebook](../notebooks/04_neurons_and_regions.ipynb) holds the per-session and per-region counts.

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

[The S1.9 notebook](../notebooks/03_trial_cleaning.ipynb) records the run for `6713a4a7-faed-4df2-acab-ee4e63326f8d` (`NYU-11`, 2020-02-18). This session serves as a workflow check and is not part of the final cohort.

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

The single-session audit verifies the cleaning workflow on that session. The cohort counts above come from the ten sessions in `data/processed`. Those files are rebuilt from IBL by [build_dataset.py](../src/data/build_dataset.py) and are not stored in the repository.
