"""Offline S1.3 tests; runnable with pytest or standard-library unittest.

All trials here are deliberately synthetic. These are correctness checks,
not experimental results, and they never contact IBL.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.data.clean_trials import (
    _reaction_time_limits,
    _validate_official_mask,
    check_choice_direction,
    clean_trials,
    load_trials,
    main,
)


def make_trials(n=8):
    """Create chronological trials with a known, consistent sign convention."""
    left = np.arange(n) % 2 == 0
    stimulus_time = np.arange(n, dtype=float) * 5
    side = np.where(left, 1, -1)
    choice = side.copy()
    if n > 2:
        choice[2] *= -1  # A real error must still satisfy the convention check.
    return pd.DataFrame({
        "stimOn_times": stimulus_time,
        "firstMovement_times": stimulus_time + 0.5,
        "feedback_times": stimulus_time + 0.8,
        "choice": choice.astype(float),
        "feedbackType": np.where(choice == side, 1.0, -1.0),
        "probabilityLeft": np.resize([0.2, 0.5, 0.8], n),
        "contrastLeft": np.where(left, 0.5, np.nan),
        "contrastRight": np.where(left, np.nan, 0.5),
    })


class TestTrialCleaning(unittest.TestCase):
    """Exercise trial selection, audit counts, labels and failure paths."""

    def setUp(self):
        """Use independent trial tables and settings in every test."""
        self.config = {"data": {"min_reaction_time": 0.08, "max_reaction_time": 2.0}}
        self.trials = make_trials()
        self.mask = pd.Series(True, index=self.trials.index)

    def test_derived_columns_and_original_columns(self):
        """Check explicit expected values for choices, sides, errors and time."""
        result, report = clean_trials(self.trials, self.mask, self.config)
        self.assertEqual(result["y"].tolist(), [1, 0, 0, 0, 1, 0, 1, 0])
        self.assertEqual(result["stim_side"].tolist(), [1, -1, 1, -1, 1, -1, 1, -1])
        self.assertEqual(result["contrast"].tolist(), [0.5, -0.5] * 4)
        self.assertEqual(result["is_error"].tolist(), [False, False, True, False, False, False, False, False])
        self.assertEqual(result["reaction_time"].tolist(), [0.5] * 8)
        self.assertEqual(result["prob_left"].tolist(), self.trials["probabilityLeft"].tolist())
        self.assertEqual(report["choice_direction"]["convention"], "same")
        self.assertTrue(report["ibl_check"]["passed"])
        pd.testing.assert_frame_equal(result[self.trials.columns], self.trials)

    def test_sequential_counts_and_extra_ibl_exclusions(self):
        """Overlapping exclusion reasons count once, in the specified order."""
        trials = make_trials(12)
        trials.loc[0, ["choice", "feedback_times"]] = [0, np.nan]
        trials.loc[1, "stimOn_times"] = np.nan
        trials.loc[2, "firstMovement_times"] = trials.loc[2, "stimOn_times"] + 0.01
        trials.loc[3, "firstMovement_times"] = trials.loc[3, "stimOn_times"] + 3.0
        mask = np.array([False] * 5 + [True] * 7)
        result, report = clean_trials(trials, mask, self.config)
        self.assertEqual([step["n_removed"] for step in report["steps"]], [1, 1, 2, 1])
        self.assertEqual([step["n_before"] for step in report["steps"]], [12, 11, 10, 8])
        self.assertEqual([step["n_after"] for step in report["steps"]], [11, 10, 8, 7])
        self.assertEqual(result["trial_number"].tolist(), list(range(5, 12)))
        self.assertEqual(report["n_removed_trials"], 5)
        self.assertEqual(report["ibl_check"]["n_extra_ibl_rejections"], 1)

    def test_each_required_missing_event_is_removed(self):
        """Every required missing field is audited against the official mask."""
        for column in ["stimOn_times", "choice", "feedback_times", "probabilityLeft", "firstMovement_times", "feedbackType"]:
            with self.subTest(column=column):
                trials = self.trials.copy()
                trials.loc[0, column] = np.nan
                mask = self.mask.copy()
                mask.iloc[0] = False
                result, report = clean_trials(trials, mask, self.config)
                self.assertNotIn(0, result["trial_number"].tolist())
                self.assertEqual(report["steps"][1]["n_removed"], 1)

    def test_nullable_missing_choice_is_removed(self):
        """Pandas nullable numeric columns remain valid cleaning inputs."""
        trials = self.trials.astype({"choice": "Float64"})
        trials.loc[0, "choice"] = pd.NA
        self.mask.iloc[0] = False
        result, report = clean_trials(trials, self.mask, self.config)
        self.assertEqual(len(result), 7)
        self.assertEqual(report["steps"][1]["n_removed"], 1)

    def test_inclusive_reaction_time_boundaries(self):
        """Exact boundaries survive; values just outside them do not."""
        trials = self.trials.copy()
        trials["stimOn_times"] = 0.0
        trials["firstMovement_times"] = [0.08, 2.0, 0.079, 2.001, 0.5, 0.5, 0.5, 0.5]
        mask = np.array([True, True, False, False, True, True, True, True])
        result, _ = clean_trials(trials, mask, self.config)
        self.assertEqual(result["trial_number"].tolist(), [0, 1, 4, 5, 6, 7])

    def test_uses_changed_config_limits(self):
        """Reaction-time settings are read from config, not hard-coded."""
        self.config["data"] = {"min_reaction_time": 0.3, "max_reaction_time": 0.8}
        self.trials.loc[0, "firstMovement_times"] = 0.2
        self.mask.iloc[0] = False
        result, report = clean_trials(self.trials, self.mask, self.config)
        self.assertNotIn(0, result["trial_number"].tolist())
        self.assertEqual(report["reaction_time_limits"]["min"], 0.3)

    def test_official_mask_disagreement_fails(self):
        """Never silently remove a trial that IBL kept."""
        self.trials.loc[0, "choice"] = 0
        with self.assertRaisesRegex(ValueError, "IBL mask disagreement"):
            clean_trials(self.trials, self.mask, self.config)

    def test_zero_contrast_trials_are_preserved(self):
        """Zero contrast is valid on either side, and excluded from sign checks."""
        self.trials.loc[6, "contrastLeft"] = 0.0
        self.trials.loc[7, "contrastRight"] = 0.0
        self.trials.loc[6:7, "feedbackType"] = -1
        result, report = clean_trials(self.trials, self.mask, self.config)
        self.assertEqual(len(result), 8)
        self.assertEqual(result["is_zero_contrast"].tolist(), [False] * 6 + [True, True])
        self.assertEqual(result.loc[6:7, "stim_side"].tolist(), [1, -1])
        self.assertEqual(report["choice_direction"]["n_trials_checked"], 6)

    def test_opposite_convention_does_not_flip_labels(self):
        """Detect the opposite convention while retaining the agreed y coding."""
        self.trials["choice"] *= -1
        result, report = clean_trials(self.trials, self.mask, self.config)
        self.assertEqual(report["choice_direction"]["convention"], "opposite")
        self.assertEqual(result["y"].tolist(), [0, 1, 1, 1, 0, 1, 0, 1])

    def test_choice_direction_exact_thresholds_and_mixed_data(self):
        """Accept exactly 99%/1% agreement; reject ambiguous sign evidence."""
        for n_agree, expected in [(99, "same"), (1, "opposite"), (50, None)]:
            with self.subTest(n_agree=n_agree):
                clean = pd.DataFrame({"choice": np.ones(100), "stim_side": np.ones(100), "contrast": np.ones(100), "feedbackType": [1] * n_agree + [-1] * (100 - n_agree)})
                if expected is None:
                    with self.assertRaisesRegex(ValueError, "Inconsistent choice direction"):
                        check_choice_direction(clean)
                else:
                    self.assertEqual(check_choice_direction(clean)["convention"], expected)

    def test_no_visible_stimuli_cannot_pass_direction_check(self):
        """An uncheckable sign convention must not be reported as verified."""
        self.trials["contrastLeft"] = self.trials["contrastLeft"].where(self.trials["contrastLeft"].isna(), 0)
        self.trials["contrastRight"] = self.trials["contrastRight"].where(self.trials["contrastRight"].isna(), 0)
        with self.assertRaisesRegex(ValueError, "no kept nonzero-contrast trials"):
            clean_trials(self.trials, self.mask, self.config)

    def test_invalid_contrast_presence_fails(self):
        """Both filled and both missing are invalid on a kept trial."""
        for values in [(0.5, 0.5), (np.nan, np.nan), (0.0, 0.0)]:
            with self.subTest(values=values):
                trials = self.trials.copy()
                trials.loc[0, ["contrastLeft", "contrastRight"]] = values
                with self.assertRaisesRegex(ValueError, "exactly one filled"):
                    clean_trials(trials, self.mask, self.config)

    def test_invalid_contrast_on_discarded_trial_is_irrelevant(self):
        """Validate contrast structure only on trials retained by IBL."""
        self.trials.loc[0, ["contrastLeft", "contrastRight"]] = np.nan
        self.mask.iloc[0] = False
        result, _ = clean_trials(self.trials, self.mask, self.config)
        self.assertEqual(len(result), 7)

    def test_row_numbers_are_original_positions_not_index_labels(self):
        """Filtering preserves original order and gaps in trial_number."""
        self.trials.index = [100, 105, 200, 201, 300, 400, 500, 600]
        mask = pd.Series([True, False, True, False, True, True, True, True], index=self.trials.index)
        result, _ = clean_trials(self.trials, mask, self.config)
        self.assertEqual(result["trial_number"].tolist(), [0, 2, 4, 5, 6, 7])
        self.assertEqual(result.index.tolist(), list(range(6)))

    def test_inputs_are_unchanged_and_report_is_json_serializable(self):
        """Do not mutate shared inputs; make the audit report easy to save."""
        trials_before = self.trials.copy(deep=True)
        mask_before = self.mask.copy(deep=True)
        _, report = clean_trials(self.trials, self.mask, self.config)
        pd.testing.assert_frame_equal(self.trials, trials_before)
        pd.testing.assert_series_equal(self.mask, mask_before)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False))["n_input_trials"], 8)

    def test_mask_rejects_bad_shape_values_or_alignment(self):
        """Catch masks that would select the wrong rows or coerce bad values."""
        bad_masks = [np.ones(7, dtype=bool), np.ones((8, 1), dtype=bool), [1] * 8, ["False"] * 8, pd.Series([True] * 7 + [pd.NA], dtype="boolean"), pd.Series(True, index=list(reversed(self.trials.index)))]
        for mask in bad_masks:
            with self.subTest(mask_type=type(mask).__name__):
                with self.assertRaises(ValueError):
                    _validate_official_mask(self.trials, mask)

    def test_nullable_boolean_mask_is_supported(self):
        """A Boolean Series without missing entries is a valid official mask."""
        result, _ = clean_trials(self.trials, self.mask.astype("boolean"), self.config)
        self.assertEqual(len(result), 8)

    def test_empty_selection_fails_clearly(self):
        """Avoid producing an unusable empty session."""
        with self.assertRaisesRegex(ValueError, "No trials remain"):
            clean_trials(self.trials, np.zeros(8, dtype=bool), self.config)

    def test_missing_required_column_fails(self):
        """Missing schema fields produce a clear error before cleaning."""
        with self.assertRaisesRegex(ValueError, "Missing required trial columns"):
            clean_trials(self.trials.drop(columns="contrastLeft"), self.mask, self.config)

    def test_duplicate_columns_fail(self):
        """Reject ambiguous duplicate column names."""
        trials = pd.concat([self.trials, self.trials[["choice"]]], axis=1)
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            clean_trials(trials, self.mask, self.config)

    def test_non_dataframe_input_fails(self):
        """The pure cleaning function requires the official table type."""
        with self.assertRaisesRegex(TypeError, "pandas DataFrame"):
            clean_trials(self.trials.to_dict(), self.mask, self.config)

    def test_invalid_retained_values_fail(self):
        """Reject corrupt retained labels, probabilities, times and contrasts."""
        for column, value, message in [("choice", 2, "choice values"), ("feedbackType", 0, "feedbackType values"), ("probabilityLeft", 0.3, "probabilityLeft values"), ("feedback_times", np.inf, "finite numeric"), ("contrastLeft", -0.5, "finite and nonnegative"), ("contrastLeft", np.inf, "finite and nonnegative")]:
            with self.subTest(column=column, value=value):
                trials = self.trials.copy()
                trials.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    clean_trials(trials, self.mask, self.config)

    def test_invalid_reaction_time_limits_fail(self):
        """Invalid configuration must not silently change the selection."""
        for minimum, maximum in [(-1, 2), (3, 2), (np.nan, 2), (0.08, np.inf)]:
            with self.subTest(minimum=minimum, maximum=maximum):
                config = {"data": {"min_reaction_time": minimum, "max_reaction_time": maximum}}
                with self.assertRaisesRegex(ValueError, "Reaction-time limits"):
                    _reaction_time_limits(config)

    def test_load_trials_forwards_exact_official_arguments(self):
        """Check the wrapper contract without downloading or mocking results as real."""
        official_loader = Mock(return_value=(self.trials, self.mask))
        fake_module = types.ModuleType("brainwidemap")
        fake_module.load_trials_and_mask = official_loader
        one = object()
        with patch.dict(sys.modules, {"brainwidemap": fake_module}):
            trials, mask = load_trials(one, "synthetic-session", self.config)
        official_loader.assert_called_once_with(one, "synthetic-session", min_rt=0.08, max_rt=2.0, nan_exclude="default", exclude_nochoice=True)
        self.assertIs(trials, self.trials)
        self.assertIs(mask, self.mask)

    def test_cli_with_session_or_frozen_release_default(self):
        """The smoke-check entry point uses shared settings and prints its audit."""
        for explicit in (True, False):
            with self.subTest(explicit=explicit):
                config = {**self.config, "ibl": {"release_freeze": "test-freeze"}}
                fake_bwm = types.ModuleType("brainwidemap")
                fake_bwm.bwm_query = Mock(return_value=pd.DataFrame({"eid": ["frozen-session"]}))
                fake_settings = types.ModuleType("src.common.settings")
                fake_settings.load_config = Mock(return_value=config)
                fake_settings.connect_ibl = Mock(return_value=object())
                args = ["clean_trials"] + (["--session", "selected-session"] if explicit else [])
                output = io.StringIO()
                with patch.dict(sys.modules, {"brainwidemap": fake_bwm, "src.common.settings": fake_settings}), patch.object(sys, "argv", args), patch("src.data.clean_trials.load_trials", return_value=(self.trials, self.mask)) as loader, contextlib.redirect_stdout(output):
                    main()
                expected = "selected-session" if explicit else "frozen-session"
                self.assertEqual(loader.call_args.args[1], expected)
                self.assertIn(f"Session: {expected}", output.getvalue())
                self.assertIn('"n_kept_trials": 8', output.getvalue())
                if explicit:
                    fake_bwm.bwm_query.assert_not_called()
                else:
                    fake_bwm.bwm_query.assert_called_once_with(freeze="test-freeze")


if __name__ == "__main__":
    unittest.main()
