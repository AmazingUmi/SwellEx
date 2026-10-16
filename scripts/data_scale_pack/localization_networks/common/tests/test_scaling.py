"""Training-count policy contracts (standard library only)."""

import math
import unittest

from scripts.data_scale_pack.localization_networks.common.scaling import (
    DEFAULT_SCALING_POLICY,
    PROFILE_REPEATS,
    STAGE_STRIDES,
    parameter_budget,
    parameter_count,
    resolve_capacity,
    validate_scaling_policy,
)


class ScalingPolicyTests(unittest.TestCase):
    def test_exact_small_width_counts_include_normalization_and_head(self):
        # Counts independently expanded for one base channel.
        expected = {"small": 258, "medium": 340, "large": 671, "xlarge": 1061, "baseline": 566}
        for profile, count in expected.items():
            self.assertEqual(parameter_count(profile, 1), count)

    def test_reference_count_selects_medium_and_extremes_clip(self):
        for n, expected in ((1, "small"), (600, "medium"), (10**100, "xlarge")):
            config = resolve_capacity("auto", train_count=n, base_channels=16, scaling_policy=DEFAULT_SCALING_POLICY)
            self.assertEqual(config["selected_profile"], expected)
            self.assertEqual(config["stage_strides"], STAGE_STRIDES)
        self.assertEqual(parameter_budget(600, 16, DEFAULT_SCALING_POLICY), parameter_count("medium", 16))

    def test_capacity_is_monotonic_and_maximal_within_budget(self):
        previous = 0
        for n in (1, 100, 300, 599, 600, 900, 1800, 4000, 8000, 100000):
            saved = resolve_capacity("auto", train_count=n, base_channels=4, scaling_policy=DEFAULT_SCALING_POLICY)
            count, budget = saved["parameter_count"], saved["parameter_budget"]
            self.assertGreaterEqual(count, previous)
            self.assertLessEqual(count, budget)
            self.assertFalse(any(count < parameter_count(p, 4) <= budget for p in PROFILE_REPEATS))
            previous = count

    def test_fixed_controls_do_not_follow_the_auto_budget(self):
        for profile in (*PROFILE_REPEATS, "baseline"):
            low = resolve_capacity(profile, train_count=1, base_channels=4, scaling_policy=DEFAULT_SCALING_POLICY)
            high = resolve_capacity(profile, train_count=100000, base_channels=4, scaling_policy=DEFAULT_SCALING_POLICY)
            self.assertEqual(low["selection_mode"], "fixed")
            self.assertEqual(low["selected_profile"], profile)
            self.assertEqual(low["parameter_count"], high["parameter_count"])
            self.assertNotEqual(low["parameter_budget"], high["parameter_budget"])

    def test_reject_invalid_count_width_profile_and_policy(self):
        for invalid in (None, 0, -1, True, 1.5, "600"):
            with self.subTest(count=invalid), self.assertRaises(ValueError):
                resolve_capacity("auto", train_count=invalid, base_channels=4, scaling_policy=DEFAULT_SCALING_POLICY)
        for invalid in (None, 0, -1, True, 1.5):
            with self.subTest(width=invalid), self.assertRaises(ValueError):
                parameter_count("small", invalid)
        for invalid in (
            None, {}, {"reference_samples": 600}, {"reference_samples": True, "exponent": 0.5},
            {"reference_samples": 600, "exponent": 0}, {"reference_samples": 600, "exponent": math.nan},
            {"reference_samples": 600, "exponent": math.inf}, {"reference_samples": 600, "exponent": True},
            {"reference_samples": 600, "exponent": 0.5, "legacy": 1},
        ):
            with self.subTest(policy=invalid), self.assertRaises(ValueError):
                validate_scaling_policy(invalid)
        with self.assertRaises(ValueError):
            resolve_capacity("unknown", train_count=600, base_channels=4, scaling_policy=DEFAULT_SCALING_POLICY)


if __name__ == "__main__":
    unittest.main()
