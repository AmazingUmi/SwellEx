"""Auditable validation-based controller; thresholds remain uncalibrated."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ControllerPolicy:
    min_relative_gain: float = 0.01
    max_old_relative_regression: float = 0.05
    novelty_threshold: float = 1.0
    balance_loss_weight: float = 0.01

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.min_relative_gain > 1:
            raise ValueError("min_relative_gain must not exceed one")


def assess_candidate(policy, current_before, current_after, old_before=None, old_after=None):
    before, after = current_before["rmse_km"], current_after["rmse_km"]
    gain = (before - after) / max(before, 1e-12)
    improved = math.isfinite(after) and after < before and gain >= policy.min_relative_gain
    historic_available = old_before is not None and old_after is not None
    old_regression = None
    old_ok = True
    if historic_available:
        old_regression = (old_after["rmse_km"] - old_before["rmse_km"]) / max(old_before["rmse_km"], 1e-12)
        old_ok = math.isfinite(old_regression) and old_regression <= policy.max_old_relative_regression
    return {"accepted": improved and old_ok, "relative_current_gain": gain,
            "relative_old_regression": old_regression,
            "current_gain_status": "PASS" if improved else "FAIL",
            "historic_guard_status": ("PASS" if old_ok else "FAIL") if historic_available else "NOT_RUN",
            "reason": "qualified" if improved and old_ok else ("insufficient_current_gain" if not improved else "historic_regression")}


def propose_add(policy, novelty, adapt_assessment, current_experts, maximum):
    if current_experts >= maximum:
        return False, "expert_budget_exhausted"
    if novelty >= policy.novelty_threshold:
        return True, "novelty_above_threshold"
    if not adapt_assessment["accepted"]:
        return True, "adapt_candidate_did_not_qualify"
    return False, "low_novelty_and_adapt_candidate_qualified"


def select_candidate(policy, adapted, added=None):
    """A qualified expansion must also justify its cost over a qualified existing-expert adaptation."""
    adapt_ok = adapted["assessment"]["accepted"]
    added_ok = added is not None and added["assessment"]["accepted"]
    if not added_ok:
        return ("ADAPT_EXISTING", "adapt_candidate_qualified") if adapt_ok else ("REUSE", "no_candidate_qualified")
    if adapt_ok:
        reference = adapted["current"]["rmse_km"]
        improvement = (reference - added["current"]["rmse_km"]) / max(reference, 1e-12)
        if added["current"]["rmse_km"] >= reference or improvement < policy.min_relative_gain:
            return "ADAPT_EXISTING", "expansion_gain_over_adaptation_below_threshold"
    return "ADD_EXPERT", "expansion_qualified"
