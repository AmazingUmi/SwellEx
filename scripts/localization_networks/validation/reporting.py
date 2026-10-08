"""Structured validation reports and strict expected configuration."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import traceback
from scripts.localization_networks.common.paths import PROJECT_ROOT as ROOT
from scripts.localization_networks.workflow.tasks import METHODS


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def number(value, name, *, minimum=0, integer=False, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"{name} must be a finite number")
    require(value >= minimum and (not positive or value > 0), f"{name} is outside its permitted range")
    require(not integer or value == int(value), f"{name} must be an integer")


def normalize_expected(raw):
    """Use the generation schema as the single authority for configured validation."""
    require(isinstance(raw, dict), "expected config must be an object")
    flat = {"num_elements", "freq_hz", "min_samples"}
    if set(raw) <= flat:
        for key in ("num_elements", "min_samples"):
            if key in raw:
                number(raw[key], key, positive=True, integer=True)
        if "freq_hz" in raw:
            require(isinstance(raw["freq_hz"], list) and raw["freq_hz"], "freq_hz must be a nonempty array")
            for value in raw["freq_hz"]:
                number(value, "freq_hz")
            require(all(a < b for a, b in zip(raw["freq_hz"], raw["freq_hz"][1:])), "freq_hz must be strictly increasing")
        return raw
    from scripts.data_processing.generation_config import normalize_config
    try:
        return normalize_config(raw, root=ROOT)
    except ValueError as exc:
        raise AssertionError(f"invalid canonical generation config: {exc}") from exc


class Report:
    def __init__(self, command, output):
        self.output = output
        self.data = {"schema_version": 1, "command": command, "status": "FAIL",
                     "started_at": datetime.now(timezone.utc).isoformat(),
                     "python": sys.version, "checks": [], "dataset_directories": {m: [] for m in METHODS}}

    def check(self, name, function):
        start = time.monotonic()
        try:
            details = function()
            result = {"name": name, "status": "PASS", "details": details or {}}
        except Exception as exc:
            result = {"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()}
        result["duration_s"] = round(time.monotonic() - start, 3)
        self.data["checks"].append(result)
        print(f"{result['status']} {name}", flush=True)
        self.save(final=False)
        return result["status"] == "PASS"

    def save(self, final=True):
        if final:
            self.data["status"] = "PASS" if self.data["checks"] and all(c["status"] == "PASS" for c in self.data["checks"]) else "FAIL"
            self.data["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_name(self.output.name + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.output)
