#!/usr/bin/env python3
"""Run quick or HDF5 dataset validation on haiqin with the frozen environment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.joint_feature_pack.localization_networks.validation.reporting import Report, normalize_expected
from scripts.joint_feature_pack.localization_networks.validation.quick import quick
from scripts.joint_feature_pack.localization_networks.validation.datasets import datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    quick_parser = sub.add_parser("quick", allow_abbrev=False, help="Validate imports, CLIs, loaders and all twelve models")
    quick_parser.add_argument("--device", required=True, help="Explicit cpu/cuda/cuda:N; CUDA never silently falls back")
    quick_parser.add_argument("--report", type=Path, required=True)
    dataset_parser = sub.add_parser("datasets", allow_abbrev=False, help="Validate all generated datasets and cross-route equivalence")
    dataset_parser.add_argument("--root", type=Path, required=True)
    dataset_parser.add_argument("--report", type=Path, required=True)
    dataset_parser.add_argument("--expected-config", type=Path, help="Canonical Python generation JSON, or flat num_elements/freq_hz/min_samples expectations; unknown fields fail")
    args = parser.parse_args()
    report = Report(args.command, args.report.resolve())
    try:
        if args.command == "quick":
            quick(report, args.device)
        else:
            expected = {}
            def read_expected():
                nonlocal expected
                raw = json.loads(args.expected_config.read_text()) if args.expected_config else {}
                expected = normalize_expected(raw)
                return {"format": "canonical Python generation" if "dataset" in expected else "flat expectations", "config": expected}
            if report.check("datasets.expected_config", read_expected):
                datasets(report, args.root.resolve(), expected)
    except Exception as exc:
        report.check("validation.unhandled_error", lambda: (_ for _ in ()).throw(exc))
    report.save()
    print(f"{report.data['status']} report={report.output}", flush=True)
    return 0 if report.data["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
