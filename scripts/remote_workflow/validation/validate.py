#!/usr/bin/env python3
"""Run the quick runtime/CLI/loader/model validation on haiqin with the frozen environment.

Generated-dataset and numerical validation live in `scripts/data_processing/validate.py`.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_processing.reporting import Report
from scripts.remote_workflow.validation.quick import quick


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    quick_parser = sub.add_parser("quick", allow_abbrev=False, help="Validate imports, CLIs, loaders and all current models")
    quick_parser.add_argument("--device", required=True, help="Explicit cpu/cuda/cuda:N; CUDA never silently falls back")
    quick_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = Report(args.command, args.report.resolve())
    try:
        quick(report, args.device)
    except Exception as exc:
        report.check("validation.unhandled_error", lambda: (_ for _ in ()).throw(exc))
    report.save()
    print(f"{report.data['status']} report={report.output}", flush=True)
    return 0 if report.data["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
