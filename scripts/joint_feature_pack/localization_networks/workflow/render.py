"""Internal, bounded worker rendering; public plotting remains PASS-only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from scripts.joint_feature_pack.visualization.plot_results import render_run
    result = render_run(args.run_dir, args.output_dir, require_pass=False)
    if result.get("status") != "PASS":
        raise ValueError("Rendering did not produce a PASS manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
