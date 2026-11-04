"""Joint neural CLI: --method METHOD train|predict [shared options]."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.network_training.joint_feature_pack.common.cli import main as neural_main
from scripts.network_training.joint_feature_pack.joint.bindings import make_bindings
from scripts.network_training.joint_feature_pack.joint.contract import METHODS


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--method", required=True, choices=METHODS)
    selected, remaining = parser.parse_known_args(argv)
    neural_main(make_bindings(selected.method), remaining)


if __name__ == "__main__":
    main()
