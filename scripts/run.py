#!/usr/bin/env python3
"""Select an experiment pack before dispatching its commands."""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.packs import PACKS, get_pack
from scripts.pack_cli import pack_arguments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="./scripts/run.sh", description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("PACK alone runs its full experiment using the bundled configuration.\n"
                "Examples:\n"
                "  ./scripts/run.sh joint_feature_pack --dry-run\n"
                "  ./scripts/run.sh base_pack RBD complex_CNN --epochs 100\n"
                "  ./scripts/run.sh joint_feature_pack status --run RUN_ID\n"
                "Use PACK --help for available methods and commands."),
    )
    parser.add_argument("pack", metavar="PACK", choices=tuple(PACKS),
                        help="Experiment pack: " + ", ".join(PACKS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Experiment options, METHOD ARCH, or a pack command")
    args = parser.parse_args(argv)
    pack = get_pack(args.pack)
    return import_module(pack.dispatcher).main(pack_arguments(pack, args.arguments))


if __name__ == "__main__":
    raise SystemExit(main())
