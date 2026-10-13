"""One strict, standard-library-only public command contract for both packs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from scripts.packs import get_pack

PRESETS = {
    "full": "haiqin_full_interleaved.toml",
    "train_before": "haiqin_range_overlap_train_before.toml",
    "train_after": "haiqin_range_overlap_train_after.toml",
}


class PackArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        arguments = list(sys.argv[1:] if args is None else args)
        seen = set()
        for argument in arguments:
            option = argument.partition("=")[0]
            if option.startswith("--") and option not in {"--", "--help"}:
                if option in seen:
                    self.error(f"option {option} may only be supplied once")
                seen.add(option)
        result = super().parse_args(arguments, namespace)
        command = result.command
        allowed = {
            None: {"--preset", "--config", "--method", "--network", "--fusion", "--no-cache"},
            "validate": {"--preset", "--config", "--suite", "--fusion", "--no-cache"},
            "check": {"--config"},
            "status": {"--config", "--run"},
            "fetch": {"--config", "--run"},
            "local-grnn": {"--preset", "--config", "--method", "--python", "--data"},
        }[command]
        invalid = seen - allowed
        if invalid:
            self.error(f"{command or 'experiment'} does not accept {', '.join(sorted(invalid))}")
        if command in {"status", "fetch"} and result.run is None:
            self.error(f"{command} requires --run")
        if command == "local-grnn" and (result.method is None or result.python is None):
            self.error("local-grnn requires --method and --python")
        return result


def make_pack_parser(pack_name: str, default_config: Path,
                     methods: tuple[str, ...], architectures: tuple[str, ...]) -> argparse.ArgumentParser:
    pack = get_pack(pack_name)
    if tuple(methods) != pack.methods:
        raise ValueError(f"{pack_name} methods differ from the pack registry")
    parser = PackArgumentParser(
        prog=f"./scripts/run.sh {pack_name}", allow_abbrev=False,
        description="Start immediately, follow progress, and fetch results. The remote task survives disconnection.",
        epilog="Omit --method or --network to select all. Epochs and seed belong in the complete TOML configuration.",
    )
    commands = ["check", "validate", "status", "fetch"]
    if pack.local_grnn:
        commands.append("local-grnn")
    parser.add_argument("command", nargs="?", choices=commands, help="Maintenance command; omit to run an experiment")
    experiment = parser.add_argument_group("Experiment selection")
    source = experiment.add_mutually_exclusive_group()
    source.add_argument("--preset", choices=tuple(PRESETS), help="Bundled dataset preset (default: full)")
    source.add_argument("--config", type=Path, help="Complete custom TOML; mutually exclusive with --preset")
    experiment.add_argument("--method", choices=methods, help="Feature selection (default: all)")
    experiment.add_argument("--network", choices=architectures, help="Network family (default: all)")
    if pack.fusion_modes:
        experiment.add_argument("--fusion", choices=pack.fusion_modes, help="Override fusion.mode (bundled default: block_rms)")
    maintenance = parser.add_argument_group("Maintenance")
    maintenance.add_argument("--suite", choices=("quick", "smoke", "experiment"), default="smoke", help="validate only (default: smoke)")
    maintenance.add_argument("--no-cache", action="store_true", help="Regenerate feature datasets")
    maintenance.add_argument("--run", metavar="RUN_ID", help="Required for status/fetch")
    if pack.local_grnn:
        maintenance.add_argument("--python", type=Path, help="local-grnn: existing CPU interpreter")
        maintenance.add_argument("--data", type=Path, help="local-grnn: matching dataset to reuse")
    parser.set_defaults(config_dir=default_config.parent)
    return parser


def config_path(args: argparse.Namespace) -> Path:
    return args.config if args.config is not None else args.config_dir / PRESETS[args.preset or "full"]
