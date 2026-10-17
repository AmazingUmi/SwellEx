"""One strict, standard-library-only public command contract for all packs."""
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
MOE_LEARN_PRESETS = {"demo": "haiqin_demo.toml"}


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
            "study": {"--plan", "--study-dir"},
            "budget-study": {"--plan", "--study-dir", "--dry-run"},
        }[command]
        if command == "study" and self.pack_name == "MoE_data_scale":
            allowed = allowed | {"--dry-run"}
        if self.pack_name == "MoE_learn" and command in {None, "validate"}:
            allowed = allowed | {"--dry-run"}
        invalid = seen - allowed
        if invalid:
            self.error(f"{command or 'experiment'} does not accept {', '.join(sorted(invalid))}")
        if command in {"status", "fetch"} and result.run is None:
            self.error(f"{command} requires --run")
        if command == "local-grnn" and (result.method is None or result.python is None):
            self.error("local-grnn requires --method and --python")
        if command in {"study", "budget-study"} and result.plan is None:
            self.error(f"{command} requires --plan")
        return result


def make_pack_parser(pack_name: str, default_config: Path,
                     methods: tuple[str, ...], architectures: tuple[str, ...]) -> argparse.ArgumentParser:
    pack = get_pack(pack_name)
    if tuple(methods) != pack.methods:
        raise ValueError(f"{pack_name} methods differ from the pack registry")
    if tuple(architectures) != pack.networks:
        raise ValueError(f"{pack_name} networks differ from the pack registry")
    parser = PackArgumentParser(
        prog=f"./scripts/run.sh {pack_name}", allow_abbrev=False,
        description="Start immediately, follow progress, and fetch results. The remote task survives disconnection.",
        epilog="Omit --method or --network to select all. Epochs and seed belong in the complete TOML configuration.",
    )
    parser.pack_name = pack_name
    commands = ["check", "validate", "status", "fetch"]
    if pack.local_grnn:
        commands.append("local-grnn")
    if pack_name in {"data_scale_pack", "MoE_data_scale"}:
        commands.append("study")
    if pack_name == "data_scale_pack":
        commands.append("budget-study")
    parser.add_argument("command", nargs="?", choices=commands, help="Maintenance command; omit to run an experiment")
    experiment = parser.add_argument_group("Experiment selection")
    source = experiment.add_mutually_exclusive_group()
    presets = MOE_LEARN_PRESETS if pack_name == "MoE_learn" else PRESETS
    default_preset = "demo" if pack_name == "MoE_learn" else "full"
    source.add_argument("--preset", choices=tuple(presets), help=f"Bundled dataset preset (default: {default_preset})")
    source.add_argument("--config", type=Path, help="Complete custom TOML; mutually exclusive with --preset")
    experiment.add_argument("--method", choices=methods, help="Feature selection (default: all)")
    experiment.add_argument("--network", choices=architectures, help="Network family (default: all)")
    if pack.fusion_modes:
        experiment.add_argument("--fusion", choices=pack.fusion_modes, help="Override fusion.mode (bundled default: block_rms)")
    maintenance = parser.add_argument_group("Maintenance")
    maintenance.add_argument("--suite", choices=("quick", "smoke", "experiment"), default="smoke", help="validate only (default: smoke)")
    maintenance.add_argument("--no-cache", action="store_true", help="Regenerate feature datasets")
    maintenance.add_argument("--run", metavar="RUN_ID", help=(
        "Required for status/fetch; matrix packs also accept the study ID printed at launch"
        if pack_name in {"data_scale_pack", "MoE_data_scale"} else "Required for status/fetch"))
    if pack.local_grnn:
        maintenance.add_argument("--python", type=Path, help="local-grnn: existing CPU interpreter")
        maintenance.add_argument("--data", type=Path, help="local-grnn: matching dataset to reuse")
    if pack_name in {"data_scale_pack", "MoE_data_scale"}:
        matrix_commands = "study" if pack_name == "MoE_data_scale" else "study/budget-study"
        dry_command = "study" if pack_name == "MoE_data_scale" else "budget-study"
        maintenance.add_argument("--plan", type=Path, help=f"{matrix_commands}: complete frozen JSON matrix plan; launch its coordinator on haiqin and return")
        maintenance.add_argument("--study-dir", type=Path, help=f"{matrix_commands}: local output/data directory naming the remote study; import an existing frozen ledger or resume that study")
        maintenance.add_argument("--dry-run", action="store_true", help=f"{dry_command}: validate and print the matrix without SSH or training")
    if pack_name == "MoE_learn":
        maintenance.add_argument("--dry-run", action="store_true", help="Validate and print the demo plan without SSH or training")
    parser.set_defaults(config_dir=default_config.parent, preset_files=presets, default_preset=default_preset)
    return parser


def config_path(args: argparse.Namespace) -> Path:
    presets = getattr(args, "preset_files", PRESETS)
    default = getattr(args, "default_preset", "full")
    return args.config if args.config is not None else args.config_dir / presets[args.preset or default]
