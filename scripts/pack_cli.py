"""Shared, standard-library-only command contract for experiment packs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from scripts.packs import Pack, get_pack


class PackArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        arguments = list(sys.argv[1:] if args is None else args)
        # These commands have no repeatable options. Check the whole command,
        # including parent/child --config, before argparse can overwrite a value.
        seen = set()
        for argument in arguments:
            option = argument.partition("=")[0]
            if option.startswith("--") and option not in {"--", "--help"}:
                if option in seen:
                    self.error(f"option {option} may only be supplied once")
                seen.add(option)
        return super().parse_args(arguments, namespace)


def pack_arguments(pack: Pack, arguments: list[str]) -> list[str]:
    """Resolve the implicit experiment or selected model without parsing values.

    Only --config is a global option. Other options follow the explicit command
    or apply to the implicit experiment. Invalid/incomplete input is left for
    the pack parser to reject, never repaired into a different model selection.
    """
    result = list(arguments)
    index = 0
    while index < len(result):
        if result[index] == "--config":
            if index + 1 == len(result) or result[index + 1].startswith("-"):
                return result
            index += 2
        elif result[index].startswith("--config="):
            index += 1
        else:
            break
    remaining = result[index:]
    if remaining and remaining[0] in pack.methods:
        result.insert(index, "run")
    elif not remaining or (remaining[0].startswith("-") and remaining[0] not in {"-h", "--help", "--"}):
        # A suite supplied explicitly takes precedence over the implicit default.
        explicit_suite = any(arg.partition("=")[0] == "--suite" for arg in remaining)
        result[index:index] = ["validate"] if explicit_suite else ["validate", "--suite", "experiment"]
    return result


def make_pack_parser(pack_name: str, default_config: Path,
                     methods: tuple[str, ...], architectures: tuple[str, ...]) -> argparse.ArgumentParser:
    pack = get_pack(pack_name)
    if tuple(methods) != pack.methods:
        raise ValueError(f"{pack_name} methods differ from the pack registry")
    parser = PackArgumentParser(
        prog=f"./scripts/run.sh {pack_name}", allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(f"Run {pack_name} on haiqin. PACK alone runs the full experiment\n"
                     f"({len(methods) * len(architectures)} models) using its configured data interval and epochs."),
        epilog=(f"Default experiment: ./scripts/run.sh {pack_name} [--config PATH] [--dry-run] [--detach]\n"
                f"One model: ./scripts/run.sh {pack_name} METHOD ARCH [options]\n"
                f"Methods: {', '.join(methods)}\nArchitectures: {', '.join(architectures)}\n"
                "Use validate --help for experiment options, or run --help for one model.\n"
                "GRNN uses build/predict and has no epochs."),
    )
    parser.add_argument("--config", type=Path, default=default_config, help="Complete pack TOML; also accepted after the command")
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check", allow_abbrev=False, help="Check haiqin repository, original inputs, and uv")
    validate_parser = sub.add_parser("validate", allow_abbrev=False, help="Run quick, smoke, or full experiment on haiqin")
    validate_parser.add_argument("--suite", choices=["quick", "smoke", "experiment"], default="smoke",
                                 help="Explicit validate defaults to smoke; PACK alone selects experiment")
    run_parser = sub.add_parser("run", allow_abbrev=False, help="Run one METHOD ARCH (the run keyword is optional)")
    run_parser.add_argument("method", metavar="METHOD", choices=methods)
    run_parser.add_argument("architecture", metavar="ARCH", choices=architectures)
    run_parser.add_argument("--epochs", type=int, help="Override neural training epochs; invalid for GRNN")
    run_parser.add_argument("--seed", type=int, help="Override the training/GRNN seed")
    for child in (validate_parser, run_parser):
        if pack.fusion_modes:
            child.add_argument("--fusion", choices=pack.fusion_modes, help="Override fusion.mode")
        child.add_argument("--detach", action="store_true", help="Launch remotely and return the run ID")
        child.add_argument("--dry-run", action="store_true", help="Print the request without SSH or computation")
        child.add_argument("--no-cache", action="store_true", help="Regenerate datasets instead of reusing the verified cache")
    children = [check_parser, validate_parser, run_parser]
    if pack.local_grnn:
        local_parser = sub.add_parser("local-grnn", allow_abbrev=False, help="Run range_overlap GRNN with an explicit local CPU Python")
        local_parser.add_argument("--method", choices=methods, required=True)
        local_parser.add_argument("--python", type=Path, required=True, help="Existing CPU environment Python executable")
        local_parser.add_argument("--data", type=Path, help="Reuse a matching method dataset directory")
        local_parser.add_argument("--dry-run", action="store_true", help="Print the local command without running it")
        children.append(local_parser)
    for name in ("status", "fetch"):
        description = "Inspect an existing run's status" if name == "status" else "Retrieve an existing run's results"
        child = sub.add_parser(name, allow_abbrev=False, help=description)
        child.add_argument("--run", required=True, metavar="RUN_ID")
        children.append(child)
    for child in children:
        child.add_argument("--config", type=Path, default=argparse.SUPPRESS,
                           help="Complete pack TOML; use the launch configuration for status/fetch")
    return parser
