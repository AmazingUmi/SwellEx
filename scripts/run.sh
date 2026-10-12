#!/bin/sh
# Stdlib dispatcher; local-grnn explicitly selects a local CPU Python.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

# Keep every argument intact, including a config path containing spaces.
case "${1-}" in
    RBD|ELM|SCM) set -- run "$@" ;;
    --config)
        if [ "$#" -ge 3 ]; then
            config_path=$2
            shift 2
            case "${1-}" in
                RBD|ELM|SCM) set -- run "$@" ;;
            esac
            set -- --config "$config_path" "$@"
        fi
        ;;
esac

exec uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py "$@"
