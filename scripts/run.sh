#!/bin/sh
# Thin launcher: Python owns defaults, validation, help, and pack dispatch.
set -eu

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required to run SwellEx; install uv and add it to PATH." >&2
    exit 127
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

# Keep every argument intact, including pack-specific config paths with spaces.
exec uv run --no-config --no-project --python python3 python -B scripts/run.py "$@"
