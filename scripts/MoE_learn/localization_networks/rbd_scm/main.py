"""Remote numerical demo entry; public launches go through scripts/run.sh."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.MoE_learn.localization_networks.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
