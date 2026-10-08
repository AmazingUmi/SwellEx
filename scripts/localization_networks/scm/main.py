"""SCM command entry; run numerical work on haiqin."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.localization_networks.scm.network.cli import main

if __name__ == "__main__":
    main()
