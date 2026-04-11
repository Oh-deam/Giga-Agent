from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
AGENT_ENTRYPOINT = ROOT / "agent" / "run.py"
AGENT_DIR = ROOT / "agent"

if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


if __name__ == "__main__":
    runpy.run_path(str(AGENT_ENTRYPOINT), run_name="__main__")
