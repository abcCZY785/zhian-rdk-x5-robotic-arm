#!/usr/bin/env python3
"""Launch the HTTP bridge server for the browser-based demo page.

Usage:
    python scripts/run_bridge.py
    python scripts/run_bridge.py --port 8766 --camera-index 2
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRIDGE_SCRIPT = ROOT / "src" / "bridge" / "ai_demo_bridge.py"


def main():
    print("=" * 60)
    print("  Zhian RDK X5 — HTTP Bridge Server")
    print("=" * 60)
    print()
    print("Prerequisites:")
    print("  1. Camera connected")
    print("  2. YOLO model: export YOLO_MODEL_PATH=/path/to/best.pt")
    print("  3. MIMO API key: export MIMO_API_KEY=your_key_here")
    print("  4. Mechanical arm connected via USB-serial")
    print()
    print("Default: http://localhost:8766")
    print()

    cmd = [sys.executable, str(BRIDGE_SCRIPT)] + sys.argv[1:]
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
