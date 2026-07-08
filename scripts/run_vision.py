#!/usr/bin/env python3
"""Launch real-time YOLO vision detection on tabletop objects.

Usage:
    python scripts/run_vision.py
    python scripts/run_vision.py --camera-index 2 --conf 0.5
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VISION_SCRIPT = ROOT / "src" / "perception" / "vision_realtime_yolo.py"


def main():
    print("=" * 60)
    print("  Zhian RDK X5 — Real-time YOLO Vision Detection")
    print("=" * 60)
    print()
    print("Prerequisites:")
    print("  1. Camera connected (USB or RGB-D)")
    print("  2. YOLO model weights available")
    print("     Set env: export YOLO_MODEL_PATH=/path/to/best.pt")
    print("     Default: models/weights/best.pt")
    print()

    cmd = [sys.executable, str(VISION_SCRIPT)] + sys.argv[1:]
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
