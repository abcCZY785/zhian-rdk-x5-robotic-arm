#!/usr/bin/env python3
"""Launch text-instruction demo: MIMO LLM → vision → pick-place task.

Usage:
    python scripts/run_demo.py --instruction "把左边的烧杯放到置物台上层"
    python scripts/run_demo.py --instruction "把棕色瓶子的废液倒掉" --camera-index 2
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = ROOT / "src" / "planning" / "text_experiment_to_v8.py"


def main():
    print("=" * 60)
    print("  Zhian RDK X5 — Text Instruction Demo")
    print("=" * 60)
    print()
    print("Prerequisites:")
    print("  1. Camera connected and ArUco markers on table corners")
    print("  2. YOLO model: export YOLO_MODEL_PATH=/path/to/best.pt")
    print("  3. MIMO API key: export MIMO_API_KEY=your_key_here")
    print("  4. Mechanical arm connected via USB-serial")
    print()

    if "--instruction" not in " ".join(sys.argv):
        print("Error: --instruction is required.")
        print()
        print("Example:")
        print('  python scripts/run_demo.py --instruction "把左边的烧杯放到置物台上层"')
        sys.exit(1)

    cmd = [sys.executable, str(DEMO_SCRIPT)] + sys.argv[1:]
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
