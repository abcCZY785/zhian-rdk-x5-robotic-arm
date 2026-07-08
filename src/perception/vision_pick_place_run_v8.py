#!/usr/bin/env python3
"""One-shot vision -> V8 task build -> optional real-arm replay.

Pipeline:
1. capture one frame and detect tabletop + object
2. map pick point into the V8 training tabletop base coordinates
3. build a V7/V8 task package
4. optionally execute it on the real arm
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# TODO: Set YOLO_PYTHON to the Python interpreter with ultralytics installed.
# Example: export YOLO_PYTHON=/path/to/yolo-venv/bin/python
YOLO_PYTHON = Path(os.getenv("YOLO_PYTHON", "python"))
PY313 = Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python313" / "python.exe"
VISION_SCRIPT = ROOT / "vision_table_yolo_locator.py"
BRIDGE_SCRIPT = ROOT / "vision_to_pick_place_task.py"
RUN_TASK_SCRIPT = ROOT / "run_pick_place_task_v7.py"
TRAJ_ROOT = ROOT / "urdf_learn" / "playground" / "playground" / "public" / "trajectories"
# TODO: Set RL_SNAPSHOT_DIR to the trained RL model snapshot directory.
# Example: export RL_SNAPSHOT_DIR=/path/to/snapshots/accepted_v8_1_...
DEFAULT_RL_DIR = os.getenv("RL_SNAPSHOT_DIR", "models/weights/rl_snapshot")
DEFAULT_BASE_CORNERS = "0.12,-0.50;-0.12,-0.50;-0.12,-0.30;0.12,-0.30"
DEFAULT_PLACE = "-0.348,-0.386,0.232"


def run_cmd(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="vision_pick_place_v8_auto")
    parser.add_argument("--camera-index", type=int, default=2)
    parser.add_argument("--place", default=DEFAULT_PLACE, help="Place target x,y,z")
    parser.add_argument("--base-corners", default=DEFAULT_BASE_CORNERS)
    parser.add_argument("--target-label", default=None, help="Optional class filter, for example beaker")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--rl-dir", default=DEFAULT_RL_DIR)
    parser.add_argument("--segment-frames", type=int, default=140)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--home-joints", default="0,0,0,0,0,0")
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--build-only", action="store_true", help="Stop after task package generation")
    parser.add_argument("--confirm", action="store_true", help="Actually execute on the real arm")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    vision_json = ROOT / "logs" / f"{args.task_name}_vision.json"
    vision_jpg = ROOT / "logs" / f"{args.task_name}_vision.jpg"
    request_json = ROOT / "logs" / f"{args.task_name}_request.json"
    task_dir = TRAJ_ROOT / f"task_{args.task_name}"
    task_config = task_dir / "task_config.json"

    vision_cmd = [
        str(YOLO_PYTHON),
        str(VISION_SCRIPT),
        "--camera-index", str(args.camera_index),
        "--output", str(vision_json),
        "--preview", str(vision_jpg),
    ]
    run_cmd(vision_cmd)

    bridge_cmd = [
        str(YOLO_PYTHON),
        str(BRIDGE_SCRIPT),
        "--vision-json", str(vision_json),
        f"--base-corners={args.base_corners}",
        f"--place={args.place}",
        "--task-name", args.task_name,
        "--segment-frames", str(args.segment_frames),
        "--sample-every", str(args.sample_every),
        "--gain", str(args.gain),
        "--rate", str(args.rate),
        "--home-joints", args.home_joints,
        "--rl-dir", args.rl_dir,
        "--output", str(request_json),
        "--run-build",
    ]
    if args.target_label:
        bridge_cmd.extend(["--target-label", args.target_label])
    if args.target_index:
        bridge_cmd.extend(["--target-index", str(args.target_index)])
    run_cmd(bridge_cmd)

    request = json.loads(request_json.read_text(encoding="utf-8"))
    print("RESULT: pick", request["pick"], "place", request["place"])
    print("RESULT: task_config", task_config)

    if args.build_only:
        return 0

    run_cmd([
        str(PY313),
        str(RUN_TASK_SCRIPT),
        "--task-config", str(task_config),
        "--port", args.port,
        *(["--confirm"] if args.confirm else []),
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
