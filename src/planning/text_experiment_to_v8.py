#!/usr/bin/env python3
"""Plan a text instruction with Xiaomi MIMO and execute the minimal pick/place path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from lab_assistant.location_resolver import (
    LocationResolutionError,
    choose_detection,
    resolve_place_target,
)
from lab_assistant.mimo_client import MimoClient, MimoClientError
from lab_assistant.task_planner import plan_text_instruction
from lab_assistant.templates import execution_readiness, expand_plan

ROOT = Path(__file__).resolve().parent
# TODO: Set YOLO_PYTHON to the Python interpreter with ultralytics installed.
# Example: export YOLO_PYTHON=/path/to/yolo-venv/bin/python
YOLO_PYTHON = Path(os.getenv("YOLO_PYTHON", "python"))
PY313 = Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python313" / "python.exe"
VISION_LOCATOR = ROOT / "vision_table_yolo_locator.py"
BUILD_TASK = ROOT / "build_pick_place_task_v7.py"
RUN_TASK = ROOT / "run_pick_place_task_v7.py"

DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5-pro-ultraspeed"
# TODO: Set RL_SNAPSHOT_DIR to the trained RL model snapshot directory.
# Example: export RL_SNAPSHOT_DIR=/path/to/snapshots/accepted_v8_1_...
DEFAULT_RL_DIR = os.getenv("RL_SNAPSHOT_DIR", "models/weights/rl_snapshot")
DEFAULT_BASE_CORNERS = "0.12,-0.50;-0.12,-0.50;-0.12,-0.30;0.12,-0.30"
DEFAULT_TASK_NAME = "text_experiment_v8_demo"
TRAJ_ROOT = ROOT / "urdf_learn" / "playground" / "playground" / "public" / "trajectories"


class AssistantExecutionError(RuntimeError):
    """Raised when the minimal assistant cannot execute the task."""


def run(cmd: list[str], *, label: str) -> None:
    print(f"[{label}] RUN: {' '.join(str(x) for x in cmd)}", flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(ROOT), check=True)


def parse_base_corners(text: str) -> list[list[float]]:
    points: list[list[float]] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        values = [float(v.strip()) for v in part.split(",")]
        if len(values) != 2:
            raise ValueError("base corners must be x,y pairs")
        points.append(values)
    if len(points) != 4:
        raise ValueError("base corners must contain four tl,tr,br,bl points")
    return points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction", help="Chinese experiment instruction")
    parser.add_argument("--api-key", default=os.environ.get("MIMO_API_KEY", ""))
    parser.add_argument("--base-url", default=os.environ.get("MIMO_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("MIMO_MODEL", DEFAULT_MODEL))
    parser.add_argument("--camera-index", type=int, default=2)
    parser.add_argument("--vision-json", type=Path, default=None)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--base-corners", default=DEFAULT_BASE_CORNERS)
    parser.add_argument("--rl-dir", default=DEFAULT_RL_DIR)
    parser.add_argument("--segment-frames", type=int, default=140)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--gain", type=float, default=0.9)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--home-reset-speed", type=float, default=40.0)
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("logs/text_experiment_execution_latest.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = None
    if args.api_key:
        client = MimoClient(api_key=args.api_key, base_url=args.base_url, model=args.model)
    try:
        task = plan_text_instruction(client, args.instruction)
        readiness = execution_readiness(task)
        steps = [step.to_dict() for step in expand_plan(task)]
    except (MimoClientError, ValueError) as exc:
        raise SystemExit(f"planner failed: {exc}") from exc


    vision_json = args.vision_json or (ROOT / "logs" / f"{args.task_name}_vision.json")
    vision_jpg = ROOT / "logs" / f"{args.task_name}_vision.jpg"
    request_json = ROOT / "logs" / f"{args.task_name}_request.json"
    task_dir = TRAJ_ROOT / f"task_{args.task_name}"
    task_config = task_dir / "task_config.json"

    try:
        base_corners = parse_base_corners(args.base_corners)
        if args.vision_json is None:
            run(
                [
                    str(YOLO_PYTHON),
                    str(VISION_LOCATOR),
                    "--camera-index",
                    str(args.camera_index),
                    f"--base-corners={args.base_corners}",
                    "--output",
                    str(vision_json),
                    "--preview",
                    str(vision_jpg),
                ],
                label="vision",
            )
        vision_payload = json.loads(vision_json.read_text(encoding="utf-8"))
        missing_marker_roles = vision_payload.get("missing_marker_roles") or []
        if args.vision_json is None and len(missing_marker_roles) >= 2:
            raise AssistantExecutionError(
                "live vision calibration is too incomplete; at least three table markers are required "
                f"for degraded real-arm grasping, missing={missing_marker_roles}"
            )
        if args.vision_json is None and len(missing_marker_roles) == 1:
            print(
                f"[vision] WARNING: degraded 3-marker calibration mode, missing={missing_marker_roles}",
                flush=True,
            )
        detections = vision_payload.get("detections_in_table") or []
        chosen = choose_detection(
            detections,
            object_label=task.object_label,
            source=task.source,
            base_corners=base_corners,
        )
        pick = [float(v) for v in chosen["base_xyz"]]
        pour = resolve_place_target(task.pour_target) if task.intent == "pick_pour_place" else None
        place = resolve_place_target(task.place_target)

        build_cmd = [
            str(PY313),
            str(BUILD_TASK),
            "--name",
            args.task_name,
            f"--pick={pick[0]},{pick[1]},{pick[2]}",
            f"--place={place[0]},{place[1]},{place[2]}",
            "--segment-frames",
            str(args.segment_frames),
            "--sample-every",
            str(args.sample_every),
            "--gain",
            str(args.gain),
            "--rate",
            str(args.rate),
            "--home-reset-speed",
            str(args.home_reset_speed),
            "--home-joints",
            "0,0,0,0,0,0",
            "--rl-dir",
            args.rl_dir,
        ]
        if pour is not None:
            build_cmd.extend([
                f"--pour={pour[0]},{pour[1]},{pour[2]}",
                "--pour-seconds",
                str(task.pour_seconds),
                "--pour-j5-speed",
                "-0.3",
                "--pour-return-j5-speed",
                "0.3",
                "--pour-rotate-seconds",
                "6.0",
            ])

        run(build_cmd, label="build")

        if not args.build_only and args.confirm:
            run(
                [
                    str(PY313),
                    str(RUN_TASK),
                    "--task-config",
                    str(task_config),
                    "--port",
                    args.port,
                    "--confirm",
                ],
                label="real-arm",
            )
    except (subprocess.CalledProcessError, LocationResolutionError, KeyError, AssistantExecutionError) as exc:
        raise SystemExit(f"execution failed: {exc}") from exc

    payload = {
        "instruction": args.instruction,
        "planner": {
            "base_url": args.base_url,
            "model": args.model,
        },
        "task": task.to_dict(),
        "steps": steps,
        "execution": {
            **readiness,
            "executor": "text_experiment_to_v8.py",
            "vision_source": "file" if args.vision_json is not None else "camera",
            "vision_json": str(vision_json),
            "vision_jpg": str(vision_jpg),
            "request_json": str(request_json),
            "task_config": str(task_config),
            "build_only": args.build_only,
            "confirm": args.confirm,
        },
        "resolved": {
            "pick": pick,
            "pour": pour,
            "place": place,
            "selected_detection": chosen,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
