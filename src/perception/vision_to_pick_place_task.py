#!/usr/bin/env python3
"""Bridge tabletop vision output into a V7/V8 pick-place task request.

This script consumes the JSON produced by vision_table_yolo_locator.py,
attaches a calibrated tabletop->base mapping, selects one target, and emits a
compact task request that can be passed into build_pick_place_task_v7.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
BUILD_SCRIPT = ROOT / "build_pick_place_task_v7.py"
PYTHON_RUNTIME = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "python" / "python.exe"
DEFAULT_VISION_JSON = ROOT / "logs" / "vision_table_yolo_latest.json"
DEFAULT_OUTPUT = ROOT / "logs" / "vision_pick_place_request_latest.json"

CLASS_GRASP_Z = {
    "beaker": 0.160,
    "brown_bottle": 0.180,
    "transparent_bottle": 0.180,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-json", type=Path, default=DEFAULT_VISION_JSON)
    parser.add_argument(
        "--base-corners",
        required=True,
        help="Table corners in robot base XY, ordered tl,tr,br,bl. Example: 0,-0.55;0.24,-0.55;0.24,-0.31;0,-0.31",
    )
    parser.add_argument("--place", required=True, help="Place target x,y,z")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-name", default="vision_pick_place")
    parser.add_argument("--segment-frames", type=int, default=140)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--home-joints", default="0,0,0,0,0,0")
    parser.add_argument("--rl-dir", default=None)
    parser.add_argument("--target-label", default=None, help="Optional label filter, for example beaker")
    parser.add_argument("--target-index", type=int, default=0, help="0-based index after filtering and sorting by confidence")
    parser.add_argument("--run-build", action="store_true", help="Invoke build_pick_place_task_v7.py after writing the task request")
    return parser


def parse_xy_triplets(text: str, dims: int) -> list[list[float]]:
    points: list[list[float]] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        values = [float(v.strip()) for v in part.split(",")]
        if len(values) != dims:
            raise ValueError(f"expected {dims} values per point")
        points.append(values)
    return points


def parse_xyz(text: str) -> list[float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(values) != 3:
        raise ValueError("point must be x,y,z")
    return values


def project_norm_to_base(norm_xy: list[float], base_corners: list[list[float]]) -> list[float]:
    tl, tr, br, bl = base_corners
    u, v = float(norm_xy[0]), float(norm_xy[1])

    top_x = tl[0] + (tr[0] - tl[0]) * u
    top_y = tl[1] + (tr[1] - tl[1]) * u
    bot_x = bl[0] + (br[0] - bl[0]) * u
    bot_y = bl[1] + (br[1] - bl[1]) * u

    x = top_x + (bot_x - top_x) * v
    y = top_y + (bot_y - top_y) * v
    return [round(x, 6), round(y, 6)]


def choose_detection(detections: list[dict[str, Any]], target_label: str | None, target_index: int) -> dict[str, Any]:
    filtered = detections
    if target_label:
        filtered = [det for det in detections if str(det.get("label")) == target_label]
    filtered = sorted(filtered, key=lambda det: float(det.get("confidence", 0.0)), reverse=True)
    if not filtered:
        raise RuntimeError("no detections matched the requested filter")
    if target_index < 0 or target_index >= len(filtered):
        raise RuntimeError(f"target-index {target_index} out of range for {len(filtered)} detections")
    return filtered[target_index]


def build_cmd(args: argparse.Namespace, pick: list[float], place: list[float]) -> list[str]:
    cmd = [
        str(PYTHON_RUNTIME),
        str(BUILD_SCRIPT),
        "--name", args.task_name,
        f"--pick={pick[0]},{pick[1]},{pick[2]}",
        f"--place={place[0]},{place[1]},{place[2]}",
        "--segment-frames", str(args.segment_frames),
        "--sample-every", str(args.sample_every),
        "--gain", str(args.gain),
        "--rate", str(args.rate),
        "--home-joints", args.home_joints,
    ]
    if args.rl_dir:
        cmd.extend(["--rl-dir", args.rl_dir])
    return cmd


def main() -> int:
    args = build_parser().parse_args()
    base_corners = parse_xy_triplets(args.base_corners, 2)
    if len(base_corners) != 4:
        raise RuntimeError("base-corners must contain four xy points ordered tl,tr,br,bl")
    place = parse_xyz(args.place)

    payload = json.loads(args.vision_json.read_text(encoding="utf-8"))
    detections = payload.get("detections_in_table") or []
    if not detections:
        raise RuntimeError("vision payload has no detections_in_table")

    selected = choose_detection(detections, args.target_label, args.target_index)
    if "table_xy_norm" not in selected:
        raise RuntimeError("selected detection has no table_xy_norm")

    base_xy = project_norm_to_base(selected["table_xy_norm"], base_corners)
    grasp_z = CLASS_GRASP_Z.get(str(selected.get("label")), 0.160)
    pick = [base_xy[0], base_xy[1], round(grasp_z, 6)]

    task_request = {
        "task_name": args.task_name,
        "camera_id": payload.get("camera_id"),
        "aruco_dict": payload.get("aruco_dict"),
        "table_size_m": payload.get("table_size_m"),
        "table_uv_tl_tr_br_bl": payload.get("table_uv_tl_tr_br_bl"),
        "base_corners_tl_tr_br_bl": base_corners,
        "selected_target": {
            "label": selected.get("label"),
            "confidence": selected.get("confidence"),
            "bbox_xyxy": selected.get("bbox_xyxy"),
            "center_uv": selected.get("center_uv"),
            "pick_uv": selected.get("pick_uv") or selected.get("center_uv"),
            "table_xy_norm": selected.get("table_xy_norm"),
            "table_xy_m": selected.get("table_xy_m"),
            "base_xyz": pick,
        },
        "pick": pick,
        "place": place,
        "build": {
            "segment_frames": args.segment_frames,
            "sample_every": args.sample_every,
            "gain": args.gain,
            "rate": args.rate,
            "home_joints": args.home_joints,
            "rl_dir": args.rl_dir,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(task_request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(task_request, ensure_ascii=False, indent=2))

    if args.run_build:
        cmd = build_cmd(args, pick, place)
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
