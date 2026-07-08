#!/usr/bin/env python3
"""Run a built pick/place task package on the real arm."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON_RUNTIME = Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python313" / "python.exe"


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.buffer.write(text.encode(encoding, errors="backslashreplace") + b"\n")
    sys.stdout.flush()


def resolve_task_path(raw_value: str | None, task_dir: Path) -> str | None:
    if not raw_value:
        return raw_value
    candidate = Path(raw_value)
    if candidate.exists():
        return str(candidate)
    fallback = task_dir / candidate.name
    if fallback.exists():
        return str(fallback)
    return str(candidate)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pick/place task package on the real arm")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.task_config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    task_dir = cfg_path.parent
    for key in ("traj_a", "traj_b", "traj_p", "traj_return", "traj_c", "master"):
        if key in cfg:
            cfg[key] = resolve_task_path(cfg.get(key), task_dir)

    cmd = [
        str(PYTHON_RUNTIME),
        str(ROOT / "serial_pick_and_place_v7.py"),
        "--port", args.port,
        "--traj-a", cfg["traj_a"],
        "--traj-b", cfg["traj_b"],
        "--traj-c", cfg["traj_c"],
        "--frames-a", str(cfg["segment_frames"]),
        "--frames-b", str(cfg["segment_frames"]),
        "--frames-c", str(cfg["segment_frames"]),
        "--rate", str(cfg["rate"]),
        "--gain", str(cfg["gain"]),
        "--max-step-deg", "0.6,0.6,0.6,0.4,0.4,0.4",
        "--log", str(cfg_path.parent / "pick_place_run.jsonl"),
    ]
    gain_segments = cfg.get("gain_segments") or {}
    if gain_segments.get("a") is not None:
        cmd.extend(["--gain-a", str(gain_segments["a"])])
    if gain_segments.get("b") is not None:
        cmd.extend(["--gain-b", str(gain_segments["b"])])
    if gain_segments.get("p") is not None:
        cmd.extend(["--gain-p", str(gain_segments["p"])])
    if gain_segments.get("return") is not None:
        cmd.extend(["--gain-return", str(gain_segments["return"])])
    if gain_segments.get("c") is not None:
        cmd.extend(["--gain-c", str(gain_segments["c"])])
    closed_loop = cfg.get("closed_loop") or {}
    if closed_loop.get("segments"):
        cmd.extend(["--closed-loop-segments", ",".join(str(x) for x in closed_loop["segments"])])
    if closed_loop.get("target_step_deg") is not None:
        cmd.extend(["--cl-target-step-deg", str(closed_loop["target_step_deg"])])
    if closed_loop.get("max_cmd_deg") is not None:
        cmd.extend(["--cl-max-cmd-deg", str(closed_loop["max_cmd_deg"])])
    if closed_loop.get("min_cmd_deg") is not None:
        cmd.extend(["--cl-min-cmd-deg", str(closed_loop["min_cmd_deg"])])
    if closed_loop.get("kp") is not None:
        cmd.extend(["--cl-kp", str(closed_loop["kp"])])
    if closed_loop.get("ff_gain") is not None:
        cmd.extend(["--cl-ff-gain", str(closed_loop["ff_gain"])])
    if closed_loop.get("max_error_deg") is not None:
        cmd.extend(["--cl-max-error-deg", str(closed_loop["max_error_deg"])])
    if cfg.get("traj_p"):
        cmd.extend(["--traj-p", cfg["traj_p"], "--frames-p", str(cfg["segment_frames"])])
    if cfg.get("traj_return"):
        cmd.extend(["--traj-return", cfg["traj_return"], "--frames-return", str(cfg["segment_frames"])])
    if cfg.get("release_after_return"):
        cmd.append("--release-after-return")
    if cfg.get("home_reset_speed") is not None:
        cmd.extend(["--home-reset-speed", str(cfg["home_reset_speed"])])
    if cfg.get("pour", {}).get("enabled"):
        pour_json = cfg_path.parent / "pour_config.json"
        pour_json.write_text(json.dumps(cfg["pour"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cmd.extend(["--pour-json", str(pour_json)])
    if cfg.get("precision_demo", {}).get("enabled"):
        precision_json = cfg_path.parent / "precision_demo_config.json"
        precision_json.write_text(json.dumps(cfg["precision_demo"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cmd.extend(["--precision-demo-json", str(precision_json)])
    if cfg.get("post_place_pour", {}).get("enabled"):
        post_place_pour_json = cfg_path.parent / "post_place_pour_config.json"
        post_place_pour_json.write_text(json.dumps(cfg["post_place_pour"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cmd.extend(["--post-place-pour-json", str(post_place_pour_json)])
    if args.confirm:
        cmd.append("--confirm")

    safe_print("RUN: " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
