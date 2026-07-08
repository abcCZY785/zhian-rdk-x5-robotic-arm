"""Closed-loop V7 joint trajectory tracker for the real arm.

This script follows an exported V6.2/V7 MuJoCo joint trajectory as relative
joint targets. Unlike serial_replay_v7_trajectory.py, it does not blindly send
precomputed deltas. Each control period reads the controller's binary feedback,
computes target - actual error, and sends a limited RL delta correction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

from serial_protocol_v7 import BAUDRATE_DEFAULT, V7SerialClient
from serial_replay_v7_trajectory import drain_input


DEFAULT_TRAJECTORY = Path("urdf_learn/playground/playground/public/trajectories/v7_browser_sync_demo.json")
JOINT_NAMES = ["J0", "J1", "J2", "J3", "J4", "J5"]
DEFAULT_MAX_SOURCE_STEP = (2.0, 2.0, 2.0, 1.0, 0.8, 0.8)
DEFAULT_MAX_CMD_STEP = (0.20, 0.25, 0.25, 0.12, 0.12, 0.12)
DEFAULT_MIN_CMD_STEP = (0.015, 0.020, 0.020, 0.020, 0.020, 0.020)


def append_jsonl(path: Path | None, record: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def fmt(values: Iterable[float]) -> str:
    return ", ".join(f"{name}={value:.3f}" for name, value in zip(JOINT_NAMES, values))


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def parse_six_floats(text: str, option_name: str) -> tuple[float, float, float, float, float, float]:
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) == 1:
        parts = parts * 6
    if len(parts) != 6:
        raise ValueError(f"{option_name} must be one value or six comma-separated values")
    return tuple(parts)  # type: ignore[return-value]


def parse_abs_limits(text: str, option_name: str) -> tuple[float, float, float, float, float, float]:
    return tuple(abs(x) for x in parse_six_floats(text, option_name))  # type: ignore[return-value]


def load_trajectory(path: Path, start: int, count: int) -> list[list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError(f"trajectory has no frames: {path}")
    end = min(len(frames), start + count)
    selected = frames[start:end]
    if len(selected) < 2:
        raise ValueError("need at least 2 frames for relative tracking")

    joints_rad: list[list[float]] = []
    for frame in selected:
        joints = frame.get("joints")
        if not isinstance(joints, list) or len(joints) < 6:
            raise ValueError("frame is missing joints[6]")
        joints_rad.append([float(x) for x in joints[:6]])
    return joints_rad


def source_deltas_deg(joints_rad: list[list[float]], gain: float, limits: tuple[float, ...]) -> list[list[float]]:
    result: list[list[float]] = []
    for prev, cur in zip(joints_rad, joints_rad[1:]):
        raw = [math.degrees(c - p) * gain for p, c in zip(prev, cur)]
        result.append([clamp(v, limit) for v, limit in zip(raw, limits)])
    return result


def interpolate_relative_targets(deltas: list[list[float]], max_target_step: tuple[float, ...]) -> list[list[float]]:
    """Build relative target points with synchronized per-segment interpolation."""
    targets: list[list[float]] = []
    current = [0.0] * 6
    for delta in deltas:
        parts = max(1, max(math.ceil(abs(v) / limit) if limit > 0 else 1 for v, limit in zip(delta, max_target_step)))
        for part in range(1, parts + 1):
            alpha = part / parts
            targets.append([current[i] + delta[i] * alpha for i in range(6)])
        current = [current[i] + delta[i] for i in range(6)]
    return targets


def apply_min_step(command: list[float], error: list[float], min_steps: tuple[float, ...]) -> list[float]:
    result: list[float] = []
    for cmd, err, min_step in zip(command, error, min_steps):
        if abs(cmd) < min_step:
            result.append(0.0)
        else:
            result.append(cmd)
    return result


def vector_add(a: Iterable[float], b: Iterable[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def vector_sub(a: Iterable[float], b: Iterable[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def run_tracker(args: argparse.Namespace) -> int:
    source_limits = parse_abs_limits(args.max_source_step_deg, "--max-source-step-deg")
    target_step_limits = parse_abs_limits(args.target_step_deg, "--target-step-deg")
    cmd_limits = parse_abs_limits(args.max_cmd_deg, "--max-cmd-deg")
    min_cmd_steps = parse_abs_limits(args.min_cmd_deg, "--min-cmd-deg")
    kp = parse_six_floats(args.kp, "--kp")
    kd = parse_six_floats(args.kd, "--kd")
    ff_gain = parse_six_floats(args.ff_gain, "--ff-gain")

    joints_rad = load_trajectory(Path(args.trajectory), args.start, args.frames)
    deltas = source_deltas_deg(joints_rad, args.gain, source_limits)
    targets_rel = interpolate_relative_targets(deltas, target_step_limits)
    if not targets_rel:
        raise ValueError("empty target trajectory")

    log_path = Path(args.log) if args.log else None
    period = 1.0 / args.rate
    planned_total = targets_rel[-1]
    max_target_step = [max(abs(targets_rel[i][j] - (targets_rel[i - 1][j] if i > 0 else 0.0)) for i in range(len(targets_rel))) for j in range(6)]

    print(f"Trajectory: {args.trajectory}")
    print(f"Frames selected: start={args.start}, frames={len(joints_rad)}, source_commands={len(deltas)}, target_points={len(targets_rel)}")
    print(f"Closed-loop rate: {args.rate:.1f} Hz, gain={args.gain:.4f}, kp=[{fmt(kp)}], ff=[{fmt(ff_gain)}]")
    print("Source step limits: " + fmt(source_limits))
    print("Target step limits: " + fmt(target_step_limits))
    print("Command limits:     " + fmt(cmd_limits))
    print("Min command steps:  " + fmt(min_cmd_steps))
    print("Planned total:      " + fmt(planned_total))
    print("Max target step:    " + fmt(max_target_step))

    if not args.confirm:
        print("Dry-run only. Add --confirm to send commands to the real arm.")
        return 0

    ok_count = 0
    miss_count = 0
    last_error = [0.0] * 6
    last_target_rel = [0.0] * 6

    with V7SerialClient(args.port, baudrate=args.baud, timeout=args.timeout) as client:
        try:
            client.flush()
            print("Sending stop before closed-loop tracking ...")
            stop_replies = client.stop()
            for reply in stop_replies:
                print(f"  stop <- {reply.raw}")
            append_jsonl(log_path, {"t": time.time(), "cmd": "stop_before", "replies": [r.raw for r in stop_replies]})
            time.sleep(args.settle)

            before = list(client.read_angle())
            print("Before angles: " + fmt(before))
            append_jsonl(log_path, {"t": time.time(), "phase": "before", "angles_deg": before})

            if args.pre_rl_delay > 0:
                print(f"Waiting {args.pre_rl_delay:.2f}s before binary control loop ...")
                time.sleep(args.pre_rl_delay)
                client.flush()

            actual = before[:]
            for wi in range(args.warmup_frames):
                client.send_rl_delta([0.0] * 6)
                report = client.read_rl_report(timeout=args.report_timeout)
                if report is not None:
                    actual = list(report.angles_deg)
                print(f"  warmup {wi:02d}: {'ok' if report is not None else 'miss'}")
                append_jsonl(log_path, {"t": time.time(), "phase": "warmup", "index": wi, "ok": report is not None, "angles_deg": actual})
                time.sleep(period)

            for i, target_rel in enumerate(targets_rel):
                loop_t = time.monotonic()
                target_abs = vector_add(before, target_rel)
                actual_rel = vector_sub(actual, before)
                error = vector_sub(target_abs, actual)
                ff = [target_rel[j] - last_target_rel[j] for j in range(6)]
                derr = [(error[j] - last_error[j]) / period for j in range(6)]
                raw_cmd = [ff_gain[j] * ff[j] + kp[j] * error[j] + kd[j] * derr[j] for j in range(6)]
                limited_cmd = [clamp(raw_cmd[j], cmd_limits[j]) for j in range(6)]
                cmd = apply_min_step(limited_cmd, error, min_cmd_steps)

                max_abs_error = max(abs(v) for v in error)
                if max_abs_error > args.max_error_deg:
                    print(f"  step {i:04d}: abort, tracking error too large: {max_abs_error:.3f} deg")
                    append_jsonl(
                        log_path,
                        {
                            "t": time.time(),
                            "step": i,
                            "ok": False,
                            "abort": "max_error",
                            "target_rel_deg": target_rel,
                            "target_abs_deg": target_abs,
                            "actual_deg": actual,
                            "error_deg": error,
                            "cmd_deg": cmd,
                        },
                    )
                    break

                try:
                    client.send_rl_delta(cmd)
                    report = client.read_rl_report(timeout=args.report_timeout)
                except Exception as exc:  # noqa: BLE001
                    print(f"  step {i:04d}: serial error: {exc}")
                    append_jsonl(log_path, {"t": time.time(), "step": i, "ok": False, "error": str(exc), "cmd_deg": cmd})
                    break

                if report is None:
                    miss_count += 1
                    print(f"  step {i:04d}: miss, cmd=[{fmt(cmd)}] err=[{fmt(error)}]")
                    append_jsonl(
                        log_path,
                        {
                            "t": time.time(),
                            "step": i,
                            "ok": False,
                            "miss": True,
                            "target_rel_deg": target_rel,
                            "target_abs_deg": target_abs,
                            "actual_rel_deg": actual_rel,
                            "actual_deg": actual,
                            "error_deg": error,
                            "feedforward_deg": ff,
                            "raw_cmd_deg": raw_cmd,
                            "cmd_deg": cmd,
                        },
                    )
                    if miss_count > args.max_misses:
                        print("Too many missed reports; aborting closed-loop tracking.")
                        break
                else:
                    actual = list(report.angles_deg)
                    actual_rel = vector_sub(actual, before)
                    post_error = vector_sub(target_abs, actual)
                    ok_count += 1
                    if args.print_every > 0 and (i % args.print_every == 0 or i == len(targets_rel) - 1):
                        print(f"  step {i:04d}: cmd=[{fmt(cmd)}] actual_rel=[{fmt(actual_rel)}] post_err=[{fmt(post_error)}]")
                    append_jsonl(
                        log_path,
                        {
                            "t": time.time(),
                            "step": i,
                            "ok": True,
                            "target_rel_deg": target_rel,
                            "target_abs_deg": target_abs,
                            "actual_rel_deg": actual_rel,
                            "actual_deg": actual,
                            "error_deg": error,
                            "post_error_deg": post_error,
                            "feedforward_deg": ff,
                            "raw_cmd_deg": raw_cmd,
                            "cmd_deg": cmd,
                            "tcp_pos_mm": report.tcp_pos_mm,
                            "velocities_deg_s": report.velocities_deg_s,
                        },
                    )
                    last_error = post_error

                last_target_rel = target_rel
                sleep_s = period - (time.monotonic() - loop_t)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            time.sleep(args.settle)
            drained = drain_input(client, args.post_drain)
            if drained:
                print(f"Drained {drained} leftover binary bytes before read_angle")
            try:
                after = list(client.read_angle())
                observed_total = vector_sub(after, before)
                final_error = vector_sub(planned_total, observed_total)
                print("After angles:   " + fmt(after))
                print("Observed total: " + fmt(observed_total))
                print("Final error:    " + fmt(final_error))
                append_jsonl(log_path, {"t": time.time(), "phase": "after", "angles_deg": after, "observed_total_deg": observed_total, "final_error_deg": final_error})
            except Exception as exc:  # noqa: BLE001
                print(f"After read_angle failed: {exc}")
                append_jsonl(log_path, {"t": time.time(), "phase": "after", "error": str(exc)})

            print("Sending stop after closed-loop tracking ...")
            stop_replies = client.stop()
            for reply in stop_replies:
                print(f"  stop <- {reply.raw}")
            append_jsonl(log_path, {"t": time.time(), "cmd": "stop_after", "replies": [r.raw for r in stop_replies]})

        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            append_jsonl(log_path, {"t": time.time(), "error": str(exc)})
            try:
                print("Trying emergency stop after error ...")
                for reply in client.stop():
                    print(f"  stop <- {reply.raw}")
            except Exception as stop_exc:  # noqa: BLE001
                print(f"Emergency stop command failed: {stop_exc}")
            return 1

    print(f"Closed-loop tracking finished: ok={ok_count}, missed={miss_count}")
    return 0 if ok_count > 0 and miss_count <= args.max_misses else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V7 closed-loop joint trajectory tracker")
    parser.add_argument("--port", default="COM10", help="Serial port, default COM10")
    parser.add_argument("--baud", type=int, default=BAUDRATE_DEFAULT, help="Baudrate, default 115200")
    parser.add_argument("--trajectory", default=str(DEFAULT_TRAJECTORY), help="Exported browser trajectory JSON")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--frames", type=int, default=10, help="Number of source frames to use")
    parser.add_argument("--rate", type=float, default=10.0, help="Control rate in Hz")
    parser.add_argument("--gain", type=float, default=0.20, help="Scale applied to source trajectory frame-to-frame deltas")
    parser.add_argument("--max-source-step-deg", default=",".join(str(x) for x in DEFAULT_MAX_SOURCE_STEP), help="Clamp source frame deltas before interpolation")
    parser.add_argument("--target-step-deg", default="0.20,0.25,0.25,0.12,0.12,0.12", help="Interpolated target spacing, one value or six comma-separated values")
    parser.add_argument("--max-cmd-deg", default=",".join(str(x) for x in DEFAULT_MAX_CMD_STEP), help="Maximum command delta per control period")
    parser.add_argument("--min-cmd-deg", default=",".join(str(x) for x in DEFAULT_MIN_CMD_STEP), help="Commands below this absolute size are suppressed")
    parser.add_argument("--kp", default="0.35,0.45,0.45,0.35,0.35,0.35", help="Per-joint proportional feedback gain")
    parser.add_argument("--kd", default="0", help="Per-joint derivative feedback gain; default disabled")
    parser.add_argument("--ff-gain", default="1.0", help="Per-joint feedforward gain on target step")
    parser.add_argument("--max-error-deg", type=float, default=5.0, help="Abort if any joint target error exceeds this")
    parser.add_argument("--timeout", type=float, default=0.1, help="Low-level serial timeout")
    parser.add_argument("--report-timeout", type=float, default=0.3, help="Binary report wait timeout")
    parser.add_argument("--response-delay", type=float, default=0.02, help="Seconds to wait after sending a command before reading feedback")
    parser.add_argument("--no-drain-stale-before-command", dest="drain_stale_before_command", action="store_false", help="Do not clear stale binary reports before each command")
    parser.set_defaults(drain_stale_before_command=True)
    parser.add_argument("--settle", type=float, default=0.5, help="Settle delay before/after tracking")
    parser.add_argument("--post-drain", type=float, default=1.0, help="Seconds to drain binary leftovers before final text read_angle")
    parser.add_argument("--max-misses", type=int, default=2, help="Abort after more than this many missed binary reports")
    parser.add_argument("--warmup-frames", type=int, default=2, help="Zero-delta RL frames before tracking")
    parser.add_argument("--pre-rl-delay", type=float, default=0.5, help="Delay after read_angle before binary loop")
    parser.add_argument("--confirm", action="store_true", help="Actually send commands to the real arm")
    parser.add_argument("--print-every", type=int, default=10, help="Print every N steps; 0 disables per-step printing")
    parser.add_argument("--log", help="Optional JSONL log path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.gain <= 0:
        parser.error("--gain must be positive")
    return run_tracker(args)


if __name__ == "__main__":
    sys.exit(main())

