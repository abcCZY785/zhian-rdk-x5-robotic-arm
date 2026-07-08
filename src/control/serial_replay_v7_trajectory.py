"""Stage-3 low-speed replay of exported V7/V6.2 joint trajectory.

The exported browser trajectory stores MuJoCo qpos[:6] in radians. For first
real-arm replay we do NOT jump to the absolute simulated pose. Instead, we replay
a scaled relative version of the trajectory increments from the arm's current
pose, with strict per-step degree clamps.
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


DEFAULT_TRAJECTORY = Path("urdf_learn/playground/playground/public/trajectories/v7_browser_sync_demo.json")
JOINT_NAMES = ["J0", "J1", "J2", "J3", "J4", "J5"]
DEFAULT_MAX_STEP = (0.2, 0.2, 0.2, 0.2, 0.1, 0.2)


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


def parse_limits(text: str) -> tuple[float, float, float, float, float, float]:
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) == 1:
        parts = parts * 6
    if len(parts) != 6:
        raise ValueError("--max-step-deg must be one value or six comma-separated values")
    return tuple(abs(x) for x in parts)  # type: ignore[return-value]


def load_trajectory(path: Path, start: int, count: int) -> list[list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError(f"trajectory has no frames: {path}")
    end = min(len(frames), start + count)
    selected = frames[start:end]
    if len(selected) < 2:
        raise ValueError("need at least 2 frames for relative replay")
    joints_rad = []
    for frame in selected:
        joints = frame.get("joints")
        if not isinstance(joints, list) or len(joints) < 6:
            raise ValueError("frame is missing joints[6]")
        joints_rad.append([float(x) for x in joints[:6]])
    return joints_rad


def trajectory_deltas_deg(joints_rad: list[list[float]], gain: float, limits: tuple[float, ...]) -> list[list[float]]:
    result: list[list[float]] = []
    for prev, cur in zip(joints_rad, joints_rad[1:]):
        raw = [math.degrees(c - p) * gain for p, c in zip(prev, cur)]
        result.append([clamp(v, limit) for v, limit in zip(raw, limits)])
    return result


def subdivide_deltas(deltas: list[list[float]], substep_limits: tuple[float, ...]) -> list[list[float]]:
    if not deltas:
        return []
    result: list[list[float]] = []
    for delta in deltas:
        parts = max(1, max(math.ceil(abs(v) / limit) if limit > 0 else 1 for v, limit in zip(delta, substep_limits)))
        for _ in range(parts):
            result.append([v / parts for v in delta])
    return result



def drain_input(client: V7SerialClient, seconds: float = 0.8) -> int:
    """Drain binary report leftovers before switching back to text commands."""
    old_timeout = client.ser.timeout
    client.ser.timeout = 0.05
    total = 0
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            chunk = client.ser.read(512)
            total += len(chunk)
    finally:
        client.ser.timeout = old_timeout
    return total
def run_replay(args: argparse.Namespace) -> int:
    limits = parse_limits(args.max_step_deg)
    joints_rad = load_trajectory(Path(args.trajectory), args.start, args.frames)
    source_deltas = trajectory_deltas_deg(joints_rad, args.gain, limits)
    substep_limits = parse_limits(args.substep_max_deg) if args.substep_max_deg else limits
    deltas = subdivide_deltas(source_deltas, substep_limits)
    log_path = Path(args.log) if args.log else None
    period = 1.0 / args.rate

    total = [sum(step[i] for step in deltas) for i in range(6)]
    max_abs = [max(abs(step[i]) for step in deltas) for i in range(6)]

    print(f"Trajectory: {args.trajectory}")
    print(f"Frames selected: start={args.start}, frames={len(joints_rad)}, source_commands={len(source_deltas)}, commands={len(deltas)}")
    print(f"Replay rate: {args.rate:.1f} Hz, gain={args.gain:.4f}")
    print("Max source step limits deg: " + fmt(limits))
    print("Substep limits deg:        " + fmt(substep_limits))
    print("Planned total delta:  " + fmt(total))
    print("Planned max step:     " + fmt(max_abs))

    if not args.confirm:
        print("Dry-run only. Add --confirm to send commands to the real arm.")
        return 0

    ok_count = 0
    miss_count = 0
    with V7SerialClient(args.port, baudrate=args.baud, timeout=args.timeout) as client:
        try:
            client.flush()
            print("Sending stop before replay ...")
            stop_replies = client.stop()
            for reply in stop_replies:
                print(f"  stop <- {reply.raw}")
            append_jsonl(log_path, {"t": time.time(), "cmd": "stop_before", "replies": [r.raw for r in stop_replies]})
            time.sleep(args.settle)

            before = client.read_angle()
            print("Before angles: " + fmt(before))
            append_jsonl(log_path, {"t": time.time(), "phase": "before", "angles_deg": before})
            if args.pre_rl_delay > 0:
                print(f"Waiting {args.pre_rl_delay:.2f}s before RL binary frames ...")
                time.sleep(args.pre_rl_delay)
                client.flush()

            if not args.open_loop:
                for wi in range(args.warmup_frames):
                    try:
                        client.send_rl_delta([0.0] * 6)
                        report = client.read_rl_report(timeout=args.report_timeout)
                        status = "ok" if report is not None else "miss"
                        print(f"  warmup {wi:02d}: {status}")
                        append_jsonl(log_path, {"t": time.time(), "phase": "warmup", "index": wi, "ok": report is not None})
                    except Exception as exc:  # noqa: BLE001
                        print(f"  warmup {wi:02d}: serial error: {exc}")
                        append_jsonl(log_path, {"t": time.time(), "phase": "warmup", "index": wi, "ok": False, "error": str(exc)})
                        raise
                    time.sleep(period)

            for i, delta in enumerate(deltas):
                loop_t = time.monotonic()
                try:
                    client.send_rl_delta(delta)
                    if args.open_loop:
                        report = None
                    else:
                        report = client.read_rl_report(timeout=args.report_timeout)
                except Exception as exc:  # noqa: BLE001
                    print(f"  step {i:04d}: serial error: {exc}")
                    append_jsonl(log_path, {"t": time.time(), "step": i, "ok": False, "error": str(exc), "delta_deg": delta})
                    break

                if args.open_loop:
                    ok_count += 1
                    if args.print_every > 0 and i % args.print_every == 0:
                        print(f"  step {i:04d}: sent, cmd=[{fmt(delta)}]")
                    append_jsonl(log_path, {"t": time.time(), "step": i, "ok": True, "mode": "open_loop", "delta_deg": delta})
                elif report is None:
                    miss_count += 1
                    print(f"  step {i:04d}: miss, cmd=[{fmt(delta)}]")
                    append_jsonl(log_path, {"t": time.time(), "step": i, "ok": False, "delta_deg": delta})
                    if miss_count > args.max_misses:
                        print("Too many missed reports; aborting replay.")
                        break
                else:
                    ok_count += 1
                    print(f"  step {i:04d}: cmd=[{fmt(delta)}] report=[{fmt(report.angles_deg)}]")
                    append_jsonl(
                        log_path,
                        {
                            "t": time.time(),
                            "step": i,
                            "ok": True,
                            "delta_deg": delta,
                            "angles_deg": report.angles_deg,
                            "tcp_pos_mm": report.tcp_pos_mm,
                            "velocities_deg_s": report.velocities_deg_s,
                        },
                    )

                sleep_s = period - (time.monotonic() - loop_t)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            time.sleep(args.settle)
            drained = drain_input(client, args.post_drain)
            if drained:
                print(f"Drained {drained} leftover binary bytes before read_angle")
            try:
                after = client.read_angle()
                print("After angles:  " + fmt(after))
                print("Observed total: " + fmt(a - b for a, b in zip(after, before)))
                append_jsonl(log_path, {"t": time.time(), "phase": "after", "angles_deg": after})
            except Exception as exc:  # noqa: BLE001
                print(f"After read_angle failed: {exc}")
                append_jsonl(log_path, {"t": time.time(), "phase": "after", "error": str(exc)})

            print("Sending stop after replay ...")
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

    print(f"Replay finished: ok={ok_count}, missed={miss_count}")
    return 0 if ok_count > 0 and miss_count <= args.max_misses else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V7 low-speed scaled trajectory replay")
    parser.add_argument("--port", default="COM16", help="Serial port, default COM16")
    parser.add_argument("--baud", type=int, default=BAUDRATE_DEFAULT, help="Baudrate, default 115200")
    parser.add_argument("--trajectory", default=str(DEFAULT_TRAJECTORY), help="Exported browser trajectory JSON")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--frames", type=int, default=10, help="Number of source frames to use")
    parser.add_argument("--rate", type=float, default=10.0, help="Replay command rate in Hz")
    parser.add_argument("--gain", type=float, default=0.02, help="Scale applied to trajectory frame-to-frame deltas")
    parser.add_argument("--max-step-deg", default=",".join(str(x) for x in DEFAULT_MAX_STEP), help="One value or six comma-separated per-source-step clamps")
    parser.add_argument("--substep-max-deg", default=None, help="Optional one value or six comma-separated substep clamps after source-step planning")
    parser.add_argument("--timeout", type=float, default=0.1, help="Low-level serial timeout")
    parser.add_argument("--report-timeout", type=float, default=0.5, help="Binary report wait timeout")
    parser.add_argument("--settle", type=float, default=0.5, help="Settle delay before/after replay")
    parser.add_argument("--post-drain", type=float, default=0.8, help="Seconds to drain binary leftovers before final text read_angle")
    parser.add_argument("--max-misses", type=int, default=1, help="Abort after more than this many missed binary reports")
    parser.add_argument("--warmup-frames", type=int, default=1, help="Zero-delta RL frames before replay; useful for waking binary report channel")
    parser.add_argument("--pre-rl-delay", type=float, default=1.0, help="Delay after text read_angle before binary RL frames")
    parser.add_argument("--open-loop", action="store_true", help="Send trajectory deltas without waiting for per-step binary reports; verify by final read_angle")
    parser.add_argument("--confirm", action="store_true", help="Actually send commands to the real arm")
    parser.add_argument("--print-every", type=int, default=10, help="Print every N open-loop steps; 0 disables per-step printing")
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
    return run_replay(args)


if __name__ == "__main__":
    sys.exit(main())











