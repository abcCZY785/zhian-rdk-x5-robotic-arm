"""Fixed-rate RL delta replay for updated V7 lower-controller tests.

This script is intentionally separate from the older gated/G2 scripts.
It tests whether the updated lower controller can track a browser-exported
trajectory when the upper PC sends RL delta frames at a fixed command rate,
starting from 20 Hz.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from serial_protocol_v7 import BAUDRATE_DEFAULT, V7SerialClient
from serial_replay_v7_trajectory import (
    DEFAULT_TRAJECTORY,
    append_jsonl,
    drain_input,
    fmt,
    load_trajectory,
    parse_limits,
    subdivide_deltas,
    trajectory_deltas_deg,
)


DEFAULT_MAX_STEP = (0.3, 0.3, 0.3, 0.2, 0.2, 0.2)


def run(args: argparse.Namespace) -> int:
    limits = parse_limits(args.max_step_deg)
    joints_rad = load_trajectory(Path(args.trajectory), args.start, args.frames)
    source_deltas = trajectory_deltas_deg(joints_rad, args.gain, limits)
    substep_limits = parse_limits(args.substep_max_deg) if args.substep_max_deg else limits
    deltas = subdivide_deltas(source_deltas, substep_limits)
    period = 1.0 / args.rate
    log_path = Path(args.log) if args.log else None

    planned_total = [sum(step[i] for step in deltas) for i in range(6)]
    planned_max = [max(abs(step[i]) for step in deltas) for i in range(6)] if deltas else [0.0] * 6

    print(f"Trajectory: {args.trajectory}")
    print(f"Frames selected: start={args.start}, frames={len(joints_rad)}, source_commands={len(source_deltas)}, commands={len(deltas)}")
    print(f"Fixed replay rate: {args.rate:.1f} Hz, period={period:.3f}s, gain={args.gain:.4f}")
    print("Source step limits: " + fmt(limits))
    print("Substep limits:     " + fmt(substep_limits))
    print("Planned total:      " + fmt(planned_total))
    print("Planned max step:   " + fmt(planned_max))

    if not args.confirm:
        print("Dry-run only. Add --confirm to send commands to the real arm.")
        return 0

    ok_count = 0
    miss_count = 0
    reports_seen = 0
    last_report_angles = None
    before = None

    with V7SerialClient(args.port, baudrate=args.baud, timeout=args.timeout) as client:
        try:
            client.flush()
            print("Sending stop before fixed-rate replay ...")
            for reply in client.stop():
                print(f"  stop <- {reply.raw}")
            time.sleep(args.settle)

            before = client.read_angle()
            last_report_angles = list(before)
            print("Before angles: " + fmt(before))
            append_jsonl(log_path, {"t": time.time(), "phase": "before", "angles_deg": before})

            if args.pre_rl_delay > 0:
                time.sleep(args.pre_rl_delay)
                client.flush()

            for wi in range(args.warmup_frames):
                client.send_rl_delta([0.0] * 6)
                report = client.read_rl_report(timeout=args.report_timeout)
                if report is not None:
                    reports_seen += 1
                    last_report_angles = list(report.angles_deg)
                print(f"  warmup {wi:02d}: {'ok' if report is not None else 'miss'}")
                append_jsonl(log_path, {"t": time.time(), "phase": "warmup", "index": wi, "ok": report is not None})
                time.sleep(period)

            for i, delta in enumerate(deltas):
                loop_t = time.monotonic()
                client.send_rl_delta(delta)
                report = client.read_rl_report(timeout=args.report_timeout)

                if report is None:
                    miss_count += 1
                    if args.print_every > 0 and i % args.print_every == 0:
                        print(f"step {i:04d}: miss cmd=[{fmt(delta)}]")
                    append_jsonl(log_path, {"t": time.time(), "step": i, "ok": False, "delta_deg": delta})
                    if miss_count > args.max_misses:
                        print("Too many missed reports; aborting replay.")
                        break
                else:
                    ok_count += 1
                    reports_seen += 1
                    last_report_angles = list(report.angles_deg)
                    if args.print_every > 0 and i % args.print_every == 0:
                        print(f"step {i:04d}: cmd=[{fmt(delta)}] report=[{fmt(report.angles_deg)}]")
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

            time.sleep(args.final_settle)
            drained = drain_input(client, args.post_drain)
            if drained:
                print(f"Drained {drained} leftover binary bytes before final read_angle")

            after = client.read_angle()
            observed_total = [a - b for a, b in zip(after, before)]
            final_error = [p - o for p, o in zip(planned_total, observed_total)]
            print("After angles:   " + fmt(after))
            print("Observed total: " + fmt(observed_total))
            print("Final error:    " + fmt(final_error))
            print(f"Summary: ok={ok_count}, missed={miss_count}, reports={reports_seen}, commands={len(deltas)}")
            append_jsonl(
                log_path,
                {
                    "t": time.time(),
                    "phase": "after",
                    "angles_deg": after,
                    "planned_delta_deg": planned_total,
                    "observed_total_deg": observed_total,
                    "final_error_deg": final_error,
                    "ok": ok_count,
                    "missed": miss_count,
                    "reports": reports_seen,
                    "commands": len(deltas),
                    "last_report_angles_deg": last_report_angles,
                },
            )

            print("Sending stop after fixed-rate replay ...")
            for reply in client.stop():
                print(f"  stop <- {reply.raw}")
            return 0 if ok_count > 0 and miss_count <= args.max_misses else 1
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V7 fixed-rate RL delta trajectory replay")
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--baud", type=int, default=BAUDRATE_DEFAULT)
    parser.add_argument("--trajectory", default=str(DEFAULT_TRAJECTORY))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--gain", type=float, default=0.05)
    parser.add_argument("--max-step-deg", default=",".join(str(x) for x in DEFAULT_MAX_STEP))
    parser.add_argument("--substep-max-deg", default=None)
    parser.add_argument("--timeout", type=float, default=0.05)
    parser.add_argument("--report-timeout", type=float, default=0.08)
    parser.add_argument("--settle", type=float, default=0.5)
    parser.add_argument("--final-settle", type=float, default=0.6)
    parser.add_argument("--post-drain", type=float, default=0.8)
    parser.add_argument("--max-misses", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--pre-rl-delay", type=float, default=0.3)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--log")
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
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
