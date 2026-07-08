"""Pick-and-place using explicit A/B/C trajectories with RL report warmup."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from serial_protocol_v7 import BAUDRATE_DEFAULT, ProtocolError, V7SerialClient
from serial_replay_v7_trajectory import append_jsonl, clamp, drain_input, fmt, load_trajectory, parse_limits


DEFAULT_MAX_STEP = (0.6, 0.6, 0.6, 0.4, 0.4, 0.4)
DEFAULT_CL_TARGET_STEP = (0.20, 0.25, 0.25, 0.12, 0.12, 0.12)
DEFAULT_CL_MAX_CMD_STEP = (0.20, 0.25, 0.25, 0.12, 0.12, 0.12)
DEFAULT_CL_MIN_CMD_STEP = (0.015, 0.020, 0.020, 0.020, 0.020, 0.020)


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.buffer.write(text.encode(encoding, errors="backslashreplace") + b"\n")
    sys.stdout.flush()


def trajectory_deltas_deg(joints_rad: list[list[float]], gain: float, limits: tuple[float, ...]) -> list[list[float]]:
    result: list[list[float]] = []
    for prev, cur in zip(joints_rad, joints_rad[1:]):
        raw = [math.degrees(c - p) * gain for p, c in zip(prev, cur)]
        result.append([clamp(v, limit) for v, limit in zip(raw, limits)])
    return result


def parse_six_floats(text: str, option_name: str) -> tuple[float, float, float, float, float, float]:
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) == 1:
        parts = parts * 6
    if len(parts) != 6:
        raise ValueError(f"{option_name} must be one value or six comma-separated values")
    return tuple(parts)  # type: ignore[return-value]


def interpolate_relative_targets(deltas: list[list[float]], max_target_step: tuple[float, ...]) -> list[list[float]]:
    targets: list[list[float]] = []
    current = [0.0] * 6
    for delta in deltas:
        parts = max(1, max(math.ceil(abs(v) / limit) if limit > 0 else 1 for v, limit in zip(delta, max_target_step)))
        for part in range(1, parts + 1):
            alpha = part / parts
            targets.append([current[i] + delta[i] * alpha for i in range(6)])
        current = [current[i] + delta[i] for i in range(6)]
    return targets


def apply_min_step(command: list[float], min_steps: tuple[float, ...]) -> list[float]:
    result: list[float] = []
    for cmd, min_step in zip(command, min_steps):
        if abs(cmd) < min_step:
            result.append(0.0)
        else:
            result.append(cmd)
    return result


def vector_add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def vector_sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def segment_gain(args: argparse.Namespace, name: str) -> float:
    value = getattr(args, f"gain_{name}", None)
    if value is None:
        return float(args.gain)
    return float(value)


def wait_for_grip(client: V7SerialClient, command: str, timeout: float) -> list[str]:
    client.write_text(command)
    deadline = time.monotonic() + timeout
    replies: list[str] = []
    while time.monotonic() < deadline:
        reply = client.read_line(timeout=0.1)
        if reply is None:
            continue
        replies.append(reply.raw)
        if reply.command == "grip":
            data = (reply.data or "").lower()
            if "done" in data or "fail" in data:
                break
    return replies


def wait_for_stop_settle(client: V7SerialClient, settle: float, post_drain: float, note: str, log_path: Path | None) -> None:
    client.write_text("stop")
    replies = drain_text_replies(client, 0.5)
    time.sleep(settle)
    drained = drain_input(client, post_drain)
    if drained:
        print(f"Drained {drained} bytes before {note}")
    append_jsonl(log_path, {"t": time.time(), "phase": f"stop_before_{note}", "replies": replies, "drained": drained})


def drain_text_replies(client: V7SerialClient, timeout: float) -> list[str]:
    deadline = time.monotonic() + timeout
    replies: list[str] = []
    while time.monotonic() < deadline:
        reply = client.read_line(timeout=0.05)
        if reply is None:
            continue
        replies.append(reply.raw)
    return replies


def read_angle_clean(
    client: V7SerialClient,
    *,
    retries: int = 3,
    retry_sleep: float = 0.08,
) -> list[float]:
    last_error: Exception | None = None
    for _ in range(max(1, retries)):
        client.flush()
        try:
            return list(client.read_angle())
        except (ProtocolError, TimeoutError) as exc:
            last_error = exc
            time.sleep(retry_sleep)
    if last_error is not None:
        raise last_error
    raise TimeoutError("read_angle_clean timed out")


def read_angle_resilient(
    client: V7SerialClient,
    *,
    note: str,
    log_path: Path | None,
    fallback_angles_deg: list[float] | None = None,
    attempts: int = 3,
    settle_between: float = 0.12,
    drain_seconds: float = 0.15,
) -> list[float]:
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            angles = read_angle_clean(client)
            append_jsonl(
                log_path,
                {
                    "t": time.time(),
                    "phase": f"{note}_read_angle_ok",
                    "attempt": attempt,
                    "angles_deg": angles,
                },
            )
            return angles
        except (ProtocolError, TimeoutError) as exc:
            last_error = exc
            client.flush()
            drained = drain_input(client, drain_seconds)
            append_jsonl(
                log_path,
                {
                    "t": time.time(),
                    "phase": f"{note}_read_angle_retry",
                    "attempt": attempt,
                    "error": repr(exc),
                    "drained": drained,
                },
            )
            time.sleep(settle_between)
    if fallback_angles_deg is not None:
        append_jsonl(
            log_path,
            {
                "t": time.time(),
                "phase": f"{note}_read_angle_fallback",
                "error": repr(last_error) if last_error is not None else "unknown",
                "angles_deg": fallback_angles_deg,
            },
        )
        print(f"{note}: read_angle degraded, continue from fallback pose")
        return fallback_angles_deg[:]
    if last_error is not None:
        raise last_error
    raise TimeoutError(f"{note}: read_angle failed")


def wait_for_async_ack(
    client: V7SerialClient,
    command_name: str,
    timeout: float,
) -> list[str]:
    deadline = time.monotonic() + timeout
    replies: list[str] = []
    while time.monotonic() < deadline:
        reply = client.read_line(timeout=0.1)
        if reply is None:
            continue
        replies.append(reply.raw)
        if (reply.command or "").lower() != command_name.lower():
            continue
        state = ((reply.data or "").split() or [""])[0].upper()
        if state in {"DONE", "FAIL", "LIMIT", "OK"}:
            break
    return replies


def reply_has_limit(replies: list[str]) -> bool:
    return any("limit" in reply.lower() for reply in replies)


def execute_lin_keep_relative(
    client: V7SerialClient,
    *,
    dx: float,
    dy: float,
    dz: float,
    speed_ratio: float,
    settle: float,
    post_drain: float,
    timeout: float,
    phase: str,
    log_path: Path | None,
) -> list[str]:
    command = f"lin_keep {dx:.3f},{dy:.3f},{dz:.3f},{speed_ratio:.1f},0"
    print(f"{phase}: {command}")
    client.write_text(command)
    replies = wait_for_async_ack(client, "lin_keep", timeout)
    append_jsonl(log_path, {"t": time.time(), "phase": phase, "command": command, "replies": replies})
    time.sleep(settle)
    drained = drain_input(client, post_drain)
    append_jsonl(log_path, {"t": time.time(), "phase": f"{phase}_drain", "drained": drained})
    return replies


def execute_jogr_burst(
    client: V7SerialClient,
    *,
    command: str,
    repeat_count: int,
    burst_seconds: float,
    ack_window_seconds: float,
    stop_settle_seconds: float,
    post_drain_seconds: float,
    phase: str,
    log_path: Path | None,
) -> None:
    sends = max(1, repeat_count)
    total_seconds = max(0.1, burst_seconds)
    period = total_seconds / sends
    ack_window = max(0.0, min(ack_window_seconds, period))
    print(f"{phase}: burst {command} x{sends} over {total_seconds:.2f}s")
    append_jsonl(
        log_path,
        {
            "t": time.time(),
            "phase": f"{phase}_plan",
            "command": command,
            "repeat_count": sends,
            "burst_seconds": total_seconds,
            "period_seconds": period,
        },
    )
    for idx in range(sends):
        start_t = time.monotonic()
        client.write_text(command)
        replies = drain_text_replies(client, ack_window)
        append_jsonl(
            log_path,
            {
                "t": time.time(),
                "phase": f"{phase}_send",
                "index": idx,
                "command": command,
                "replies": replies,
            },
        )
        remaining = period - (time.monotonic() - start_t)
        if remaining > 0:
            time.sleep(remaining)
    client.write_text("stop")
    stop_replies = drain_text_replies(client, max(ack_window, 0.05))
    append_jsonl(log_path, {"t": time.time(), "phase": f"{phase}_stop", "replies": stop_replies})
    if stop_settle_seconds > 0:
        time.sleep(stop_settle_seconds)
    drained = drain_input(client, post_drain_seconds)
    append_jsonl(log_path, {"t": time.time(), "phase": f"{phase}_drain", "drained": drained})


def execute_precision_demo(client: V7SerialClient, args: argparse.Namespace, log_path: Path | None) -> None:
    demo_cfg = getattr(args, "precision_demo", None) or {}
    if not demo_cfg.get("enabled"):
        return
    lift_mm = float(demo_cfg.get("lift_mm", 0.0))
    lifted = False
    if abs(lift_mm) > 1e-6:
        lift_replies = execute_lin_keep_relative(
            client,
            dx=0.0,
            dy=0.0,
            dz=lift_mm,
            speed_ratio=float(demo_cfg.get("lift_speed_ratio", 20.0)),
            settle=args.settle,
            post_drain=args.post_drain,
            timeout=15.0,
            phase="precision_lift_up",
            log_path=log_path,
        )
        if reply_has_limit(lift_replies):
            print("precision_lift_up hit limit; skip extra lift and keep current pose")
            append_jsonl(log_path, {"t": time.time(), "phase": "precision_lift_up_limit", "replies": lift_replies})
        else:
            lifted = True
    lift_angles = read_angle_clean(client)
    print("Precision demo anchor: " + fmt(lift_angles))
    append_jsonl(log_path, {"t": time.time(), "phase": "precision_anchor", "angles_deg": lift_angles})

    repeat_count = int(demo_cfg.get("repeat_count", 10))
    sequence = list(demo_cfg.get("sequence", []))
    direction_seconds = float(demo_cfg.get("total_seconds", 4.0)) / max(1, len(sequence))
    for seq in sequence:
        label = str(seq.get("label", "jogr"))
        command = str(seq.get("command", ""))
        seq_repeat_count = int(seq.get("repeat_count", repeat_count))
        seq_duration_seconds = float(seq.get("duration_seconds", direction_seconds))
        execute_jogr_burst(
            client,
            command=command,
            repeat_count=seq_repeat_count,
            burst_seconds=seq_duration_seconds,
            ack_window_seconds=float(demo_cfg.get("ack_window_seconds", 0.03)),
            stop_settle_seconds=float(demo_cfg.get("direction_stop_settle_seconds", 0.05)),
            post_drain_seconds=float(demo_cfg.get("direction_post_drain_seconds", 0.05)),
            phase=f"precision_{label}",
            log_path=log_path,
        )

    if lifted:
        execute_lin_keep_relative(
            client,
            dx=0.0,
            dy=0.0,
            dz=-lift_mm,
            speed_ratio=float(demo_cfg.get("lift_speed_ratio", 20.0)),
            settle=args.settle,
            post_drain=args.post_drain,
            timeout=15.0,
            phase="precision_lift_down",
            log_path=log_path,
        )


def execute_pour_template(
    client: V7SerialClient,
    template_cfg: dict | None,
    *,
    settle: float,
    log_path: Path | None,
    phase_prefix: str,
) -> None:
    pour_cfg = template_cfg or {}
    if not pour_cfg.get("enabled"):
        return
    command = str(pour_cfg.get("command", "jog_j5"))
    down_speed = float(pour_cfg.get("j5_speed", -0.3))
    up_speed = float(pour_cfg.get("return_j5_speed", 0.3))
    rotate_seconds = float(pour_cfg.get("rotate_seconds", 6.0))
    hold_seconds = float(pour_cfg.get("hold_seconds", 2.0))

    print(f"{phase_prefix} start: {command} {down_speed}")
    client.write_text(f"{command} {down_speed}")
    down_replies = drain_text_replies(client, 0.5)
    append_jsonl(log_path, {"t": time.time(), "phase": f"{phase_prefix}_down_start", "command": command, "speed": down_speed, "replies": down_replies})
    time.sleep(rotate_seconds)
    client.write_text("stop")
    down_stop_replies = drain_text_replies(client, 0.5)
    append_jsonl(log_path, {"t": time.time(), "phase": f"{phase_prefix}_down_stop", "replies": down_stop_replies})
    time.sleep(settle)

    if hold_seconds > 0:
        print(f"{phase_prefix} hold: {hold_seconds:.2f}s")
        append_jsonl(log_path, {"t": time.time(), "phase": f"{phase_prefix}_hold", "seconds": hold_seconds})
        time.sleep(hold_seconds)

    print(f"{phase_prefix} return: {command} {up_speed}")
    client.write_text(f"{command} {up_speed}")
    up_replies = drain_text_replies(client, 0.5)
    append_jsonl(log_path, {"t": time.time(), "phase": f"{phase_prefix}_up_start", "command": command, "speed": up_speed, "replies": up_replies})
    time.sleep(rotate_seconds)
    client.write_text("stop")
    up_stop_replies = drain_text_replies(client, 0.5)
    append_jsonl(log_path, {"t": time.time(), "phase": f"{phase_prefix}_up_stop", "replies": up_stop_replies})
    time.sleep(settle)


def execute_reset_pose(
    client: V7SerialClient,
    *,
    speed_ratio: float,
    settle: float,
    post_drain: float,
    timeout: float,
    log_path: Path | None,
) -> None:
    command = f"reset {speed_ratio:.1f}"
    print(f"home_reset: {command}")
    client.write_text(command)
    replies = wait_for_async_ack(client, "reset", timeout)
    append_jsonl(log_path, {"t": time.time(), "phase": "home_reset", "command": command, "replies": replies})
    time.sleep(settle)
    drained = drain_input(client, post_drain)
    append_jsonl(log_path, {"t": time.time(), "phase": "home_reset_drain", "drained": drained})


def warmup_rl_reports(client: V7SerialClient, rate: float, report_timeout: float, warmup_frames: int, log_path: Path | None) -> None:
    for wi in range(warmup_frames):
        client.send_rl_delta([0.0] * 6)
        report = client.read_rl_report(timeout=report_timeout)
        print(f"  warmup {wi:02d}: {'ok' if report is not None else 'miss'}")
        append_jsonl(log_path, {"t": time.time(), "phase": "warmup", "index": wi, "ok": report is not None})
        time.sleep(1.0 / rate)


def replay_segment(
    client: V7SerialClient,
    deltas: list[list[float]],
    rate: float,
    report_timeout: float,
    print_every: int,
    log_path: Path | None,
    phase: str,
) -> tuple[int, int]:
    period = 1.0 / rate
    ok_count = 0
    miss_count = 0
    for i, delta in enumerate(deltas):
        loop_t = time.monotonic()
        client.send_rl_delta(delta)
        report = client.read_rl_report(timeout=report_timeout)
        if report is None:
            miss_count += 1
            if print_every > 0 and i % print_every == 0:
                print(f"{phase} step {i:04d}: miss cmd=[{fmt(delta)}]")
            append_jsonl(log_path, {"t": time.time(), "phase": phase, "step": i, "ok": False, "delta_deg": delta})
        else:
            ok_count += 1
            if print_every > 0 and i % print_every == 0:
                print(f"{phase} step {i:04d}: cmd=[{fmt(delta)}] report=[{fmt(report.angles_deg)}]")
            append_jsonl(
                log_path,
                {
                    "t": time.time(),
                    "phase": phase,
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
    return ok_count, miss_count


def replay_segment_closed_loop(
    client: V7SerialClient,
    joints_rad: list[list[float]],
    *,
    gain: float,
    source_limits: tuple[float, ...],
    target_step_limits: tuple[float, ...],
    cmd_limits: tuple[float, ...],
    min_cmd_steps: tuple[float, ...],
    kp: tuple[float, ...],
    ff_gain: tuple[float, ...],
    max_error_deg: float,
    rate: float,
    report_timeout: float,
    print_every: int,
    log_path: Path | None,
    phase: str,
) -> tuple[int, int]:
    deltas = trajectory_deltas_deg(joints_rad, gain, source_limits)
    targets_rel = interpolate_relative_targets(deltas, target_step_limits)
    if not targets_rel:
        return 0, 0

    period = 1.0 / rate
    anchor = read_angle_clean(client)
    actual = anchor[:]
    last_error = [0.0] * 6
    last_target_rel = [0.0] * 6
    ok_count = 0
    miss_count = 0
    append_jsonl(
        log_path,
        {
            "t": time.time(),
            "phase": f"{phase}_closed_loop_anchor",
            "angles_deg": anchor,
            "target_points": len(targets_rel),
        },
    )
    for i, target_rel in enumerate(targets_rel):
        loop_t = time.monotonic()
        target_abs = vector_add(anchor, target_rel)
        error = vector_sub(target_abs, actual)
        max_abs_error = max(abs(v) for v in error)
        if max_abs_error > max_error_deg:
            append_jsonl(
                log_path,
                {
                    "t": time.time(),
                    "phase": phase,
                    "step": i,
                    "ok": False,
                    "abort": "max_error",
                    "target_rel_deg": target_rel,
                    "target_abs_deg": target_abs,
                    "actual_deg": actual,
                    "error_deg": error,
                },
            )
            print(f"{phase} step {i:04d}: abort, tracking error too large {max_abs_error:.3f} deg")
            break

        ff = [target_rel[j] - last_target_rel[j] for j in range(6)]
        raw_cmd = [ff_gain[j] * ff[j] + kp[j] * error[j] for j in range(6)]
        limited_cmd = [clamp(raw_cmd[j], cmd_limits[j]) for j in range(6)]
        cmd = apply_min_step(limited_cmd, min_cmd_steps)

        client.send_rl_delta(cmd)
        report = client.read_rl_report(timeout=report_timeout)
        if report is None:
            miss_count += 1
            append_jsonl(
                log_path,
                {
                    "t": time.time(),
                    "phase": phase,
                    "step": i,
                    "ok": False,
                    "miss": True,
                    "target_rel_deg": target_rel,
                    "target_abs_deg": target_abs,
                    "actual_deg": actual,
                    "error_deg": error,
                    "feedforward_deg": ff,
                    "cmd_deg": cmd,
                },
            )
            if print_every > 0 and i % print_every == 0:
                print(f"{phase} step {i:04d}: miss cmd=[{fmt(cmd)}] err=[{fmt(error)}]")
        else:
            actual = list(report.angles_deg)
            post_error = vector_sub(target_abs, actual)
            ok_count += 1
            append_jsonl(
                log_path,
                {
                    "t": time.time(),
                    "phase": phase,
                    "step": i,
                    "ok": True,
                    "target_rel_deg": target_rel,
                    "target_abs_deg": target_abs,
                    "actual_deg": actual,
                    "error_deg": error,
                    "post_error_deg": post_error,
                    "feedforward_deg": ff,
                    "cmd_deg": cmd,
                    "tcp_pos_mm": report.tcp_pos_mm,
                    "velocities_deg_s": report.velocities_deg_s,
                },
            )
            last_error = post_error
            if print_every > 0 and (i % print_every == 0 or i == len(targets_rel) - 1):
                print(f"{phase} step {i:04d}: cmd=[{fmt(cmd)}] post_err=[{fmt(post_error)}]")
        last_target_rel = target_rel
        sleep_s = period - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return ok_count, miss_count


def restore_joint_anchor(
    client: V7SerialClient,
    *,
    current_angles_deg: list[float],
    target_angles_deg: list[float],
    min_frames: int,
    limits: tuple[float, ...],
    rate: float,
    report_timeout: float,
    print_every: int,
    log_path: Path | None,
    phase: str,
) -> tuple[int, int]:
    diffs = [target - current for target, current in zip(target_angles_deg, current_angles_deg)]
    required_frames = max(1, max(math.ceil(abs(diff) / limit) if limit > 0 else 1 for diff, limit in zip(diffs, limits)))
    frames = max(min_frames, required_frames)
    delta = [diff / frames for diff in diffs]
    append_jsonl(
        log_path,
        {
            "t": time.time(),
            "phase": f"{phase}_plan",
            "current_angles_deg": current_angles_deg,
            "target_angles_deg": target_angles_deg,
            "delta_deg": delta,
            "frames": frames,
        },
    )
    deltas = [delta[:] for _ in range(frames)]
    return replay_segment(client, deltas, rate, report_timeout, print_every, log_path, phase)


def run(args: argparse.Namespace) -> int:
    limits = parse_limits(args.max_step_deg)
    cl_target_step = parse_limits(args.cl_target_step_deg)
    cl_max_cmd = parse_limits(args.cl_max_cmd_deg)
    cl_min_cmd = parse_limits(args.cl_min_cmd_deg)
    cl_kp = parse_six_floats(args.cl_kp, "--cl-kp")
    cl_ff = parse_six_floats(args.cl_ff_gain, "--cl-ff-gain")
    gain_a = segment_gain(args, "a")
    gain_b = segment_gain(args, "b")
    gain_p = segment_gain(args, "p")
    gain_return = segment_gain(args, "return")
    gain_c = segment_gain(args, "c")
    joints_a = load_trajectory(Path(args.traj_a), 0, args.frames_a)
    joints_b = load_trajectory(Path(args.traj_b), 0, args.frames_b)
    joints_p = load_trajectory(Path(args.traj_p), 0, args.frames_p) if args.traj_p else None
    joints_return = load_trajectory(Path(args.traj_return), 0, args.frames_return) if args.traj_return else None
    joints_c = load_trajectory(Path(args.traj_c), 0, args.frames_c)
    deltas_a = trajectory_deltas_deg(joints_a, gain_a, limits)
    deltas_b = trajectory_deltas_deg(joints_b, gain_b, limits)
    deltas_p = trajectory_deltas_deg(joints_p, gain_p, limits) if joints_p else None
    deltas_return = trajectory_deltas_deg(joints_return, gain_return, limits) if joints_return else None
    deltas_c = trajectory_deltas_deg(joints_c, gain_c, limits)
    log_path = Path(args.log) if args.log else None
    precision_cfg = getattr(args, "precision_demo", None) or {}
    precision_phase = str(precision_cfg.get("phase", "after_grip_close"))

    print(f"Trajectory A: {args.traj_a} frames={len(joints_a)} commands={len(deltas_a)}")
    print(f"Trajectory B: {args.traj_b} frames={len(joints_b)} commands={len(deltas_b)}")
    if joints_p:
        print(f"Trajectory P: {args.traj_p} frames={len(joints_p)} commands={len(deltas_p)}")
    if joints_return:
        print(f"Trajectory R: {args.traj_return} frames={len(joints_return)} commands={len(deltas_return)}")
    print(f"Trajectory C: {args.traj_c} frames={len(joints_c)} commands={len(deltas_c)}")
    print(
        "Fixed replay rate: "
        f"{args.rate:.1f} Hz, gains="
        f"A:{gain_a:.3f} B:{gain_b:.3f} "
        f"P:{gain_p:.3f} R:{gain_return:.3f} C:{gain_c:.3f}"
    )
    if args.closed_loop_segments:
        print(
            "Closed-loop segments: "
            f"{args.closed_loop_segments} "
            f"target_step=[{fmt(cl_target_step)}] "
            f"cmd_limit=[{fmt(cl_max_cmd)}] "
            f"kp=[{fmt(cl_kp)}] ff=[{fmt(cl_ff)}]"
        )

    if not args.confirm:
        print("Dry-run only. Add --confirm to send commands to the real arm.")
        return 0

    with V7SerialClient(args.port, baudrate=args.baud, timeout=args.timeout) as client:
        client.flush()
        for reply in client.stop():
            safe_print(f"  stop <- {reply.raw}")
        time.sleep(args.settle)

        print(f"Sending {args.pre_open_command} ...")
        open_replies = wait_for_grip(client, args.pre_open_command, args.grip_timeout)
        for raw in open_replies:
            safe_print(f"  grip <- {raw}")
        append_jsonl(log_path, {"t": time.time(), "phase": "pre_open", "replies": open_replies})
        time.sleep(args.grip_pause)

        start_angles = read_angle_clean(client)
        if args.pre_rl_delay > 0:
            time.sleep(args.pre_rl_delay)
            client.flush()
        warmup_rl_reports(client, args.rate, args.report_timeout, args.warmup_frames, log_path)
        print("Trajectory A start: " + fmt(start_angles))
        append_jsonl(log_path, {"t": time.time(), "phase": "traj_a_start", "angles_deg": start_angles})
        if "a" in args.closed_loop_segments:
            ok_a, miss_a = replay_segment_closed_loop(
                client,
                joints_a,
                gain=gain_a,
                source_limits=limits,
                target_step_limits=cl_target_step,
                cmd_limits=cl_max_cmd,
                min_cmd_steps=cl_min_cmd,
                kp=cl_kp,
                ff_gain=cl_ff,
                max_error_deg=args.cl_max_error_deg,
                rate=args.rate,
                report_timeout=args.report_timeout,
                print_every=args.print_every,
                log_path=log_path,
                phase="traj_a",
            )
        else:
            ok_a, miss_a = replay_segment(client, deltas_a, args.rate, args.report_timeout, args.print_every, log_path, "traj_a")

        wait_for_stop_settle(client, args.settle, args.post_drain, "grip_close", log_path)

        grasp_angles = read_angle_clean(client)
        print("At grasp point: " + fmt(grasp_angles))
        append_jsonl(log_path, {"t": time.time(), "phase": "at_grasp", "angles_deg": grasp_angles})

        print(f"Sending {args.grip_close_command} ...")
        close_replies = wait_for_grip(client, args.grip_close_command, args.grip_timeout)
        for raw in close_replies:
            safe_print(f"  grip <- {raw}")
        append_jsonl(log_path, {"t": time.time(), "phase": "grip_close", "replies": close_replies})
        time.sleep(args.grip_pause)

        if precision_phase == "after_grip_close":
            execute_precision_demo(client, args, log_path)

        transport_start = read_angle_resilient(
            client,
            note="after_grip_close",
            log_path=log_path,
            fallback_angles_deg=grasp_angles,
        )
        if args.pre_rl_delay > 0:
            time.sleep(args.pre_rl_delay)
            client.flush()
        warmup_rl_reports(client, args.rate, args.report_timeout, args.warmup_frames, log_path)
        print("Trajectory B start: " + fmt(transport_start))
        append_jsonl(log_path, {"t": time.time(), "phase": "traj_b_start", "angles_deg": transport_start})
        if "b" in args.closed_loop_segments:
            ok_b, miss_b = replay_segment_closed_loop(
                client,
                joints_b,
                gain=gain_b,
                source_limits=limits,
                target_step_limits=cl_target_step,
                cmd_limits=cl_max_cmd,
                min_cmd_steps=cl_min_cmd,
                kp=cl_kp,
                ff_gain=cl_ff,
                max_error_deg=args.cl_max_error_deg,
                rate=args.rate,
                report_timeout=args.report_timeout,
                print_every=args.print_every,
                log_path=log_path,
                phase="traj_b",
            )
        else:
            ok_b, miss_b = replay_segment(client, deltas_b, args.rate, args.report_timeout, args.print_every, log_path, "traj_b")

        ok_p = miss_p = 0
        if deltas_p is not None:
            wait_for_stop_settle(client, args.settle, args.post_drain, "pour", log_path)
            pour_start = read_angle_resilient(
                client,
                note="before_pour",
                log_path=log_path,
                fallback_angles_deg=transport_start,
            )
            print("At pour point: " + fmt(pour_start))
            append_jsonl(log_path, {"t": time.time(), "phase": "at_pour", "angles_deg": pour_start})
            execute_pour_template(
                client,
                getattr(args, "pour", None),
                settle=args.settle,
                log_path=log_path,
                phase_prefix="pour",
            )
            place_start = read_angle_resilient(
                client,
                note="after_pour",
                log_path=log_path,
                fallback_angles_deg=pour_start,
            )
            if args.pre_rl_delay > 0:
                time.sleep(args.pre_rl_delay)
                client.flush()
            warmup_rl_reports(client, args.rate, args.report_timeout, args.warmup_frames, log_path)
            print("Trajectory P start: " + fmt(place_start))
            append_jsonl(log_path, {"t": time.time(), "phase": "traj_p_start", "angles_deg": place_start})
            if "p" in args.closed_loop_segments:
                ok_p, miss_p = replay_segment_closed_loop(
                    client,
                    joints_p,
                    gain=gain_p,
                    source_limits=limits,
                    target_step_limits=cl_target_step,
                    cmd_limits=cl_max_cmd,
                    min_cmd_steps=cl_min_cmd,
                    kp=cl_kp,
                    ff_gain=cl_ff,
                    max_error_deg=args.cl_max_error_deg,
                    rate=args.rate,
                    report_timeout=args.report_timeout,
                    print_every=args.print_every,
                    log_path=log_path,
                    phase="traj_p",
                )
            else:
                ok_p, miss_p = replay_segment(client, deltas_p, args.rate, args.report_timeout, args.print_every, log_path, "traj_p")

        wait_for_stop_settle(client, args.settle, args.post_drain, "after_traj_b", log_path)

        place_angles = read_angle_clean(client)
        print("At place point: " + fmt(place_angles))
        append_jsonl(log_path, {"t": time.time(), "phase": "at_place", "angles_deg": place_angles})

        ok_restore = miss_restore = 0
        if precision_phase == "at_place":
            precision_anchor_angles = place_angles[:]
            execute_precision_demo(client, args, log_path)
            wait_for_stop_settle(client, args.settle, args.post_drain, "after_precision_demo", log_path)
            place_angles = read_angle_clean(client)
            print("After precision demo: " + fmt(place_angles))
            append_jsonl(log_path, {"t": time.time(), "phase": "after_precision_demo", "angles_deg": place_angles})
            if precision_cfg.get("restore_anchor_after_demo"):
                if args.pre_rl_delay > 0:
                    time.sleep(args.pre_rl_delay)
                    client.flush()
                warmup_rl_reports(client, args.rate, args.report_timeout, args.warmup_frames, log_path)
                ok_restore, miss_restore = restore_joint_anchor(
                    client,
                    current_angles_deg=place_angles,
                    target_angles_deg=precision_anchor_angles,
                    min_frames=int(precision_cfg.get("restore_anchor_frames", 120)),
                    limits=limits,
                    rate=args.rate,
                    report_timeout=args.report_timeout,
                    print_every=args.print_every,
                    log_path=log_path,
                    phase="precision_restore",
                )
                wait_for_stop_settle(client, args.settle, args.post_drain, "after_precision_restore", log_path)
                place_angles = read_angle_clean(client)
                print("Restored place anchor: " + fmt(place_angles))
                append_jsonl(
                    log_path,
                    {
                        "t": time.time(),
                        "phase": "after_precision_restore",
                        "angles_deg": place_angles,
                        "ok_restore": ok_restore,
                        "miss_restore": miss_restore,
                    },
                )

        execute_pour_template(
            client,
            getattr(args, "post_place_pour", None),
            settle=args.settle,
            log_path=log_path,
            phase_prefix="post_place_pour",
        )
        if getattr(args, "post_place_pour", None):
            place_angles = read_angle_clean(client)
            print("After post-place pour: " + fmt(place_angles))
            append_jsonl(log_path, {"t": time.time(), "phase": "after_post_place_pour", "angles_deg": place_angles})

        ok_return = miss_return = 0
        ok_c = miss_c = 0
        if deltas_return is not None:
            return_start = read_angle_clean(client)
            if args.pre_rl_delay > 0:
                time.sleep(args.pre_rl_delay)
                client.flush()
            warmup_rl_reports(client, args.rate, args.report_timeout, args.warmup_frames, log_path)
            print("Trajectory R start: " + fmt(return_start))
            append_jsonl(log_path, {"t": time.time(), "phase": "traj_r_start", "angles_deg": return_start})
            if "return" in args.closed_loop_segments:
                ok_return, miss_return = replay_segment_closed_loop(
                    client,
                    joints_return,
                    gain=gain_return,
                    source_limits=limits,
                    target_step_limits=cl_target_step,
                    cmd_limits=cl_max_cmd,
                    min_cmd_steps=cl_min_cmd,
                    kp=cl_kp,
                    ff_gain=cl_ff,
                    max_error_deg=args.cl_max_error_deg,
                    rate=args.rate,
                    report_timeout=args.report_timeout,
                    print_every=args.print_every,
                    log_path=log_path,
                    phase="traj_r",
                )
            else:
                ok_return, miss_return = replay_segment(client, deltas_return, args.rate, args.report_timeout, args.print_every, log_path, "traj_r")
            wait_for_stop_settle(client, args.settle, args.post_drain, "after_traj_r", log_path)
            return_angles = read_angle_clean(client)
            print("Back at pick point: " + fmt(return_angles))
            append_jsonl(log_path, {"t": time.time(), "phase": "at_return_pick", "angles_deg": return_angles})
            if args.release_after_return:
                print(f"Sending {args.grip_open_command} ...")
                release_replies = wait_for_grip(client, args.grip_open_command, args.grip_timeout)
                for raw in release_replies:
                    safe_print(f"  grip <- {raw}")
                append_jsonl(log_path, {"t": time.time(), "phase": "grip_open_release", "replies": release_replies})
                time.sleep(args.grip_pause)
            if args.home_reset_speed is not None:
                execute_reset_pose(
                    client,
                    speed_ratio=args.home_reset_speed,
                    settle=args.settle,
                    post_drain=args.post_drain,
                    timeout=30.0,
                    log_path=log_path,
                )
        else:
            print(f"Sending {args.grip_open_command} ...")
            release_replies = wait_for_grip(client, args.grip_open_command, args.grip_timeout)
            for raw in release_replies:
                safe_print(f"  grip <- {raw}")
            append_jsonl(log_path, {"t": time.time(), "phase": "grip_open_release", "replies": release_replies})
            time.sleep(args.grip_pause)

            if args.home_reset_speed is not None:
                execute_reset_pose(
                    client,
                    speed_ratio=args.home_reset_speed,
                    settle=args.settle,
                    post_drain=args.post_drain,
                    timeout=30.0,
                    log_path=log_path,
                )
            else:
                retreat_start = read_angle_clean(client)
                if args.pre_rl_delay > 0:
                    time.sleep(args.pre_rl_delay)
                    client.flush()
                warmup_rl_reports(client, args.rate, args.report_timeout, args.warmup_frames, log_path)
                print("Trajectory C start: " + fmt(retreat_start))
                append_jsonl(log_path, {"t": time.time(), "phase": "traj_c_start", "angles_deg": retreat_start})
                if "c" in args.closed_loop_segments:
                    ok_c, miss_c = replay_segment_closed_loop(
                        client,
                        joints_c,
                        gain=gain_c,
                        source_limits=limits,
                        target_step_limits=cl_target_step,
                        cmd_limits=cl_max_cmd,
                        min_cmd_steps=cl_min_cmd,
                        kp=cl_kp,
                        ff_gain=cl_ff,
                        max_error_deg=args.cl_max_error_deg,
                        rate=args.rate,
                        report_timeout=args.report_timeout,
                        print_every=args.print_every,
                        log_path=log_path,
                        phase="traj_c",
                    )
                else:
                    ok_c, miss_c = replay_segment(client, deltas_c, args.rate, args.report_timeout, args.print_every, log_path, "traj_c")

        time.sleep(args.final_settle)
        drained = drain_input(client, args.post_drain)
        if drained:
            print(f"Drained {drained} leftover binary bytes before final read_angle")
        after = read_angle_clean(client)
        print("Final read_angle: " + fmt(after))
        append_jsonl(
            log_path,
            {
                "t": time.time(),
                "phase": "after",
                "angles_deg": after,
                "ok_a": ok_a,
                "miss_a": miss_a,
                "ok_b": ok_b,
                "miss_b": miss_b,
                "ok_p": ok_p,
                "miss_p": miss_p,
                "ok_restore": ok_restore,
                "miss_restore": miss_restore,
                "ok_return": ok_return,
                "miss_return": miss_return,
                "ok_c": ok_c,
                "miss_c": miss_c,
            },
        )
        for reply in client.stop():
            safe_print(f"  stop <- {reply.raw}")
        return 0 if (miss_a + miss_b + miss_p + miss_restore + miss_return + miss_c) == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pick/place using explicit A/B/C trajectories with RL warmup")
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--baud", type=int, default=BAUDRATE_DEFAULT)
    parser.add_argument("--traj-a", required=True)
    parser.add_argument("--traj-b", required=True)
    parser.add_argument("--traj-c", required=True)
    parser.add_argument("--traj-p", default=None)
    parser.add_argument("--traj-return", default=None)
    parser.add_argument("--frames-a", type=int, default=140)
    parser.add_argument("--frames-b", type=int, default=140)
    parser.add_argument("--frames-c", type=int, default=140)
    parser.add_argument("--frames-p", type=int, default=140)
    parser.add_argument("--frames-return", type=int, default=140)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--gain-a", type=float, default=None)
    parser.add_argument("--gain-b", type=float, default=None)
    parser.add_argument("--gain-p", type=float, default=None)
    parser.add_argument("--gain-return", type=float, default=None)
    parser.add_argument("--gain-c", type=float, default=None)
    parser.add_argument("--closed-loop-segments", default="", help="Comma-separated segments to run in closed loop: a,b,p,return,c")
    parser.add_argument("--cl-target-step-deg", default=",".join(str(x) for x in DEFAULT_CL_TARGET_STEP))
    parser.add_argument("--cl-max-cmd-deg", default=",".join(str(x) for x in DEFAULT_CL_MAX_CMD_STEP))
    parser.add_argument("--cl-min-cmd-deg", default=",".join(str(x) for x in DEFAULT_CL_MIN_CMD_STEP))
    parser.add_argument("--cl-kp", default="0.35,0.45,0.45,0.35,0.35,0.35")
    parser.add_argument("--cl-ff-gain", default="1.0")
    parser.add_argument("--cl-max-error-deg", type=float, default=8.0)
    parser.add_argument("--max-step-deg", default=",".join(str(x) for x in DEFAULT_MAX_STEP))
    parser.add_argument("--timeout", type=float, default=0.05)
    parser.add_argument("--report-timeout", type=float, default=0.08)
    parser.add_argument("--settle", type=float, default=0.6)
    parser.add_argument("--final-settle", type=float, default=0.8)
    parser.add_argument("--post-drain", type=float, default=0.8)
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--pre-rl-delay", type=float, default=0.3)
    parser.add_argument("--pre-open-command", default="grip_open")
    parser.add_argument("--grip-close-command", default="grip_close")
    parser.add_argument("--grip-open-command", default="grip_open")
    parser.add_argument("--release-after-return", action="store_true")
    parser.add_argument("--home-reset-speed", type=float, default=None)
    parser.add_argument("--grip-timeout", type=float, default=20.0)
    parser.add_argument("--grip-pause", type=float, default=0.6)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--pour-json", default=None)
    parser.add_argument("--precision-demo-json", default=None)
    parser.add_argument("--post-place-pour-json", default=None)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--log")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.frames_a < 2 or args.frames_b < 2 or args.frames_c < 2:
        parser.error("--frames-a, --frames-b and --frames-c must be at least 2")
    if args.traj_p and args.frames_p < 2:
        parser.error("--frames-p must be at least 2 when --traj-p is set")
    if args.traj_return and args.frames_return < 2:
        parser.error("--frames-return must be at least 2 when --traj-return is set")
    if args.pour_json:
        import json
        args.pour = json.loads(Path(args.pour_json).read_text(encoding="utf-8"))
    else:
        args.pour = None
    if args.precision_demo_json:
        import json
        args.precision_demo = json.loads(Path(args.precision_demo_json).read_text(encoding="utf-8"))
    else:
        args.precision_demo = None
    if args.post_place_pour_json:
        import json
        args.post_place_pour = json.loads(Path(args.post_place_pour_json).read_text(encoding="utf-8"))
    else:
        args.post_place_pour = None
    args.closed_loop_segments = {
        item.strip().lower()
        for item in str(args.closed_loop_segments).split(",")
        if item.strip()
    }
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
