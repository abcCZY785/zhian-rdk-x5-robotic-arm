"""V7 lower-controller serial protocol helpers.

This module implements the protocol described by upper_pc_protocol.md:
- text commands terminated by LF
- RL binary delta command: 0x55 0xBB + 6 float32 deltas + checksum
- RL binary report frame: 58 bytes on the current firmware, with 65-byte compatibility parsing

All angles in this file are degrees unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import struct
import time
from typing import Iterable, Optional

try:
    import serial
except ImportError:  # pragma: no cover - handled at runtime by scripts
    serial = None


BAUDRATE_DEFAULT = 115200
TEXT_EOL = b"\n"
RL_COMMAND_HEADER = b"\x55\xBB"
RL_REPORT_HEADER = b"\x55\xAA"
RL_COMMAND_SIZE = 27
RL_REPORT_SIZE_58 = 58
RL_REPORT_SIZE_65 = 65
RL_REPORT_MAX_SIZE = 65
RL_REPORT_PAYLOAD_LEN = 0x37


class ProtocolError(RuntimeError):
    """Raised when a serial protocol frame is malformed."""


@dataclass(frozen=True)
class AckLine:
    raw: str
    status: str
    command: Optional[str] = None
    data: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status.upper() in {"OK", "DONE", "ACK"}


@dataclass(frozen=True)
class RlReport:
    angles_deg: tuple[float, float, float, float, float, float]
    tcp_pos_mm: tuple[float, float, float]
    velocities_deg_s: tuple[float, float, float, float, float, float]
    raw: bytes


_ACK_RE = re.compile(r"^(?P<prefix>ACK|OK|DONE|ERROR|ERR)(?:\s+(?P<rest>.*))?$", re.I)
_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _as_six_floats(values: Iterable[float], name: str) -> tuple[float, float, float, float, float, float]:
    vals = tuple(float(v) for v in values)
    if len(vals) != 6:
        raise ValueError(f"{name} must contain exactly 6 values, got {len(vals)}")
    return vals  # type: ignore[return-value]


def checksum(data: bytes) -> int:
    return sum(data) & 0xFF


def pack_rl_delta(deltas_deg: Iterable[float]) -> bytes:
    """Pack a 6-axis RL delta command frame.

    Frame layout: 0x55 0xBB + <6f delta_deg> + checksum(sum bytes[2:26]).
    """
    deltas = _as_six_floats(deltas_deg, "deltas_deg")
    body = struct.pack("<6f", *deltas)
    return RL_COMMAND_HEADER + body + bytes([checksum(body)])


def pack_text_command(command: str) -> bytes:
    """Pack a text command with one trailing LF."""
    command = command.strip()
    if not command:
        raise ValueError("command cannot be empty")
    return command.encode("ascii") + TEXT_EOL


def parse_ack_line(line: bytes | str) -> AckLine:
    """Parse a single text reply line.

    The firmware docs do not fully pin down ACK shape, so this parser keeps the
    raw line and extracts a conservative status/command/data view.
    """
    if isinstance(line, bytes):
        raw = line.decode("utf-8", errors="replace").strip()
    else:
        raw = line.strip()

    if not raw:
        return AckLine(raw=raw, status="EMPTY")

    match = _ACK_RE.match(raw)
    if not match:
        return AckLine(raw=raw, status="DATA", data=raw)

    status = match.group("prefix").upper()
    rest = (match.group("rest") or "").strip()
    if not rest:
        return AckLine(raw=raw, status=status)

    parts = rest.split(maxsplit=1)
    command = parts[0]
    data = parts[1] if len(parts) > 1 else None
    return AckLine(raw=raw, status=status, command=command, data=data)


def parse_float_list(text: str, expected: int | None = None) -> tuple[float, ...]:
    values = tuple(float(x) for x in _FLOAT_RE.findall(text))
    if expected is not None and len(values) != expected:
        raise ProtocolError(f"expected {expected} floats, got {len(values)} from: {text!r}")
    return values


def parse_angles_text(text: str) -> tuple[float, float, float, float, float, float]:
    return _as_six_floats(parse_float_list(text, expected=6), "angles")


def parse_pose_text(text: str) -> tuple[float, float, float, float, float, float]:
    return _as_six_floats(parse_float_list(text, expected=6), "pose")


def parse_rl_report(frame: bytes) -> Optional[RlReport]:
    """Parse one RL report frame.

    Current firmware on COM16 was measured returning 58 bytes:
    header2 + len1 + angles24 + pos12 + 4 float32 reserved/velocity values +
    padding1 + checksum1 + tail1. The protocol table/example also describes a
    65-byte variant with 6 float32 velocities, so both variants are accepted.
    """
    if len(frame) < RL_REPORT_SIZE_58:
        return None
    if frame[:2] != RL_REPORT_HEADER:
        return None
    if frame[2] != RL_REPORT_PAYLOAD_LEN:
        return None

    if len(frame) >= RL_REPORT_SIZE_65 and frame[64] == 0x0D and checksum(frame[2:63]) == frame[63]:
        raw = frame[:RL_REPORT_SIZE_65]
        angles = struct.unpack("<6f", raw[3:27])
        pos = struct.unpack("<3f", raw[27:39])
        vel = struct.unpack("<6f", raw[39:63])
        return RlReport(
            angles_deg=angles,  # type: ignore[arg-type]
            tcp_pos_mm=pos,  # type: ignore[arg-type]
            velocities_deg_s=vel,  # type: ignore[arg-type]
            raw=raw,
        )

    if frame[57] != 0x0D:
        return None
    if checksum(frame[2:56]) != frame[56]:
        return None

    raw = frame[:RL_REPORT_SIZE_58]
    angles = struct.unpack("<6f", raw[3:27])
    pos = struct.unpack("<3f", raw[27:39])
    vel4 = struct.unpack("<4f", raw[39:55])
    vel6 = vel4 + (0.0, 0.0)
    return RlReport(
        angles_deg=angles,  # type: ignore[arg-type]
        tcp_pos_mm=pos,  # type: ignore[arg-type]
        velocities_deg_s=vel6,  # type: ignore[arg-type]
        raw=raw,
    )


class V7SerialClient:
    """Small blocking client for the V7 lower-controller protocol."""

    def __init__(self, port: str, baudrate: int = BAUDRATE_DEFAULT, timeout: float = 0.2):
        if serial is None:
            raise RuntimeError("pyserial is not installed. Install with: python -m pip install pyserial")
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout, write_timeout=timeout)
        self._binary_rx = bytearray()

    def close(self) -> None:
        self.ser.close()

    def __enter__(self) -> "V7SerialClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def flush(self) -> None:
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self._binary_rx.clear()

    def write_text(self, command: str) -> None:
        self.ser.write(pack_text_command(command))
        self.ser.flush()

    def read_line(self, timeout: Optional[float] = None) -> Optional[AckLine]:
        old_timeout = self.ser.timeout
        if timeout is not None:
            self.ser.timeout = timeout
        try:
            line = self.ser.readline()
        finally:
            self.ser.timeout = old_timeout
        if not line:
            return None
        return parse_ack_line(line)

    def command(self, command: str, timeout: float = 1.0) -> list[AckLine]:
        """Send one text command and collect reply lines until timeout gap."""
        self.write_text(command)
        deadline = time.monotonic() + timeout
        replies: list[AckLine] = []
        while time.monotonic() < deadline:
            reply = self.read_line(timeout=min(0.1, max(0.01, deadline - time.monotonic())))
            if reply is None:
                continue
            replies.append(reply)
            if reply.status in {"ACK", "OK", "DONE", "ERROR", "ERR", "DATA"}:
                # Keep the command helper conservative: most sync commands return
                # one useful line. Probe scripts can call read_line again if needed.
                break
        return replies

    def stop(self) -> list[AckLine]:
        return self.command("stop", timeout=1.0)

    def read_angle(self, retries: int = 2) -> tuple[float, float, float, float, float, float]:
        last_error: Optional[Exception] = None
        for _ in range(max(1, retries + 1)):
            replies = self.command("read_angle", timeout=1.0)
            if replies:
                text = replies[-1].data or replies[-1].raw
                try:
                    return parse_angles_text(text)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
            else:
                last_error = TimeoutError("read_angle timed out")
            time.sleep(0.1)
        raise last_error or TimeoutError("read_angle timed out")

    def read_pos(self, retries: int = 2) -> tuple[float, float, float, float, float, float]:
        last_error: Optional[Exception] = None
        for _ in range(max(1, retries + 1)):
            replies = self.command("read_pos", timeout=1.0)
            if replies:
                text = replies[-1].data or replies[-1].raw
                try:
                    return parse_pose_text(text)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
            else:
                last_error = TimeoutError("read_pos timed out")
            time.sleep(0.1)
        raise last_error or TimeoutError("read_pos timed out")

    def send_rl_delta(self, deltas_deg: Iterable[float]) -> None:
        self.ser.write(pack_rl_delta(deltas_deg))
        self.ser.flush()

    def read_rl_report(self, timeout: float = 0.1) -> Optional[RlReport]:
        """Read and parse one binary report frame, resyncing on 0x55 0xAA."""
        old_timeout = self.ser.timeout
        self.ser.timeout = timeout
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                # Keep partial binary frames across calls. At 10-20Hz a report may
                # arrive split across several serial reads; dropping partial bytes
                # makes a closed-loop controller look like it lost feedback.
                chunk = self.ser.read(RL_REPORT_SIZE_58)
                if not chunk:
                    continue
                self._binary_rx.extend(chunk)
                while len(self._binary_rx) >= 2 and bytes(self._binary_rx[:2]) != RL_REPORT_HEADER:
                    del self._binary_rx[0]
                while len(self._binary_rx) >= RL_REPORT_SIZE_58:
                    report = parse_rl_report(bytes(self._binary_rx[:RL_REPORT_SIZE_58]))
                    if report is not None:
                        del self._binary_rx[:len(report.raw)]
                        return report
                    if len(self._binary_rx) >= RL_REPORT_SIZE_65:
                        report = parse_rl_report(bytes(self._binary_rx[:RL_REPORT_SIZE_65]))
                        if report is not None:
                            del self._binary_rx[:len(report.raw)]
                            return report
                    del self._binary_rx[0]
            return None
        finally:
            self.ser.timeout = old_timeout









