#!/usr/bin/env python3
"""Minimal local HTTP bridge for the V8/V8.1 demo page."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.planning.location_resolver import choose_detection, resolve_place_target
from src.planning.task_schema import PlannedTask
from src.safety.execution_authorization import (
    ExecutionConfirmationRequired,
    require_execution_confirmation,
    unique_json_object,
)

ROOT = Path(__file__).resolve().parents[2]
# TODO: Set YOLO_PYTHON to the Python interpreter with ultralytics installed.
# Example: export YOLO_PYTHON=/path/to/yolo-venv/bin/python
YOLO_PYTHON = Path(os.getenv("YOLO_PYTHON", "python"))
PY313 = Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python313" / "python.exe"
PLAN_SCRIPT = ROOT / "src" / "planning" / "text_experiment_to_v8.py"
VISION_LOCATOR = ROOT / "src" / "perception" / "vision_table_yolo_locator.py"
BUILD_TASK = ROOT / "build_pick_place_task_v7.py"
LATEST_STATE = ROOT / "logs" / "assistant_web_latest.json"
PUBLIC_VISION_DIR = ROOT / "urdf_learn" / "playground" / "playground" / "public" / "vision"
ASSISTANT_PREVIEW = PUBLIC_VISION_DIR / "assistant_latest.jpg"
DEFAULT_PORT = 8766
DEFAULT_CAMERA_INDEX = 2
DEFAULT_PORT_NAME = "COM10"
DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5-pro-ultraspeed"
# TODO: Set RL_SNAPSHOT_DIR to the trained RL model snapshot directory.
# Example: export RL_SNAPSHOT_DIR=/path/to/snapshots/accepted_v8_1_...
DEFAULT_RL_DIR = os.getenv("RL_SNAPSHOT_DIR", "models/weights/rl_snapshot")
DEFAULT_BASE_CORNERS = "0.12,-0.50;-0.12,-0.50;-0.12,-0.30;0.12,-0.30"

STATE_LOCK = threading.Lock()

TEMPLATES = {
    "demo_pour_beaker_left_to_table_right": {
        "intent": "pick_pour_place",
        "object_label": "beaker",
        "source": "table_left",
        "place_target": "table_right",
        "pour_target": "waste_bucket_hover",
        "pour_seconds": 2.0,
        "instruction": "把配置台左边那个烧杯拿起来，倒到废液桶上方 2 秒，再放到配置台右侧",
    },
    "demo_pick_brown_center_to_table_right": {
        "intent": "pick_place",
        "object_label": "brown_bottle",
        "source": "table_center",
        "place_target": "table_right",
        "instruction": "把配置台中间那个棕色试剂瓶拿起来，放到配置台右侧",
    },
    "demo_move_beaker_left_to_bucket_hover": {
        "intent": "pick_place",
        "object_label": "beaker",
        "source": "table_left",
        "place_target": "waste_bucket_hover",
        "instruction": "把左边那个烧杯拿起来，放到废液桶上方",
    },
}


class BridgeError(RuntimeError):
    pass


def normalize_error(message: str) -> str:
    if "object_label=auto" in message:
        return "未能确定目标器材类型。请在指令中明确写出烧杯、棕色试剂瓶或透明试剂瓶。"
    if "no detection found" in message:
        return "当前画面里没有识别到符合条件的目标器材，请检查相机画面和物体位置。"
    if "marker" in message.lower():
        return "当前画面中的标定码不足，无法完成配置台定位。请确保至少能看到三张标定码。"
    return message


def json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.end_headers()
    handler.wfile.write(body)


def load_latest_state() -> dict:
    if not LATEST_STATE.exists():
        return {"status": "idle", "message": "no assistant task yet"}
    return json.loads(LATEST_STATE.read_text(encoding="utf-8"))


def save_latest_state(payload: dict) -> None:
    LATEST_STATE.parent.mkdir(parents=True, exist_ok=True)
    LATEST_STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def derive_preview_url(task_config: str | None) -> str | None:
    if not task_config:
        return None
    cfg = Path(task_config)
    master = cfg.parent / "task_master_ab.json"
    if not master.exists():
        return None
    return f"/trajectories/{cfg.parent.name}/task_master_ab.json"


def sync_preview_image(path_str: str | None) -> str | None:
    if not path_str:
        return None
    src = Path(path_str)
    if not src.exists():
        return None
    PUBLIC_VISION_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, ASSISTANT_PREVIEW)
    return f"/vision/assistant_latest.jpg?t={int(time.time() * 1000)}"


def run_subprocess(cmd: list[str], *, cwd: Path | None = None, label: str) -> None:
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise BridgeError(f"{label} failed: {proc.stderr.strip() or proc.stdout.strip() or proc.returncode}")


def run_mimo_pipeline(
    *, instruction: str, mode: str, camera_index: int, port_name: str, confirm: bool = False
) -> dict:
    require_execution_confirmation(mode, confirm=confirm)
    api_key = os.environ.get("MIMO_API_KEY", "").strip()
    if not api_key:
        raise BridgeError("MIMO_API_KEY not set in bridge process environment")
    if not PY313.exists():
        raise BridgeError(f"python313 not found: {PY313}")

    output = ROOT / "logs" / ("assistant_web_plan.json" if mode == "plan" else "assistant_web_execute.json")
    cmd = [
        str(PY313),
        str(PLAN_SCRIPT),
        instruction,
        "--api-key",
        api_key,
        "--base-url",
        os.environ.get("MIMO_BASE_URL", DEFAULT_BASE_URL),
        "--model",
        os.environ.get("MIMO_MODEL", DEFAULT_MODEL),
        "--camera-index",
        str(camera_index),
        "--task-name",
        "text_experiment_v8_demo",
        "--rl-dir",
        os.environ.get("DEMO_RL_DIR", DEFAULT_RL_DIR),
        "--port",
        port_name,
        "--output",
        str(output),
    ]
    if mode == "plan":
        cmd.append("--build-only")
    elif mode == "execute":
        cmd.append("--confirm")

    run_subprocess(cmd, label="text_experiment_to_v8.py")
    if not output.exists():
        raise BridgeError(f"expected output not found: {output}")

    payload = json.loads(output.read_text(encoding="utf-8"))
    execution = payload.setdefault("execution", {})
    execution["mode"] = mode
    execution["preview_url"] = derive_preview_url(execution.get("task_config"))
    execution["assistant_preview_url"] = sync_preview_image(execution.get("vision_jpg"))
    execution["camera_index"] = camera_index
    execution["serial_port"] = port_name
    payload["status"] = "planned" if mode == "plan" else "executed"
    payload["ok"] = True
    payload["entry"] = "mimo"
    return payload


def run_template_pipeline(
    *, template_id: str, mode: str, camera_index: int, port_name: str, confirm: bool = False
) -> dict:
    require_execution_confirmation(mode, confirm=confirm)
    if template_id not in TEMPLATES:
        raise BridgeError(f"unknown template_id={template_id}")
    if not YOLO_PYTHON.exists():
        raise BridgeError(f"python311 not found: {YOLO_PYTHON}")
    if not PY313.exists():
        raise BridgeError(f"python313 not found: {PY313}")

    template = dict(TEMPLATES[template_id])
    template.setdefault("confirm_before_run", True)
    PlannedTask.from_dict(template)
    task_name = "text_experiment_v8_demo"
    vision_json = ROOT / "logs" / f"{task_name}_vision.json"
    vision_jpg = ROOT / "logs" / f"{task_name}_vision.jpg"
    task_dir = ROOT / "urdf_learn" / "playground" / "playground" / "public" / "trajectories" / f"task_{task_name}"
    task_config = task_dir / "task_config.json"

    vision_cmd = [
        str(YOLO_PYTHON),
        str(VISION_LOCATOR),
        "--camera-index",
        str(camera_index),
        f"--base-corners={DEFAULT_BASE_CORNERS}",
        "--output",
        str(vision_json),
        "--preview",
        str(vision_jpg),
    ]
    run_subprocess(vision_cmd, label="vision_table_yolo_locator.py")
    if not vision_json.exists():
        raise BridgeError(f"vision output missing: {vision_json}")
    vision_payload = json.loads(vision_json.read_text(encoding="utf-8"))
    assistant_preview_url = sync_preview_image(str(vision_jpg))
    detections = vision_payload.get("detections_in_table") or []
    base_corners = [[0.12, -0.50], [-0.12, -0.50], [-0.12, -0.30], [0.12, -0.30]]
    try:
        chosen = choose_detection(
            detections,
            object_label=template["object_label"],
            source=template["source"],
            base_corners=base_corners,
        )
    except Exception as exc:
        raise BridgeError(str(exc)) from exc
    pick = [float(v) for v in chosen["base_xyz"]]
    place = resolve_place_target(template["place_target"])
    pour = resolve_place_target(template["pour_target"]) if template.get("pour_target") else None

    build_cmd = [
        str(PY313),
        str(BUILD_TASK),
        "--name",
        task_name,
        f"--pick={pick[0]},{pick[1]},{pick[2]}",
        f"--place={place[0]},{place[1]},{place[2]}",
        "--segment-frames",
        "140",
        "--sample-every",
        "2",
        "--gain",
        "0.9",
        "--rate",
        "20.0",
        "--home-reset-speed",
        "40.0",
        "--home-joints",
        "0,0,0,0,0,0",
        "--rl-dir",
        os.environ.get("DEMO_RL_DIR", DEFAULT_RL_DIR),
    ]
    if pour is not None:
        build_cmd.extend([
            f"--pour={pour[0]},{pour[1]},{pour[2]}",
            "--pour-seconds",
            str(float(template.get("pour_seconds", 2.0))),
            "--pour-j5-speed",
            "-0.3",
            "--pour-return-j5-speed",
            "0.3",
            "--pour-rotate-seconds",
            "6.0",
        ])
    run_subprocess(build_cmd, label="build_pick_place_task_v7.py")

    if mode == "execute":
        run_cmd = [
            str(PY313),
            str(ROOT / "src" / "control" / "run_pick_place_task_v7.py"),
            "--task-config",
            str(task_config),
            "--port",
            port_name,
            "--confirm",
        ]
        run_subprocess(run_cmd, label="run_pick_place_task_v7.py")

    payload = {
        "ok": True,
        "status": "planned" if mode == "plan" else "executed",
        "mode": mode,
        "entry": "template",
        "instruction": template["instruction"],
        "task": template,
        "steps": [
            {"title": "视觉识别目标"},
            {"title": "固定模板生成任务"},
            {"title": "浏览器预演轨迹"},
            {"title": "真实机械臂执行" if mode == "execute" else "等待确认执行"},
        ],
        "execution": {
            "mode": mode,
            "camera_index": camera_index,
            "serial_port": port_name,
            "task_config": str(task_config),
            "preview_url": derive_preview_url(str(task_config)),
            "vision_jpg": str(vision_jpg),
            "assistant_preview_url": assistant_preview_url,
        },
        "resolved": {
            "pick": pick,
            "place": place,
            "pour": pour,
            "selected_detection": chosen,
        },
    }
    return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "ZhianDemoBridge/0.2"

    def do_OPTIONS(self) -> None:  # noqa: N802
        json_response(self, HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            json_response(self, HTTPStatus.OK, {"ok": True, "service": "ai_demo_bridge"})
            return
        if parsed.path == "/api/assistant/latest":
            json_response(self, HTTPStatus.OK, load_latest_state())
            return
        json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/assistant/plan", "/api/assistant/execute"}:
            json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length < 0:
                raise ValueError("Content-Length must not be negative")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_json_object)
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")

            # Only the route selects the operation. Task/model fields cannot grant consent.
            mode = "plan" if parsed.path == "/api/assistant/plan" else "execute"
            confirm = body.get("confirm", False)
            require_execution_confirmation(mode, confirm=confirm)

            instruction = body.get("instruction", "")
            template_id = body.get("template_id", "")
            if not isinstance(instruction, str) or not isinstance(template_id, str):
                raise ValueError("instruction and template_id must be strings")
            instruction = instruction.strip()
            template_id = template_id.strip() or None
            if not instruction and not template_id:
                raise ValueError("instruction or template_id is required")
            camera_index = body.get("camera_index", DEFAULT_CAMERA_INDEX)
            if type(camera_index) is not int or camera_index < 0:
                raise ValueError("camera_index must be a non-negative integer")
            port_name = body.get("port", DEFAULT_PORT_NAME)
            if not isinstance(port_name, str) or not port_name.strip():
                raise ValueError("port must be a non-empty string")
        except ExecutionConfirmationRequired as exc:
            json_response(self, HTTPStatus.FORBIDDEN, {"ok": False, "error": str(exc)})
            return
        except (ValueError, UnicodeDecodeError) as exc:
            json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        with STATE_LOCK:
            try:
                save_latest_state({
                    "ok": True,
                    "status": "running",
                    "mode": mode,
                    "instruction": instruction,
                    "template_id": template_id,
                    "message": "planning" if mode == "plan" else "executing",
                })
                if template_id:
                    payload = run_template_pipeline(
                        template_id=template_id,
                        mode=mode,
                        camera_index=camera_index,
                        port_name=port_name,
                        confirm=confirm if mode == "execute" else False,
                    )
                else:
                    payload = run_mimo_pipeline(
                        instruction=instruction,
                        mode=mode,
                        camera_index=camera_index,
                        port_name=port_name,
                        confirm=confirm if mode == "execute" else False,
                    )
                save_latest_state(payload)
                json_response(self, HTTPStatus.OK, payload)
            except Exception as exc:
                error_payload = {
                    "ok": False,
                    "status": "error",
                    "mode": mode,
                    "instruction": instruction,
                    "template_id": template_id,
                    "error": normalize_error(str(exc)),
                    "assistant_preview_url": f"/vision/assistant_latest.jpg?t={int(time.time() * 1000)}" if ASSISTANT_PREVIEW.exists() else None,
                }
                save_latest_state(error_payload)
                json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, error_payload)

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write((fmt % args) + "\n")
        sys.stdout.flush()


def main() -> int:
    port = int(os.environ.get("AI_DEMO_BRIDGE_PORT", str(DEFAULT_PORT)))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[ai_demo_bridge] listening on http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
