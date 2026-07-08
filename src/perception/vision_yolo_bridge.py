#!/usr/bin/env python3
"""Vision bridge for Zhian visual grasp pipeline.

Responsibilities:
1. probe local cameras
2. capture a frame from a chosen camera
3. run the existing YOLO detector when the local Python env is valid
4. emit a normalized JSON detection payload for downstream fusion

This is the first integration bridge between the existing YOLO project and the
arm-side planner. It intentionally stops at 2D detections.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
# TODO: Set YOLO_ROOT to your YOLO training project root.
# Example: export YOLO_ROOT=/path/to/yolo_training
YOLO_ROOT = Path(os.getenv("YOLO_ROOT", "."))
DEFAULT_MODEL = Path(os.getenv("YOLO_MODEL_PATH", "models/weights/best.pt"))
DEFAULT_OUTPUT = ROOT / "logs" / "vision_detection_latest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-cameras", action="store_true", help="Probe local camera indices and print results")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--save-frame", type=Path, default=None, help="Optional path to save captured raw frame")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cameras", type=int, default=5)
    return parser


def camera_backends() -> list[tuple[str, int | None]]:
    backends: list[tuple[str, int | None]] = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(("dshow", int(cv2.CAP_DSHOW)))
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(("msmf", int(cv2.CAP_MSMF)))
    backends.append(("default", None))
    return backends


def try_capture(index: int, width: int, height: int, backend_id: int | None):
    cap = cv2.VideoCapture(index) if backend_id is None else cv2.VideoCapture(index, backend_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    opened = bool(cap.isOpened())
    if not opened:
        cap.release()
        return False, None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return True, None
    return True, frame


def probe_cameras(max_cameras: int, width: int, height: int) -> list[dict]:
    results: list[dict] = []
    for index in range(max_cameras):
        opened = False
        frame_ok = False
        shape = None
        backend_used = None
        for backend_name, backend_id in camera_backends():
            opened, frame = try_capture(index, width, height, backend_id)
            if frame is not None:
                frame_ok = True
                shape = list(frame.shape)
                backend_used = backend_name
                break
        results.append(
            {
                "index": index,
                "opened": opened,
                "frame_ok": frame_ok,
                "shape": shape,
                "backend": backend_used,
            }
        )
    return results


def open_camera(index: int, width: int, height: int):
    errors: list[str] = []
    for backend_name, backend_id in camera_backends():
        opened, frame = try_capture(index, width, height, backend_id)
        if frame is not None:
            return frame, backend_name
        if opened:
            errors.append(f"{backend_name}: no frame")
        else:
            errors.append(f"{backend_name}: open failed")
    raise RuntimeError(f"camera {index} capture failed: {'; '.join(errors)}")


def save_frame_image(path: Path, frame: np.ndarray) -> None:
    suffix = path.suffix.lower() or ".jpg"
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    ok, encoded = cv2.imencode(ext, frame)
    if not ok:
        raise RuntimeError(f"failed to encode frame as {ext}")
    encoded.tofile(str(path))


def load_yolo(model_path: Path):
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "failed to import ultralytics YOLO; local Python env is inconsistent. "
            "This machine currently appears to have a torch/torchvision mismatch."
        ) from exc

    try:
        return YOLO(str(model_path))
    except Exception as exc:
        raise RuntimeError(f"failed to load model: {model_path}") from exc


def run_detection(model, frame: np.ndarray, conf: float) -> list[dict]:
    result = model.predict(frame, conf=conf, show=False, verbose=False)[0]
    detections: list[dict] = []

    if result.boxes is None:
        return detections

    names = result.names
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
    cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
    scores = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)

    for bbox, class_id, score in zip(xyxy, cls, scores):
        x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
        center_u = (x1 + x2) * 0.5
        center_v = (y1 + y2) * 0.5
        pick_u = center_u
        pick_v = y2 - (y2 - y1) * 0.2
        detections.append(
            {
                "label": str(names[int(class_id)]),
                "confidence": round(float(score), 6),
                "bbox_xyxy": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                "center_uv": [round(center_u, 3), round(center_v, 3)],
                "pick_uv": [round(pick_u, 3), round(pick_v, 3)],
                "pick_strategy": "bbox_lower_center_20pct",
            }
        )
    return detections


def main() -> int:
    args = build_parser().parse_args()

    if args.list_cameras:
        payload = {
            "timestamp": time.time(),
            "cameras": probe_cameras(args.max_cameras, args.width, args.height),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    frame, backend_name = open_camera(args.camera_index, args.width, args.height)
    if args.save_frame is not None:
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        save_frame_image(args.save_frame, frame)

    payload = {
        "timestamp": time.time(),
        "camera_id": f"usb_cam_{args.camera_index}",
        "camera_backend": backend_name,
        "image_size": [int(frame.shape[1]), int(frame.shape[0])],
        "detections": [],
    }

    try:
        model = load_yolo(args.model)
        payload["detections"] = run_detection(model, frame, args.conf)
        payload["model"] = str(args.model)
        payload["status"] = "ok"
    except Exception as exc:
        payload["status"] = "env_error"
        payload["error"] = str(exc)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
