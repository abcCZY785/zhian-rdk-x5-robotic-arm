#!/usr/bin/env python3
"""Realtime YOLO detector for the confirmed default camera (cam2)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from vision_yolo_bridge import load_yolo, run_detection

ROOT = Path(__file__).resolve().parent
PUBLIC_VISION_DIR = ROOT / "urdf_learn" / "playground" / "playground" / "public" / "vision"
DEFAULT_OUTPUT = PUBLIC_VISION_DIR / "vision_realtime_latest.json"
DEFAULT_SNAPSHOT = PUBLIC_VISION_DIR / "vision_realtime_latest.jpg"
# TODO: Set YOLO_MODEL_PATH to your trained YOLO weights file.
# Example: export YOLO_MODEL_PATH=/path/to/weights/best.pt
DEFAULT_MODEL = Path(os.getenv("YOLO_MODEL_PATH", "models/weights/best.pt"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-preview-every", type=float, default=1.0)
    parser.add_argument("--backend-order", default="default,dshow,msmf")
    return parser


def parse_backend_order(text: str) -> list[tuple[str, int | None]]:
    order: list[tuple[str, int | None]] = []
    for item in [x.strip().lower() for x in text.split(',') if x.strip()]:
        if item == 'default':
            order.append((item, None))
        elif item == 'dshow' and hasattr(cv2, 'CAP_DSHOW'):
            order.append((item, int(cv2.CAP_DSHOW)))
        elif item == 'msmf' and hasattr(cv2, 'CAP_MSMF'):
            order.append((item, int(cv2.CAP_MSMF)))
    if not order:
        order.append(('default', None))
    return order


def open_stream(index: int, width: int, height: int, backend_order: list[tuple[str, int | None]]):
    errors: list[str] = []
    for backend_name, backend_id in backend_order:
        cap = cv2.VideoCapture(index) if backend_id is None else cv2.VideoCapture(index, backend_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cap.isOpened():
            cap.release()
            errors.append(f"{backend_name}: open failed")
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, backend_name, frame
        cap.release()
        errors.append(f"{backend_name}: no frame")
    raise RuntimeError(f"camera {index} stream failed: {'; '.join(errors)}")


def annotate_frame(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    vis = frame.copy()
    colors = {
        'beaker': (0, 0, 255),
        'brown_bottle': (0, 255, 0),
        'transparent_bottle': (255, 0, 0),
    }
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox_xyxy']]
        label = det['label']
        score = det['confidence']
        color = colors.get(label, (0, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{label} {score:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def save_snapshot(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or '.jpg', frame)
    if ok:
        encoded.tofile(str(path))


def main() -> int:
    args = build_parser().parse_args()
    model = load_yolo(args.model)
    backend_order = parse_backend_order(args.backend_order)
    cap, backend_name, _ = open_stream(args.camera_index, args.width, args.height, backend_order)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    last_snapshot_t = 0.0

    print(f"[vision] camera={args.camera_index} backend={backend_name} model={args.model}", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError('camera stream read failed')

            detections = run_detection(model, frame, args.conf)
            payload = {
                'timestamp': time.time(),
                'camera_id': f'usb_cam_{args.camera_index}',
                'camera_backend': backend_name,
                'image_size': [int(frame.shape[1]), int(frame.shape[0])],
                'detections': detections,
                'model': str(args.model),
                'status': 'ok',
            }
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

            annotated = annotate_frame(frame, detections)
            now = time.time()
            if now - last_snapshot_t >= args.save_preview_every:
                save_snapshot(args.snapshot, annotated)
                last_snapshot_t = now

            if args.show:
                cv2.imshow('Zhian Realtime Vision', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
