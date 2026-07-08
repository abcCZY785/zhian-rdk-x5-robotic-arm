#!/usr/bin/env python3
"""Locate tabletop objects with four ArUco markers and the existing YOLO model.

This is a deliberately small integration probe:
- capture one frame from cam2 by default
- detect four tabletop ArUco markers
- run the existing YOLO detector
- keep detections whose center lies inside the marker rectangle
- optionally map image pixels into robot-base XY using a homography

Default marker roles:
  tag36h11 ID 12 = top-left
  tag36h11 ID 13 = top-right
  tag36h11 ID 14 = bottom-right
  tag36h11 ID 15 = bottom-left

Table definition:
  use the outward tag corners instead of marker centers
  id12 uses its bottom-right corner
  id13 uses its bottom-left corner
  id14 uses its top-left corner
  id15 uses its top-right corner
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_yolo_bridge import load_yolo, open_camera, run_detection, save_frame_image

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = Path(
    os.getenv("YOLO_MODEL_PATH", "models/weights/best.pt")
)
DEFAULT_OUTPUT = ROOT / "logs" / "vision_table_yolo_latest.json"
DEFAULT_PREVIEW = ROOT / "logs" / "vision_table_yolo_latest.jpg"
TABLE_SIZE_M = 0.24

ARUCO_DICTS = {
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
}

ROLE_ORDER = ["tl", "tr", "br", "bl"]
DEFAULT_MARKER_ROLES = {"tl": 12, "tr": 13, "br": 14, "bl": 15}

CLASS_GRASP_Z = {
    "beaker": 0.220,
    "brown_bottle": 0.220,
    "transparent_bottle": 0.220,
}

CLASS_BASE_XY_OFFSET = {
    # Config-table beaker grasp was validated on the real arm to need a left/back correction.
    "beaker": (-0.05, -0.06),
    "brown_bottle": (0.0, -0.05),
    "transparent_bottle": (0.0, -0.05),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--aruco-dict", choices=sorted(ARUCO_DICTS), default="tag36h11")
    parser.add_argument(
        "--marker-roles",
        default="tl:12,tr:13,br:14,bl:15",
        help="Marker role mapping, for example tl:12,tr:13,br:14,bl:15",
    )
    parser.add_argument(
        "--base-corners",
        default=None,
        help=(
            "Optional robot-base XY corners in meters, ordered tl,tr,br,bl. "
            "Example: --base-corners=0,-0.55;0.30,-0.55;0.30,-0.25;0,-0.25"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--save-raw", type=Path, default=None)
    parser.add_argument("--no-yolo", action="store_true", help="Only detect tabletop markers")
    return parser


def parse_marker_roles(text: str) -> dict[str, int]:
    roles: dict[str, int] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        role, value = part.split(":", 1)
        role = role.strip().lower()
        if role not in ROLE_ORDER:
            raise ValueError(f"unknown marker role: {role}")
        roles[role] = int(value.strip())
    missing = [role for role in ROLE_ORDER if role not in roles]
    if missing:
        raise ValueError(f"marker roles missing: {missing}")
    return roles


def parse_base_corners(text: str | None) -> np.ndarray | None:
    if not text:
        return None
    points = []
    for part in text.split(";"):
        xy = [float(v.strip()) for v in part.split(",")]
        if len(xy) != 2:
            raise ValueError("each base corner must be x,y")
        points.append(xy)
    if len(points) != 4:
        raise ValueError("base corners must contain four points ordered tl,tr,br,bl")
    return np.asarray(points, dtype=np.float32)


def detect_aruco(frame: np.ndarray, dict_name: str) -> tuple[list[np.ndarray], np.ndarray | None]:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    return corners, ids


def marker_geometry_by_id(corners: list[np.ndarray], ids: np.ndarray | None) -> dict[int, dict[str, np.ndarray]]:
    markers: dict[int, dict[str, np.ndarray]] = {}
    if ids is None:
        return markers
    for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
        pts = np.asarray(marker_corners, dtype=np.float32).reshape(-1, 2)
        markers[int(marker_id)] = {
            "corners": pts,
            "center": pts.mean(axis=0),
        }
    return markers


def order_points_fallback(points: list[np.ndarray]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 4:
        raise ValueError("need at least four markers for fallback ordering")
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]
    tl = pts[int(np.argmin(sums))]
    br = pts[int(np.argmax(sums))]
    tr = pts[int(np.argmax(diffs))]
    bl = pts[int(np.argmin(diffs))]
    return np.asarray([tl, tr, br, bl], dtype=np.float32)


def outward_corner_for_role(role: str, marker: dict[str, np.ndarray]) -> np.ndarray:
    corners = marker["corners"]
    if role == "tl":
        return corners[2]
    if role == "tr":
        return corners[3]
    if role == "br":
        return corners[0]
    if role == "bl":
        return corners[1]
    raise ValueError(f"unknown role: {role}")


def infer_missing_corner(points: dict[str, np.ndarray], missing_role: str) -> np.ndarray:
    if missing_role == "tl":
        return points["tr"] + points["bl"] - points["br"]
    if missing_role == "tr":
        return points["tl"] + points["br"] - points["bl"]
    if missing_role == "br":
        return points["tr"] + points["bl"] - points["tl"]
    if missing_role == "bl":
        return points["tl"] + points["br"] - points["tr"]
    raise ValueError(f"unknown missing role: {missing_role}")


def outer_table_polygon_from_markers(markers: dict[int, dict[str, np.ndarray]], roles: dict[str, int]) -> tuple[np.ndarray | None, list[str]]:
    missing = [role for role, marker_id in roles.items() if marker_id not in markers]
    if len(missing) >= 2:
        centers = [marker["center"] for marker in markers.values()]
        if len(centers) >= 4:
            return order_points_fallback(centers), missing
        return None, missing

    points: dict[str, np.ndarray] = {}
    for role in ROLE_ORDER:
        marker_id = roles[role]
        if marker_id in markers:
            points[role] = outward_corner_for_role(role, markers[marker_id])

    if len(missing) == 1:
        points[missing[0]] = infer_missing_corner(points, missing[0])

    if len(points) != 4:
        return None, missing

    table_uv = np.asarray([points[role] for role in ROLE_ORDER], dtype=np.float32)
    return table_uv, missing


def normalized_table_homography(table_uv: np.ndarray) -> np.ndarray:
    norm_xy = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H, _ = cv2.findHomography(table_uv, norm_xy)
    if H is None:
        raise RuntimeError("failed to build normalized table homography")
    return H


def homography_from_uv_to_base(table_uv: np.ndarray, base_xy: np.ndarray) -> np.ndarray:
    H, _ = cv2.findHomography(table_uv, base_xy)
    if H is None:
        raise RuntimeError("failed to build base homography")
    return H


def transform_point(H: np.ndarray, uv: list[float] | tuple[float, float] | np.ndarray) -> list[float]:
    point = np.asarray([[[float(uv[0]), float(uv[1])]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, H)[0, 0]
    return [round(float(mapped[0]), 6), round(float(mapped[1]), 6)]


def point_inside_polygon(uv: list[float], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon.reshape(-1, 1, 2), (float(uv[0]), float(uv[1])), False) >= 0


def enrich_detections(detections: list[dict[str, Any]], table_uv: np.ndarray | None, base_H: np.ndarray | None) -> list[dict[str, Any]]:
    if table_uv is None:
        return detections
    norm_H = normalized_table_homography(table_uv)
    enriched = []
    for det in detections:
        point_uv = det.get("pick_uv") or det.get("center_uv")
        if point_uv is None:
            continue
        if not point_inside_polygon(point_uv, table_uv):
            continue
        det = dict(det)
        if "center_uv" in det and isinstance(det["center_uv"], list) and len(det["center_uv"]) >= 2:
            det["center_uv"] = [round(float(det["center_uv"][0]), 3), round(float(det["center_uv"][1]), 3)]
        det["pick_uv"] = [round(float(point_uv[0]), 3), round(float(point_uv[1]), 3)]
        det["table_xy_norm"] = transform_point(norm_H, point_uv)
        det["table_xy_m"] = [
            round(det["table_xy_norm"][0] * TABLE_SIZE_M, 6),
            round(det["table_xy_norm"][1] * TABLE_SIZE_M, 6),
        ]
        if base_H is not None:
            base_xy = transform_point(base_H, point_uv)
            offset_xy = CLASS_BASE_XY_OFFSET.get(str(det.get("label")), (0.0, 0.0))
            grasp_z = CLASS_GRASP_Z.get(str(det.get("label")), 0.160)
            det["base_xyz"] = [
                round(base_xy[0] + float(offset_xy[0]), 6),
                round(base_xy[1] + float(offset_xy[1]), 6),
                round(grasp_z, 6),
            ]
        enriched.append(det)
    return enriched


def annotate(frame: np.ndarray, table_uv: np.ndarray | None, detections: list[dict[str, Any]], markers: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
    vis = frame.copy()
    for marker_id, marker in markers.items():
        center = marker["center"]
        x, y = int(center[0]), int(center[1])
        cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(vis, f"id {marker_id}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.polylines(vis, [marker["corners"].astype(np.int32).reshape(-1, 1, 2)], True, (0, 200, 200), 1)
    if table_uv is not None:
        cv2.polylines(vis, [table_uv.astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 2)
        for role, pt in zip(ROLE_ORDER, table_uv):
            cv2.putText(vis, role, (int(pt[0]) + 8, int(pt[1]) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    colors = {
        "beaker": (0, 0, 255),
        "brown_bottle": (0, 255, 0),
        "transparent_bottle": (255, 0, 0),
    }
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        label = str(det["label"])
        color = colors.get(label, (0, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if "pick_uv" in det:
            pu, pv = [int(v) for v in det["pick_uv"]]
            cv2.circle(vis, (pu, pv), 5, color, -1)
        text = f"{label} {float(det['confidence']):.2f}"
        if "table_xy_m" in det:
            txy = det["table_xy_m"]
            text += f" [{txy[0]:.3f},{txy[1]:.3f}]m"
        if "base_xyz" in det:
            xyz = det["base_xyz"]
            text += f" [{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}]"
        cv2.putText(vis, text, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return vis


def main() -> int:
    args = build_parser().parse_args()
    roles = parse_marker_roles(args.marker_roles)
    base_corners = parse_base_corners(args.base_corners)

    frame, backend_name = open_camera(args.camera_index, args.width, args.height)
    if args.save_raw is not None:
        args.save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_frame_image(args.save_raw, frame)

    corners, ids = detect_aruco(frame, args.aruco_dict)
    markers = marker_geometry_by_id(corners, ids)
    table_uv, missing_roles = outer_table_polygon_from_markers(markers, roles)

    base_H = None
    if table_uv is not None and base_corners is not None:
        base_H = homography_from_uv_to_base(table_uv, base_corners)

    detections: list[dict[str, Any]] = []
    if not args.no_yolo:
        model = load_yolo(args.model)
        detections = run_detection(model, frame, args.conf)
    table_detections = enrich_detections(detections, table_uv, base_H)

    payload: dict[str, Any] = {
        "timestamp": time.time(),
        "status": "ok" if table_uv is not None else "marker_incomplete",
        "camera_id": f"usb_cam_{args.camera_index}",
        "camera_backend": backend_name,
        "image_size": [int(frame.shape[1]), int(frame.shape[0])],
        "table_size_m": TABLE_SIZE_M,
        "aruco_dict": args.aruco_dict,
        "marker_roles": roles,
        "markers_found": {
            str(marker_id): {
                "center": [round(float(marker["center"][0]), 3), round(float(marker["center"][1]), 3)],
                "corners": [[round(float(x), 3), round(float(y), 3)] for x, y in marker["corners"]],
            }
            for marker_id, marker in markers.items()
        },
        "missing_marker_roles": missing_roles,
        "table_uv_tl_tr_br_bl": None if table_uv is None else [[round(float(x), 3), round(float(y), 3)] for x, y in table_uv],
        "base_corners_tl_tr_br_bl": None if base_corners is None else base_corners.tolist(),
        "detections_all": detections,
        "detections_in_table": table_detections,
        "selected_target": table_detections[0] if table_detections else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    preview = annotate(frame, table_uv, table_detections, markers)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    save_frame_image(args.preview, preview)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
