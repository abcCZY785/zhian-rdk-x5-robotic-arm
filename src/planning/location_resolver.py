"""Resolve symbolic task locations into vision detections or fixed V8 coordinates."""

from __future__ import annotations

from typing import Any

from .area_registry import WORK_AREAS, fixed_target_for_area

TABLE_ZONE_BOUNDS = {
    "config_table": (0.0, 1.0),
    "table_left": (0.0, 1.0 / 3.0),
    "table_center": (1.0 / 3.0, 2.0 / 3.0),
    "table_right": (2.0 / 3.0, 1.0),
}

AREA_BOUNDS = WORK_AREAS

CLASS_GRASP_Z = {
    "auto": 0.220,
    "beaker": 0.220,
    "brown_bottle": 0.220,
    "transparent_bottle": 0.220,
}

CLASS_BASE_XY_OFFSET = {
    "auto": (0.0, 0.0),
    "beaker": (-0.05, -0.06),
    "brown_bottle": (0.0, -0.05),
    "transparent_bottle": (0.0, -0.05),
}

FIXED_TARGETS = {
    "config_table": [0.00, -0.45, 0.22],
    "table_left": [-0.06, -0.45, 0.22],
    "table_center": [0.00, -0.45, 0.22],
    "table_right": [0.06, -0.45, 0.22],
    "stand_lower": fixed_target_for_area("stand_lower"),
    "stand_upper": fixed_target_for_area("stand_upper"),
    "waste_bucket_hover": fixed_target_for_area("waste_bucket_hover"),
}


class LocationResolutionError(RuntimeError):
    """Raised when a symbolic location cannot be resolved."""


def _zone_match(source: str, x_norm: float) -> bool:
    if source == "vision_selected":
        return True
    low, high = TABLE_ZONE_BOUNDS[source]
    if source == "table_right":
        return low <= x_norm <= high
    return low <= x_norm < high


def choose_detection(
    detections: list[dict[str, Any]],
    *,
    object_label: str,
    source: str,
    base_corners: list[list[float]] | None = None,
) -> dict[str, Any]:
    if object_label == "auto":
        candidates = [det for det in detections if str(det.get("label")) in {"beaker", "brown_bottle", "transparent_bottle"}]
    else:
        candidates = [det for det in detections if str(det.get("label")) == object_label]
    if not candidates:
        raise LocationResolutionError(f"no detection found for object_label={object_label}")

    if source in TABLE_ZONE_BOUNDS:
        zoned = []
        for det in candidates:
            table_xy_norm = det.get("table_xy_norm")
            if not isinstance(table_xy_norm, list) or len(table_xy_norm) < 2:
                continue
            if _zone_match(source, float(table_xy_norm[0])):
                zoned.append(det)
        candidates = zoned
        if not candidates:
            raise LocationResolutionError(f"no {object_label} detection found in source zone {source}")
    elif source != "vision_selected":
        raise LocationResolutionError(f"source location {source} is not vision-resolvable in the minimal executor")

    candidates = sorted(candidates, key=lambda det: float(det.get("confidence", 0.0)), reverse=True)
    chosen = dict(candidates[0])
    base_xyz = chosen.get("base_xyz")
    if (not isinstance(base_xyz, list) or len(base_xyz) != 3) and base_corners is not None:
        table_xy_norm = chosen.get("table_xy_norm")
        if isinstance(table_xy_norm, list) and len(table_xy_norm) >= 2:
            chosen["base_xyz"] = project_norm_to_base_xyz(
                [float(table_xy_norm[0]), float(table_xy_norm[1])],
                base_corners,
                object_label,
            )
            base_xyz = chosen["base_xyz"]
    if not isinstance(base_xyz, list) or len(base_xyz) != 3:
        raise LocationResolutionError("chosen detection has no base_xyz and could not be projected")
    return chosen


def resolve_place_target(symbolic_target: str) -> list[float]:
    if symbolic_target not in FIXED_TARGETS:
        raise LocationResolutionError(f"unsupported place_target={symbolic_target}")
    return [float(v) for v in FIXED_TARGETS[symbolic_target]]


def project_norm_to_base_xyz(
    norm_xy: list[float],
    base_corners: list[list[float]],
    object_label: str,
) -> list[float]:
    tl, tr, br, bl = base_corners
    u, v = float(norm_xy[0]), float(norm_xy[1])

    top_x = tl[0] + (tr[0] - tl[0]) * u
    top_y = tl[1] + (tr[1] - tl[1]) * u
    bot_x = bl[0] + (br[0] - bl[0]) * u
    bot_y = bl[1] + (br[1] - bl[1]) * u

    x = top_x + (bot_x - top_x) * v
    y = top_y + (bot_y - top_y) * v
    offset_xy = CLASS_BASE_XY_OFFSET.get(object_label, (0.0, 0.0))
    z = CLASS_GRASP_Z.get(object_label, 0.160)
    return [
        round(x + float(offset_xy[0]), 6),
        round(y + float(offset_xy[1]), 6),
        round(z, 6),
    ]
