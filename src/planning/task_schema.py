"""Structured task schema for the minimal text experiment assistant."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SUPPORTED_INTENTS = {"pick_place", "pick_pour_place"}
SUPPORTED_OBJECTS = {"auto", "beaker", "brown_bottle", "transparent_bottle"}
SUPPORTED_LOCATIONS = {
    "vision_selected",
    "config_table",
    "table_left",
    "table_center",
    "table_right",
    "stand_lower",
    "stand_upper",
    "waste_bucket_hover",
}


@dataclass
class PlannedTask:
    intent: str
    object_label: str
    source: str
    place_target: str
    pour_target: str | None = None
    pour_seconds: float | None = None
    confirm_before_run: bool = True
    notes: str | None = None

    def __post_init__(self) -> None:
        # This is a mandatory policy, never evidence of operator consent.
        if self.confirm_before_run is not True:
            raise ValueError("confirm_before_run must be the boolean true")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlannedTask":
        if not isinstance(payload, dict):
            raise ValueError("task must be a JSON object")
        confirm_before_run = payload.get("confirm_before_run")
        if confirm_before_run is not True:
            raise ValueError("confirm_before_run is required and must be the boolean true")
        intent = str(payload.get("intent", "")).strip()
        object_label = str(payload.get("object_label", "")).strip()
        source = str(payload.get("source", "")).strip()
        place_target = str(payload.get("place_target", "")).strip()
        pour_target = payload.get("pour_target")
        pour_seconds = payload.get("pour_seconds")
        notes = payload.get("notes")

        if intent not in SUPPORTED_INTENTS:
            raise ValueError(f"unsupported intent: {intent}")
        if object_label not in SUPPORTED_OBJECTS:
            raise ValueError(f"unsupported object_label: {object_label}")
        if source not in SUPPORTED_LOCATIONS:
            raise ValueError(f"unsupported source: {source}")
        if place_target not in SUPPORTED_LOCATIONS:
            raise ValueError(f"unsupported place_target: {place_target}")

        if intent == "pick_pour_place":
            if not pour_target:
                raise ValueError("pick_pour_place requires pour_target")
            if str(pour_target) not in SUPPORTED_LOCATIONS:
                raise ValueError(f"unsupported pour_target: {pour_target}")
            if pour_seconds is None:
                raise ValueError("pick_pour_place requires pour_seconds")
            pour_seconds = float(pour_seconds)
            if pour_seconds <= 0:
                raise ValueError("pour_seconds must be > 0")
        else:
            pour_target = None
            pour_seconds = None

        return cls(
            intent=intent,
            object_label=object_label,
            source=source,
            place_target=place_target,
            pour_target=None if pour_target is None else str(pour_target),
            pour_seconds=pour_seconds,
            confirm_before_run=confirm_before_run,
            notes=None if notes is None else str(notes),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskStep:
    action: str
    target: str | None = None
    object_label: str | None = None
    seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}
