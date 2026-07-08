"""Stable symbolic work areas for the V8 text experiment assistant."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RectArea:
    name: str
    x: tuple[float, float]
    y: tuple[float, float]
    z: float

    @property
    def center(self) -> list[float]:
        return [round((self.x[0] + self.x[1]) / 2.0, 6), round((self.y[0] + self.y[1]) / 2.0, 6), self.z]


@dataclass(frozen=True)
class CircleArea:
    name: str
    center_xy: tuple[float, float]
    radius: float
    z: float

    @property
    def center(self) -> list[float]:
        return [self.center_xy[0], self.center_xy[1], self.z]


WORK_AREAS = {
    "config_table": RectArea("config_table", (-0.085, 0.085), (-0.535, -0.365), 0.160),
    "stand_lower": RectArea("stand_lower", (-0.467942, -0.277058), (-0.375359, -0.224737), 0.142),
    "stand_upper": RectArea("stand_upper", (-0.532942, -0.342058), (-0.487942, -0.337321), 0.232),
    "waste_bucket_hover": CircleArea("waste_bucket_hover", (0.250, -0.360), 0.025, 0.260),
}


def fixed_target_for_area(name: str) -> list[float]:
    return WORK_AREAS[name].center
