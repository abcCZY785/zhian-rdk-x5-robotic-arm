"""Template expansion for the minimal text experiment assistant."""

from __future__ import annotations

from typing import Any

from .task_schema import PlannedTask, TaskStep


def expand_plan(task: PlannedTask) -> list[TaskStep]:
    steps = [
        TaskStep(action="pick", target=task.source, object_label=task.object_label),
    ]
    if task.intent == "pick_pour_place":
        steps.extend(
            [
                TaskStep(action="move", target=task.pour_target),
                TaskStep(action="pour", target=task.pour_target, seconds=task.pour_seconds),
            ]
        )
    steps.extend(
        [
            TaskStep(action="place", target=task.place_target),
            TaskStep(action="home"),
        ]
    )
    return steps


def execution_readiness(task: PlannedTask) -> dict[str, Any]:
    missing_capabilities: list[str] = []
    executable_now = True

    return {
        "executable_now": executable_now,
        "missing_capabilities": missing_capabilities,
        "recommended_executor": "prepare_v8_demo.py" if task.intent == "pick_place" else "text_experiment_to_v8.py",
    }

