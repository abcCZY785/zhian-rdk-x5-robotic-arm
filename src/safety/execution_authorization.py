"""Explicit execution consent, separate from task data and model output."""

from __future__ import annotations

from typing import Any


class ExecutionConfirmationRequired(PermissionError):
    """The caller did not explicitly authorize this execution request."""


def require_execution_confirmation(mode: str, *, confirm: bool = False) -> None:
    """Fail closed; never coerce strings, numbers, or unknown modes."""
    if mode not in ("plan", "execute"):
        raise ValueError("mode must be plan or execute")
    if type(confirm) is not bool:
        raise ValueError("confirm must be a JSON boolean")
    if mode == "execute" and confirm is not True:
        raise ExecutionConfirmationRequired("execute requires explicit confirm=true")


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous duplicate keys, including conflicting consent fields."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result
