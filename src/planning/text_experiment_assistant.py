#!/usr/bin/env python3
"""Minimal text experiment assistant powered by Xiaomi MIMO."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lab_assistant.mimo_client import MimoClient, MimoClientError
from lab_assistant.task_planner import plan_text_instruction
from lab_assistant.templates import execution_readiness, expand_plan

DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5-pro-ultraspeed"
DEFAULT_OUTPUT = Path("logs/text_experiment_assistant_latest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction", help="Chinese experiment instruction")
    parser.add_argument("--api-key", default=os.environ.get("MIMO_API_KEY", ""))
    parser.add_argument("--base-url", default=os.environ.get("MIMO_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("MIMO_MODEL", DEFAULT_MODEL))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = None
    if args.api_key:
        client = MimoClient(api_key=args.api_key, base_url=args.base_url, model=args.model)
    try:
        task = plan_text_instruction(client, args.instruction)
    except (MimoClientError, ValueError) as exc:
        raise SystemExit(f"planner failed: {exc}") from exc

    steps = [step.to_dict() for step in expand_plan(task)]
    readiness = execution_readiness(task)
    payload = {
        "instruction": args.instruction,
        "planner": {
            "base_url": args.base_url,
            "model": args.model,
        },
        "task": task.to_dict(),
        "steps": steps,
        "execution": readiness,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
