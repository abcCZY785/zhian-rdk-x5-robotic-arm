"""Prompting and JSON parsing for the minimal text experiment assistant."""

from __future__ import annotations

import json
import re
from typing import Any

from .mimo_client import MimoClient
from .task_schema import PlannedTask


SYSTEM_PROMPT = """You are a robotics task planner.
Convert the user's Chinese instruction into one JSON object only.

Rules:
- Supported intents: pick_place, pick_pour_place
- Supported object_label values: auto, beaker, brown_bottle, transparent_bottle
- Primary area targets for task generation:
  config_table, stand_lower, stand_upper, waste_bucket_hover
- Supported source/place/pour targets:
  vision_selected, config_table, table_left, table_center, table_right, stand_lower, stand_upper, waste_bucket_hover
- Area coordinate constraints in robot base frame:
  config_table: x[-0.085,0.085], y[-0.535,-0.365], z=0.160
  stand_lower: x[-0.467942,-0.277058], y[-0.375359,-0.224737], z=0.142
  stand_upper: x[-0.532942,-0.342058], y[-0.487942,-0.337321], z=0.232
  waste_bucket_hover: center(0.250,-0.360), r=0.025, z=0.190
- Prefer the four primary area targets for place_target and pour_target.
- If the user says left beaker on the table, source should be table_left.
- If the user says right side of the table, place_target should be table_right.
- If the user says configuration table or table without side detail, use config_table.
- If the user says lower stand/lower shelf/lower platform, use stand_lower.
- If the user says upper stand/upper shelf/upper platform, use stand_upper.
- If the user says above the waste bucket, use waste_bucket_hover.
- If the user says pour for N seconds, set intent=pick_pour_place and pour_seconds=N.
- If the user only asks for pick and place, set intent=pick_place.
- If the user asks to dump/discard/pour away a solution but does not specify the vessel type, object_label can be auto.
- Always set confirm_before_run=true.
- Return JSON only, no markdown, no explanation.

JSON schema:
{
  "intent": "pick_place" | "pick_pour_place",
  "object_label": "auto" | "beaker" | "brown_bottle" | "transparent_bottle",
  "source": "vision_selected" | "config_table" | "table_left" | "table_center" | "table_right" | "stand_lower" | "stand_upper" | "waste_bucket_hover",
  "place_target": "vision_selected" | "config_table" | "table_left" | "table_center" | "table_right" | "stand_lower" | "stand_upper" | "waste_bucket_hover",
  "pour_target": "waste_bucket_hover" | "config_table" | "table_left" | "table_center" | "table_right" | "stand_lower" | "stand_upper" | null,
  "pour_seconds": number | null,
  "confirm_before_run": true,
  "notes": string
}
"""


def _parse_pour_seconds(instruction: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*秒", instruction)
    if match:
        return float(match.group(1))
    return 2.0


def _detect_object_label(instruction: str) -> str:
    text = instruction.lower()
    if "烧杯" in instruction or "beaker" in text:
        return "beaker"
    if "棕色试剂瓶" in instruction or "brown_bottle" in text or "棕瓶" in instruction:
        return "brown_bottle"
    if "透明试剂瓶" in instruction or "transparent_bottle" in text or "透明瓶" in instruction:
        return "transparent_bottle"
    return "auto"


def _detect_source(instruction: str) -> str:
    if "左边" in instruction or "左侧" in instruction:
        return "table_left"
    if "右边" in instruction or "右侧" in instruction:
        return "table_right"
    if "中间" in instruction or "中心" in instruction:
        return "table_center"
    if "下层" in instruction or "下台阶" in instruction or "一级台阶" in instruction:
        return "stand_lower"
    if "上层" in instruction or "上台阶" in instruction or "二级台阶" in instruction:
        return "stand_upper"
    return "config_table"


def _looks_like_discard_request(instruction: str) -> bool:
    keywords = ("倒掉", "倒掉吧", "废液桶", "处理掉", "倒了", "清掉", "清理掉")
    return any(word in instruction for word in keywords)


def try_rule_based_instruction(instruction: str) -> PlannedTask | None:
    text = instruction.strip()
    if not text:
        return None
    if not _looks_like_discard_request(text):
        return None
    return PlannedTask(
        intent="pick_pour_place",
        object_label=_detect_object_label(text),
        source=_detect_source(text),
        place_target="table_center",
        pour_target="waste_bucket_hover",
        pour_seconds=_parse_pour_seconds(text),
        confirm_before_run=True,
        notes="rule_based_discard_template",
    )


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = next((part for part in parts if "{" in part), text).strip()
        if text.startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"no json object found in response: {text[:400]}")
    return json.loads(text[start : end + 1])


def plan_text_instruction(client: MimoClient | None, instruction: str) -> PlannedTask:
    shortcut = try_rule_based_instruction(instruction)
    if shortcut is not None:
        return shortcut
    if client is None:
        raise ValueError("MIMO client is required for non-template instructions")
    response = client.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction.strip()},
        ],
        temperature=0.1,
    )
    choices = response.get("choices") or []
    if not choices:
        raise ValueError(f"no choices in response: {response}")
    content = choices[0].get("message", {}).get("content", "")
    payload = extract_json_object(content)
    return PlannedTask.from_dict(payload)
