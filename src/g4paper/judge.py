"""Judgment utilities for g4paper.

Functions to build LLM prompts, parse their JSON responses, and run batch judgments.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

def build_messages(
    problem_statement: str, patch: str, max_chars: int = 12000
) -> List[Dict[str, Any]]:
    """
    Construct system and user messages for a judge.

    Each of ``problem_statement`` and ``patch`` is truncated to
    ``max_chars // 2`` characters.  If truncation occurs the marker
    ``…[truncated]…`` is appended to the truncated text.

    Returns a list ``[system_msg, user_msg]`` where each element is a dict
    with at least the keys ``"role"`` and ``"content"``.
    """

    limit = max_chars // 2

    def _maybe_truncate(text: str) -> str:
        if len(text) > limit:
            return text[: limit - len("...[truncated]...")] + "...[truncated]..."
        return text

    truncated_problem = _maybe_truncate(problem_statement)
    truncated_patch = _maybe_truncate(patch)

    system_msg: Dict[str, Any] = {
        "role": "system",
        "content": "You are an impartial judge evaluating whether a code patch resolves the given issue.",
    }

    user_msg: Dict[str, Any] = {
        "role": "user",
        "content": (
            "Please evaluate whether the following code patch resolves the issue described.\n\n"
            "Issue description:\n"
            f"{truncated_problem}\n\n"
            "Patch:\n"
            f"{truncated_patch}\n\n"
            "Respond with a JSON object containing the keys p_resolved (a float between 0 and 1) "
            "and reason (a string explaining your judgment)."
        ),
    }

    return [system_msg, user_msg]

def parse_judgment(text: str) -> Tuple[float | None, str]:
    """
    Extract the first JSON object that contains a ``p_resolved`` key.

    The extracted probability is clamped to the interval [0.0, 1.0].
    If no suitable JSON object can be found, ``(None, "unparseable")`` is returned.
    """
    n = len(text)
    i = 0
    while i < n:
        if text[i] != '{':
            i += 1
            continue
        depth = 0
        j = i
        in_string = False
        escaped = False
        while j < n:
            c = text[j]
            if not escaped and c == '\\':
                escaped = True
            elif escaped:
                escaped = False
            elif c == '"' and not escaped:
                in_string = not in_string
            elif not in_string:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[i:j + 1]
                        try:
                            data = json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
                        else:
                            if isinstance(data, dict) and "p_resolved" in data:
                                raw_p = data["p_resolved"]
                                try:
                                    p = float(raw_p)
                                except (TypeError, ValueError):
                                    p = None
                                else:
                                    p = max(0.0, min(1.0, p))
                                reason = data.get("reason", "")
                                if not isinstance(reason, str):
                                    reason = str(reason)
                                return p, reason
                        i = j
                        break
            j += 1
        i += 1
    return None, "unparseable"

def run_judgments(
    items: List[Dict[str, Any]],
    judges: List[str],
    ask_fns: Dict[str, Callable[[List[Dict[str, Any]]], str]],
    out_path: str,
    rate_limits: Dict[str, float],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """
    Run judgments for each (item, judge) pair not already recorded.

    Results are appended to ``out_path`` as JSON lines with the keys:
    ``submission``, ``instance_id``, ``judge``, ``p``, ``reason``, ``error``.
    Calls respect ``rate_limits`` (minimum seconds between calls per judge) using
    the injected ``sleep`` function.  Errors are captured in the ``error`` field
    and do not abort the overall run.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    recorded: set[Tuple[Any, Any, Any]] = set()
    if out_file.is_file():
        with out_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (
                    record.get("submission"),
                    record.get("instance_id"),
                    record.get("judge"),
                )
                if None not in key:
                    recorded.add(key)

    last_called: Dict[str, float] = {}

    with out_file.open("a", encoding="utf-8") as f_append:
        for item in items:
            submission = item.get("submission")
            instance_id = item.get("instance_id")
            problem = item.get("problem_statement", "")
            patch_text = item.get("patch", "")
            if submission is None or instance_id is None:
                continue
            for judge in judges:
                key = (submission, instance_id, judge)
                if key in recorded:
                    continue

                now = time.time()
                elapsed = now - last_called.get(judge, 0.0)
                min_gap = rate_limits.get(judge, 0.0)
                if elapsed < min_gap:
                    sleep(min_gap - elapsed)

                messages = build_messages(problem, patch_text)
                try:
                    response = ask_fns[judge](messages)
                    p, reason = parse_judgment(response)
                    error = None
                except Exception as exc:
                    p = None
                    reason = ""
                    error = str(exc)

                row = {
                    "submission": submission,
                    "instance_id": instance_id,
                    "judge": judge,
                    "p": p,
                    "reason": reason,
                    "error": error,
                }
                f_append.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_append.flush()

                last_called[judge] = time.time()
                recorded.add(key)
