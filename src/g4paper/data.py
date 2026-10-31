"""Data loading utilities for g4paper.

Provides functions to load model submission patches and SWE‑bench issue metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def load_submission(dir_path: str) -> Dict[int, Dict[str, Any]]:
    """Load a submission directory.

    Args:
        dir_path: Path to a directory containing ``all_preds.jsonl`` and a
            ``results/results.json`` file.
    Returns:
        Mapping from ``instance_id`` to a dict with keys ``"patch"`` (str) and
        ``"resolved"`` (bool).  Entries with missing or empty ``model_patch`` are omitted.
    Raises:
        FileNotFoundError: If required files are missing.
        JSONDecodeError: If files contain invalid JSON.
    """
    base = Path(dir_path)
    preds_path = base / "all_preds.jsonl"
    results_path = base / "results" / "results.json"

    preds = {}
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            instance_id = entry["instance_id"]
            # SPEC: skip if model_patch is missing or empty
            if "model_patch" not in entry:
                continue
            model_patch = entry["model_patch"]
            if model_patch is None or model_patch == "":
                continue
            preds[instance_id] = {"patch": model_patch}

    with open(results_path, "r", encoding="utf-8") as f:
        results_data = json.load(f)
    resolved_ids = set(results_data.get("resolved", []))

    result: Dict[int, Dict[str, Any]] = {}
    for instance_id, data in preds.items():
        result[instance_id] = {
            "patch": data["patch"],
            "resolved": instance_id in resolved_ids,
        }
    return result


def load_issues(jsonl_path: str) -> Dict[int, Dict[str, Any]]:
    """Load a SWE‑bench Verified JSONL file.

    Args:
        jsonl_path: Path to a JSONL file where each line contains an ``instance_id``
            and ``repo`` / ``problem_statement`` fields.
    Returns:
        Mapping from ``instance_id`` to a dict with keys ``"repo"`` and ``"problem_statement"``.
    Raises:
        FileNotFoundError: If the file does not exist.
        JSONDecodeError: If a line contains invalid JSON.
    """
    path = Path(jsonl_path)
    result: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            instance_id = entry["instance_id"]
            result[instance_id] = {
                "repo": entry["repo"],
                "problem_statement": entry["problem_statement"],
            }
    return result
