from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from utils.utils import DEFAULT_DATASETS, DEFAULT_OUTPUTS_ROOT, write_jsonl


TYPE2_FILENAME = "Type2.jsonl"
TYPE3_FILENAME = "Type3.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def turn_keys(cases: list[dict[str, Any]]) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for case in cases:
        case_id = case.get("id")
        if case_id is None:
            continue
        for turn in case.get("cheating_turn") or []:
            turn_id = turn.get("turn")
            if turn_id is not None:
                keys.add((int(case_id), int(turn_id)))
    return keys


def remove_overlapping_turns(
    type2_cases: list[dict[str, Any]],
    type3_keys: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    filtered_cases: list[dict[str, Any]] = []

    for case in type2_cases:
        case_id = case.get("id")
        if case_id is None:
            filtered_cases.append(case)
            continue

        new_case = deepcopy(case)
        remaining_turns = []
        for turn in new_case.get("cheating_turn") or []:
            turn_id = turn.get("turn")
            if turn_id is not None and (int(case_id), int(turn_id)) in type3_keys:
                continue
            remaining_turns.append(turn)

        if remaining_turns:
            new_case["cheating_turn"] = remaining_turns
            filtered_cases.append(new_case)

    return filtered_cases


def filter_dataset(dataset_name: str, outputs_root: Path) -> None:
    dataset_root = outputs_root / dataset_name
    type2_path = dataset_root / TYPE2_FILENAME
    type3_path = dataset_root / TYPE3_FILENAME

    type2_cases = read_jsonl(type2_path)
    type3_cases = read_jsonl(type3_path)
    filtered_cases = remove_overlapping_turns(
        type2_cases,
        turn_keys(type3_cases),
    )
    write_jsonl(type2_path, filtered_cases)


def main() -> None:
    for dataset_name in DEFAULT_DATASETS:
        filter_dataset(dataset_name, DEFAULT_OUTPUTS_ROOT)


if __name__ == "__main__":
    main()
