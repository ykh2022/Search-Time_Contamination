from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_ROOT = Path("../output/Tongyi_DeepResearch")
DEFAULT_OUTPUTS_ROOT = Path(os.environ.get("OUTPUTS_ROOT", "../detection/Tongyi_DeepResearch"))

DEFAULT_DATASETS = (
    "HLE149",
    "MedBulltes5op",
    "MedMCQA",
    "MedQA",
    "MMLU",
    "MedXpertQA",
)

def input_path_for_dataset(input_root: str | Path, dataset_name: str) -> Path:
    return Path(input_root) / f"Tongyi_DeepResearch_{dataset_name}.jsonl"


def split_question_and_options(question_text: str) -> tuple[str, str]:
    normalized_text = str(question_text or "").replace("\\n", "\n")
    parts = re.split(r"(?im)\n\s*options\s*:\s*", normalized_text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return normalized_text.strip(), ""


def indexed_questions_by_id(
    questions: list[dict[str, Any]],
) -> tuple[list[tuple[int, dict[str, Any]]], list[int]]:
    indexed_questions = [
        (int(question_record["id"]), question_record)
        for question_record in questions
    ]
    question_ids = [
        idx
        for idx, _ in indexed_questions
    ]
    return indexed_questions, question_ids


def limit_questions_for_test(
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    limit_text = os.environ.get("DETECTION_LIMIT", "").strip()
    if not limit_text:
        return questions

    limit = int(limit_text)
    if limit <= 0:
        return questions
    return questions[:limit]


def build_cheat_case_annotation(
    question_id: int,
    question_text: str,
    turns: list[dict[str, Any]],
    cheating_turns: list[dict[str, Any]],
    final_answer: Any,
    ground_truth: Any,
) -> dict[str, Any]:
    return {
        "id": question_id,
        "prefix": question_text[10:200],
        "total_turn": len(turns),
        "cheating_turn": cheating_turns,
        "final_answer": final_answer,
        "ground_truth": str(ground_truth or "").upper(),
    }


def write_jsonl(path: str | Path | None, records: list[dict[str, Any]]) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def sorted_indexed_records(
    indexed_items: list[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        result
        for _, result in sorted(indexed_items, key=lambda item: item[0])
    ]


def write_indexed_jsonl(
    path: str | Path | None,
    indexed_items: list[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    records = sorted_indexed_records(indexed_items)
    write_jsonl(path, records)
    return records


def accuracy(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for record in records if record["correct"])
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
    }


def compute_detection_accuracy(
    questions: list[dict[str, Any]],
    detected_results: list[dict[str, Any]],
    detection_type: str,
    question_ids: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    if question_ids is None:
        question_ids = [
            int(question["id"]) if question.get("id") is not None else idx
            for idx, question in enumerate(questions, start=1)
        ]

    detected_question_ids = {
        int(result["id"])
        for result in detected_results
        if result.get("id") is not None
    }
    with_detection = [
        question
        for idx, question in zip(question_ids, questions)
        if idx in detected_question_ids
    ]
    return {
        f"with_{detection_type}": accuracy(with_detection),
    }
