from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from utils.extract_answer import extract_answer_from_text, extract_answer_hle
from utils.load import load_analysis_data
from utils.utils import (
    DEFAULT_DATASETS,
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_RESULTS_ROOT,
    input_path_for_dataset,
)


ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Z])\s*</answer>", re.I)
OPTION_LABEL_RE = re.compile(r"(?im)^\s*([A-Z])\s*:")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*.*?\s*</tool_call>", re.DOTALL | re.I)
VALID_OPTIONS = set(string.ascii_uppercase[:5])
DEFAULT_CASE_FILENAMES = ["Type1.jsonl", "Type2.jsonl", "Type3.jsonl"]


@dataclass
class AnswerExtractionResult:
    answer: str | None
    checked: bool
    correct: bool | None
    reasoning: str | None = None
    llm_input: dict[str, str] | None = None


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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_question_and_options(question_text: str) -> tuple[str, str]:
    normalized_text = str(question_text or "").replace("\\n", "\n")
    parts = re.split(r"(?im)\n\s*options\s*:\s*", normalized_text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return normalized_text.strip(), ""


def valid_options_from_text(options: str) -> set[str]:
    labels = {match.upper() for match in OPTION_LABEL_RE.findall(options or "")}
    return labels or VALID_OPTIONS


def normalize_answer(text: Any, valid_options: set[str] | None = None) -> str | None:
    valid_options = valid_options or VALID_OPTIONS
    value = str(text or "").strip()
    if not value:
        return None

    upper = value.upper()
    if upper in valid_options:
        return upper

    tag_matches = ANSWER_TAG_RE.findall(value)
    if tag_matches:
        return normalize_answer(tag_matches[-1], valid_options)

    letter_colon_match = re.match(r"^\s*([A-Z])\s*:", value, re.I)
    if letter_colon_match:
        candidate = letter_colon_match.group(1).upper()
        if candidate in valid_options:
            return candidate

    return None


def extract_thinking_text(assistant_content: str) -> str:
    without_tool_calls = TOOL_CALL_RE.sub("", assistant_content)
    return without_tool_calls.strip()


def normalize_correct_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1", "correct"}:
        return True
    if normalized in {"no", "false", "0", "incorrect"}:
        return False
    return None


def is_choice_correct(answer: Any, correct_answer: Any, valid_options: set[str]) -> bool | None:
    normalized_answer = normalize_answer(answer, valid_options)
    if normalized_answer is None:
        return False
    normalized_correct = normalize_answer(correct_answer, valid_options)
    if normalized_correct is None:
        return None
    return normalized_answer == normalized_correct


def stored_checked_result(
    turn: dict[str, Any] | None,
    correct_answer: str,
    valid_options: set[str] | None = None,
) -> AnswerExtractionResult:
    if not turn or turn.get("checked") is not True:
        return AnswerExtractionResult(None, False, None)
    answer = None if turn.get("answers") is None else str(turn.get("answers")).strip()
    if answer is not None and answer.lower() in {"none", "null"}:
        answer = None
    correct = normalize_correct_flag(turn.get("correct"))
    if correct is None and answer is not None and valid_options is not None:
        correct = is_choice_correct(answer, correct_answer, valid_options)
        turn["correct"] = correct
    reasoning = None if turn.get("reasoning") is None else str(turn.get("reasoning"))
    llm_input = turn.get("llm_input")
    if not isinstance(llm_input, dict):
        llm_input = None
    return AnswerExtractionResult(answer, True, correct, reasoning, llm_input)


def set_checked_result(
    turn: dict[str, Any],
    answer: Any,
    correct: bool | None = None,
    reasoning: str | None = None,
    llm_input: dict[str, str] | None = None,
) -> AnswerExtractionResult:
    normalized_answer = None if answer is None else str(answer).strip()
    if normalized_answer is not None and normalized_answer.lower() in {"none", "null"}:
        normalized_answer = None
    turn["answers"] = normalized_answer
    turn["checked"] = True
    turn["correct"] = correct
    if reasoning is not None:
        turn["reasoning"] = reasoning
    if llm_input is not None:
        turn["llm_input"] = llm_input
    return AnswerExtractionResult(normalized_answer, True, correct, reasoning, llm_input)


def answer_from_turn_field(
    turn: dict[str, Any] | None,
    valid_options: set[str],
) -> str | None:
    if not turn:
        return None
    answer = str(turn.get("answers", "")).strip()
    if answer == "":
        return None
    if answer == "null":
        return answer
    normalized_answer = normalize_answer(answer, valid_options)
    if normalized_answer:
        turn["answers"] = normalized_answer
        return normalized_answer
    return None


async def answer_from_turn_text(
    turn: dict[str, Any] | None,
    options: str,
    question_id: int,
    turn_id: int,
    dataset_name: str,
    question_text: str,
    correct_answer: str,
) -> AnswerExtractionResult:
    if not turn:
        return AnswerExtractionResult(None, False, None)
    
    turn.setdefault("checked", False)
    assistant_content = str(turn.get("assistant") or "")
    thinking = extract_thinking_text(assistant_content)

    if dataset_name.lower() == "hle149":
        if turn.get("checked") is True:
            # print(
            #     f"[question_id={question_id}, turn_id={turn_id}] "
            #     f"Using stored checked result for HLE149.",
            #     flush=True,
            # )
            return stored_checked_result(turn, correct_answer)

        try:
            llm_response = thinking or assistant_content
            result = await extract_answer_hle(
                question_text,
                correct_answer,
                llm_response,
            )
        except Exception as exc:
            print(
                f"[question_id={question_id}, turn_id={turn_id}] "
                f"HLE answer extraction failed: {exc}",
                flush=True,
            )
            return AnswerExtractionResult(None, False, None)
        answer = result.get("model_answer") if result else None
        correct = normalize_correct_flag(result.get("correct") if result else None)
        reasoning = result.get("reasoning") if result else None
        llm_input = {
            "question": question_text,
            "correct_answer": correct_answer,
            "response": llm_response,
        }
        return set_checked_result(turn, answer, correct, reasoning, llm_input)

    if turn.get("checked") is True:
        if normalize_correct_flag(turn.get("correct")) is not None:
            return stored_checked_result(turn, correct_answer)
        return stored_checked_result(
            turn,
            correct_answer,
            valid_options_from_text(options),
        )

    valid_options = valid_options_from_text(options)
    for answer in (
        answer_from_turn_field(turn, valid_options),
        normalize_answer(assistant_content, valid_options),
        normalize_answer(thinking, valid_options),
    ):
        if answer is not None:
            correct = is_choice_correct(answer, correct_answer, valid_options)
            return set_checked_result(turn, answer, correct)

    try:
        answer = await extract_answer_from_text(
            thinking,
            options,
            question_id=question_id,
            turn_id=turn_id,
        )
    except Exception as exc:
        print(
            f"[question_id={question_id}, turn_id={turn_id}] "
            f"LLM answer extraction failed: {exc}",
            flush=True,
        )
        return AnswerExtractionResult(None, False, None)
    correct = is_choice_correct(answer, correct_answer, valid_options)
    return set_checked_result(turn, answer, correct)


def final_output_path(input_path: Path) -> Path:
    return input_path


def hle_reasons_output_path(dataset_root: Path) -> Path:
    return dataset_root / "hle_answer_extraction_reasons.jsonl"


def records_without_hle_reasoning(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned_records = copy.deepcopy(records)
    for record in cleaned_records:
        for turn in record.get("cheating_turn") or []:
            turn.pop("before_reasoning", None)
            turn.pop("after_reasoning", None)
            turn.pop("before_llm_input", None)
            turn.pop("after_llm_input", None)
    return cleaned_records


def default_input_paths(
    dataset_root: Path,
) -> list[Path]:
    input_paths: list[Path] = []
    for filename in DEFAULT_CASE_FILENAMES:
        path = dataset_root / filename
        if path.exists():
            input_paths.append(path)
        else:
            print(f"[missing input] {path}", flush=True)
    return input_paths


def collect_turn_tasks(
    records_by_path: dict[Path, list[dict[str, Any]]],
) -> list[tuple[int, int, Path, int, int]]:
    tasks: list[tuple[int, int, Path, int, int]] = []
    for input_path, records in records_by_path.items():
        for case_index, case in enumerate(records):
            case_id = case.get("id")
            if case_id is None:
                continue
            for turn_index, turn in enumerate(case.get("cheating_turn") or []):
                turn_id = turn.get("turn")
                if turn_id is not None:
                    tasks.append(
                        (
                            int(case_id),
                            int(turn_id),
                            input_path,
                            case_index,
                            turn_index,
                        )
                    )
    return sorted(tasks, key=lambda item: (item[0], item[1], str(item[2])))


async def annotate_case_records(
    records_by_path: dict[Path, list[dict[str, Any]]],
    questions: list[dict[str, Any]],
    dataset_name: str,
    desc: str,
    concurrency: int = 1,
) -> dict[Path, list[dict[str, Any]]]:
    tasks = collect_turn_tasks(records_by_path)
    if not tasks:
        return records_by_path

    concurrency = max(1, concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    tasks_by_question: dict[int, list[tuple[int, int, Path, int, int]]] = {}
    for task in tasks:
        tasks_by_question.setdefault(task[0], []).append(task)
    questions_by_id = {
        int(question["id"]): question
        for question in questions
        if question.get("id") is not None
    }

    async def answer_at_turn(
        turns: list[dict[str, Any]],
        turn_number: int,
        options: str,
        question_id: int,
        dataset_name: str,
        question_text: str,
        correct_answer: str,
    ) -> AnswerExtractionResult:
        turn_index = turn_number - 1
        turn = turns[turn_index] if 0 <= turn_index < len(turns) else None
        return await answer_from_turn_text(
            turn,
            options,
            question_id,
            turn_number,
            dataset_name,
            question_text,
            correct_answer,
        )

    async def annotate_question(
        question_id: int,
        question_tasks: list[tuple[int, int, Path, int, int]],
        progress: tqdm,
    ) -> None:
        async with semaphore:
            # print(f"[{dataset_name}] Annotating question_id={question_id} with {len(question_tasks)} turn tasks.", flush=True)
            question = questions_by_id.get(question_id)
            if question is None:
                print(f"[{dataset_name}] Warning: question_id={question_id} not found in questions data.", flush=True)
                progress.update(len(question_tasks))
                return

            question_text = str(question.get("question", ""))
            correct_answer = str(question.get("answer", ""))
            _, options = split_question_and_options(question_text)
            turns = question.get("new_messages", {}).get("turns", [])

            for _, turn_number, input_path, case_index, turn_index in question_tasks:
                turn = records_by_path[input_path][case_index]["cheating_turn"][
                    turn_index
                ]
                before_result = await answer_at_turn(
                    turns,
                    turn_number,
                    options,
                    question_id,
                    dataset_name,
                    question_text,
                    correct_answer,
                )
                after_turn_number = (
                    turn_number + 1
                    if turn_number < len(turns)
                    else turn_number
                )
                after_result = await answer_at_turn(
                    turns,
                    after_turn_number,
                    options,
                    question_id,
                    dataset_name,
                    question_text,
                    correct_answer,
                )
                turn["before_cheat_answer"] = before_result.answer
                turn["before_correct"] = before_result.correct
                if before_result.reasoning is not None:
                    turn["before_reasoning"] = before_result.reasoning
                if before_result.llm_input is not None:
                    turn["before_llm_input"] = before_result.llm_input
                turn["after_cheat_answer"] = after_result.answer
                turn["after_correct"] = after_result.correct
                if after_result.reasoning is not None:
                    turn["after_reasoning"] = after_result.reasoning
                if after_result.llm_input is not None:
                    turn["after_llm_input"] = after_result.llm_input
                progress.update(1)

    with tqdm(total=len(tasks), desc=desc, unit="turn") as progress:
        await asyncio.gather(
            *(
                annotate_question(question_id, question_tasks, progress)
                for question_id, question_tasks in tasks_by_question.items()
            )
        )

    return records_by_path


async def annotate_dataset(
    dataset_name: str,
    results_root: Path,
    outputs_root: Path,
    concurrency: int,
) -> dict[str, list[dict[str, Any]]]:
    data_path = input_path_for_dataset(results_root, dataset_name)
    dataset_root = outputs_root / dataset_name
    input_paths = default_input_paths(dataset_root)
    print(
        f"[{dataset_name}] input files: "
        f"{[str(input_path) for input_path in input_paths]}",
        flush=True,
    )
    if not input_paths:
        print(
            f"[{dataset_name}] no input files found: {DEFAULT_CASE_FILENAMES}",
            flush=True,
        )
        return {}

    questions = load_analysis_data(data_path)
    records_by_path = {input_path: read_jsonl(input_path) for input_path in input_paths}
    annotated_by_path = await annotate_case_records(
        records_by_path,
        questions,
        dataset_name,
        desc=f"Annotating {dataset_name}",
        concurrency=concurrency,
    )

    for input_path, records in annotated_by_path.items():
        output_path = final_output_path(input_path)
        print(
            f"[{dataset_name}] writing {output_path} records={len(records)}",
            flush=True,
        )
        if dataset_name.lower() == "hle149":
            write_jsonl(output_path, records_without_hle_reasoning(records))
        else:
            write_jsonl(output_path, records)

    if dataset_name.lower() == "hle149":
        reason_records: list[dict[str, Any]] = []
        for input_path, records in annotated_by_path.items():
            for record in records:
                for turn in record.get("cheating_turn") or []:
                    reason_record = {
                        "source_file": input_path.name,
                        "id": record.get("id"),
                        "turn": turn.get("turn"),
                        "before_cheat_answer": turn.get("before_cheat_answer"),
                        "before_correct": turn.get("before_correct"),
                        "before_llm_input": turn.get("before_llm_input"),
                        "before_reasoning": turn.get("before_reasoning"),
                        "after_cheat_answer": turn.get("after_cheat_answer"),
                        "after_correct": turn.get("after_correct"),
                        "after_llm_input": turn.get("after_llm_input"),
                        "after_reasoning": turn.get("after_reasoning"),
                    }
                    if (
                        reason_record["before_reasoning"] is not None
                        or reason_record["after_reasoning"] is not None
                        or reason_record["before_llm_input"] is not None
                        or reason_record["after_llm_input"] is not None
                    ):
                        reason_records.append(reason_record)
        reason_output_path = hle_reasons_output_path(dataset_root)
        print(
            f"[{dataset_name}] writing {reason_output_path} records={len(reason_records)}",
            flush=True,
        )
        write_jsonl(reason_output_path, reason_records)

    return {input_path.stem: records for input_path, records in annotated_by_path.items()}


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate selected cheat case files with before/after answers."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names under results-root and outputs-root.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing Tongyi DeepResearch JSONL input files.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Directory containing per-dataset cheating case JSONL files.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=25,
        help=(
            "Number of question-level annotation tasks to run concurrently. "
            "Turns within each question are processed sequentially."
        ),
    )
    args = parser.parse_args()

    for dataset_name in args.datasets:
        await annotate_dataset(
            dataset_name,
            args.results_root,
            args.outputs_root,
            concurrency=args.concurrency,
        )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
