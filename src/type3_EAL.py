from __future__ import annotations

import asyncio
import argparse
import ast
import json
import re
import sys
from typing import Any

from utils.llm import SYSTEM_PROMPTv1, USER_PROMPT, deepseek_completion_async, openai_completion_async
from utils.load import load_analysis_data
from utils.utils import (
    DEFAULT_DATASETS,
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_RESULTS_ROOT,
    build_cheat_case_annotation,
    compute_detection_accuracy,
    input_path_for_dataset,
    limit_questions_for_test,
    split_question_and_options,
    write_indexed_jsonl,
    write_jsonl,
)
from tqdm import tqdm

LLM_MODEL_CHOICES = ("gpt", "deepseek")

def extract_options(options_text: str, answer: str) -> str:
    options_text = str(options_text or "").replace("\\n", "\n")
    answer = str(answer or "").strip().upper()
    if not answer:
        return ""

    try:
        parsed_options = ast.literal_eval(options_text)
    except (SyntaxError, ValueError):
        parsed_options = None

    if isinstance(parsed_options, dict):
        answer_value = parsed_options.get(answer)
        if answer_value is not None:
            return f"{answer}: {str(answer_value).strip()}"

    dict_like_match = re.search(
        rf"['\"]{re.escape(answer)}['\"]\s*:\s*['\"](?P<value>.*?)(?=['\"]\s*,\s*['\"][A-E]['\"]\s*:|['\"]\s*}}\s*$)",
        options_text,
        re.S,
    )
    if dict_like_match:
        value = dict_like_match.group("value")
        value = re.sub(r"\s+", " ", value).strip()
        value = value.rstrip("'\"").strip()
        return f"{answer}: {value}"

    match = re.search(
        rf"(?im)^\s*({re.escape(answer)})[\.\):]\s*(.+)$",
        options_text,
    )
    if not match:
        return answer
    return f"{match.group(1)}: {match.group(2).strip()}"

def is_visit_tool_call(tool_call: Any = None) -> bool:
    return (
        isinstance(tool_call, dict)
        and str(tool_call.get("name", "")).strip().lower() == "visit"
    )

def visit_tool_responses(turn: dict[str, Any] | None = None) -> list[str]:
    turn = turn or {}
    tools = turn.get("tools") or []
    tool_responses = turn.get("tool_responses") or []
    visit_indices = [
        idx for idx, tool in enumerate(tools) if is_visit_tool_call(tool)
    ]
    responses: list[str] = []
    for idx in visit_indices:
        if idx < len(tool_responses):
            response_text = str(tool_responses[idx]).strip()
            if response_text:
                responses.append(response_text)
    return responses

def parse_llm_judgment(raw_response: Any = None) -> dict[str, Any]:
    text = str(raw_response or "").strip()
    if not text:
        return {
            "contaminated": False,
            "reason": "",
            "raw_response": text,
        }
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'"?contaminated"?\s*[:=]\s*(true|false)', text, re.I)
        return {
            "contaminated": bool(match and match.group(1).lower() == "true"),
            "reason": text,
            "raw_response": text,
        }
    return {
        "contaminated": parse_bool(parsed.get("contaminated", False)),
        "reason": str(parsed.get("reason") or parsed.get("reasoning") or "").strip(),
        "raw_response": text,
    }

def parse_bool(value: Any = None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return value != 0
    return False

async def type3_check(
    question_record: dict[str, Any],
    idx: int,
    model: str = "deepseek",
) -> dict[str, Any] | None:
    raw_question_text = str(question_record.get("question", ""))
    question_text, options_text = split_question_and_options(raw_question_text)
    answer = str(question_record.get("answer") or "").upper()
    answer_text = extract_options(options_text, answer)

    turns = question_record.get("new_messages", {}).get("turns", [])
    cheating_turns: list[dict[str, Any]] = []

    for turn_idx, turn in enumerate(turns, start=1):
        # Check this turn's tool responses
        visit_responses = visit_tool_responses(turn)
        if not visit_responses:
            continue

        if len(visit_responses) > 1:
            visit_responses = "\n".join(visit_responses)
        else:
            visit_responses = visit_responses[0]

        user_prompt = USER_PROMPT.format(
            question=question_text,
            answer_text=answer_text,
            response_text=visit_responses,
        )
        if model == "gpt":
            raw_response = await openai_completion_async(user_prompt, SYSTEM_PROMPTv1)
        else:
            raw_response = await deepseek_completion_async(user_prompt, SYSTEM_PROMPTv1)
        judgment = parse_llm_judgment(raw_response)

        # output
        if judgment["contaminated"]:
            cheating_turns.append({
                "turn": turn_idx,
                "cheating_type": 3,
                "cheating_reasons": judgment.get("reason", judgment.get("raw_response", "")),
            })

    final_answer = question_record.get("prediction")
    if not cheating_turns:
        return None

    return build_cheat_case_annotation(
        idx,
        question_text,
        turns,
        cheating_turns,
        final_answer,
        answer,
    )

async def analyze_dataset(
    input_path,
    output_path,
    concurrency,
    dataset_name: str,
    write_every: int = 20,
    model: str = "deepseek",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    model = str(model).strip().lower()
    questions = limit_questions_for_test(load_analysis_data(input_path))
    concurrency = max(1, concurrency)
    write_every = max(1, write_every)
    question_semaphore = asyncio.Semaphore(concurrency)
    write_jsonl(output_path, [])

    async def run_check(
        idx: int,
        question_record: dict[str, Any],
    ) -> tuple[int, dict[str, Any] | None]:
        async with question_semaphore:
            try:
                result = await type3_check(
                    question_record,
                    idx,
                    model=model,
                )
            except Exception as exc:
                print(
                    f"Question {idx} failed after retries: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return idx, None
            return idx, result

    tasks = [
        asyncio.create_task(run_check(int(question_record["id"]), question_record))
        for question_record in questions
        if question_record.get("id") is not None
    ]
    indexed_results: list[tuple[int, dict[str, Any]]] = []
    completed_count = 0
    with tqdm(total=len(tasks), desc=f"Type3 {dataset_name}") as progress:
        for task in asyncio.as_completed(tasks):
            idx, result = await task
            completed_count += 1
            if result is not None:
                indexed_results.append((idx, result))
            if completed_count % write_every == 0:
                write_indexed_jsonl(output_path, indexed_results)
            progress.update(1)

    results = write_indexed_jsonl(output_path, indexed_results)

    accuracy_stats = compute_detection_accuracy(questions, results, "type3")
    return results, accuracy_stats

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect type-3 cheating cases with an LLM judge.",
    )
    parser.add_argument(
        "--model",
        choices=LLM_MODEL_CHOICES,
        default="deepseek",
        help="LLM provider used to generate raw_response.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of concurrent LLM checks to run.",
    )
    return parser.parse_args()

async def main_async(
    model: str = "deepseek",
    concurrency: int = 20,
) -> None:
    model = str(model).strip().lower()
    dataset_names = DEFAULT_DATASETS
    results_root = DEFAULT_RESULTS_ROOT
    outputs_root = DEFAULT_OUTPUTS_ROOT
    summary: dict[str, Any] = {}

    for dataset_name in dataset_names:
        input_path = input_path_for_dataset(results_root, dataset_name)
        output_path = outputs_root / dataset_name / "Type3.jsonl"
        results, accuracy_stats = await analyze_dataset(
            input_path,
            output_path,
            dataset_name=dataset_name,
            model=model,
            concurrency=concurrency,
        )
        summary[dataset_name] = {
            "num_detected_cases": len(results),
            "accuracy": accuracy_stats,
        }
        print(json.dumps({dataset_name: summary[dataset_name]}, ensure_ascii=False))

def main() -> None:
    args = parse_args()
    asyncio.run(
        main_async(
            model=args.model,
            concurrency=args.concurrency,
        )
    )

if __name__ == "__main__":
    main()
