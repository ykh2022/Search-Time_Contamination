from __future__ import annotations

import re
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from utils.load import load_analysis_data
from utils.utils import (
    DEFAULT_DATASETS,
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_RESULTS_ROOT,
    build_cheat_case_annotation,
    compute_detection_accuracy,
    indexed_questions_by_id,
    input_path_for_dataset,
    limit_questions_for_test,
    split_question_and_options,
    write_jsonl,
)
from tqdm import tqdm


ANY_WORD_RE = re.compile(r"[a-z0-9]+")

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def words_division(text: str) -> list[str]:
    return ANY_WORD_RE.findall(normalize_ws(text).lower())

def longest_common_substring_and_sequence(question, response):
    X = words_division(question)
    Y = words_division(response)
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    dp_sequence = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    end_pos = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                dp_sequence[i][j] = dp_sequence[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0
                dp_sequence[i][j] = max(dp_sequence[i - 1][j], dp_sequence[i][j - 1])

    sequence_words = []
    i, j = m, n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            sequence_words.append(X[i - 1])
            i -= 1
            j -= 1
        elif dp_sequence[i - 1][j] >= dp_sequence[i][j - 1]:
            i -= 1
        else:
            j -= 1
    sequence_words.reverse()

    longest_substring = " ".join(X[end_pos - max_length:end_pos])
    longest_substring_sequence = " ".join(sequence_words)
    return max_length, longest_substring, dp_sequence[m][n], longest_substring_sequence

def matching_blocks_coverage(
    question: str,
    response: str,
    min_block_size: int = 3,
) -> tuple[int, float, list[dict[str, Any]]]:
    X = words_division(question)
    Y = words_division(response)
    matcher = SequenceMatcher(None, X, Y, autojunk=False)
    blocks = [
        block
        for block in matcher.get_matching_blocks()
        if block.size >= min_block_size
    ]
    matched_len = sum(block.size for block in blocks)
    ratio = matched_len / len(X) if X else 0
    block_items = [
        {
            "size": block.size,
            "text": " ".join(X[block.a:block.a + block.size]),
        }
        for block in blocks
    ]
    return matched_len, ratio, block_items

def matching_metrics(
    question_text: str,
    response_text: str,
    dataset_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    common_substring_len, common_substring, common_subsequence_len, common_subsequence = longest_common_substring_and_sequence(question_text, response_text)
    question_length = len(words_division(question_text))
    ratio_substring = common_substring_len / question_length if question_length else 0
    ratio_subsequence = common_subsequence_len / question_length if question_length else 0
    matching_blocks_len, matching_blocks_ratio, matching_blocks = matching_blocks_coverage(
        question_text,
        response_text,
    )
    metrics = {
        # "question_length": question_length,
        "common_substring_len": common_substring_len,
        "common_substring_ratio": ratio_substring,
        "common_substring": common_substring,
        "common_subsequence_len": common_subsequence_len,
        "common_subsequence_ratio": ratio_subsequence,
        "common_subsequence": common_subsequence,
        "matching_blocks_len": matching_blocks_len,
        "matching_blocks_ratio": matching_blocks_ratio,
        "matching_blocks_count": len(matching_blocks),
        "matching_blocks": matching_blocks,
    }
    # 搜到了原网址提高要求
    signal = ratio_substring >= 0.5 or ratio_subsequence >= 0.8 or matching_blocks_ratio >= 0.5
    if dataset_name in ["MedMCQA", "MMLU"]:
        signal = len(matching_blocks) >= 1 and (ratio_substring >= 0.7 or (ratio_substring >= 0.5 and ratio_subsequence >= 0.8 and matching_blocks_ratio >= 0.65))
    if not signal:
        return None, metrics

    event = {
        "response_snippet": normalize_ws(response_text)[:350],
        **metrics,
    }
    return event, metrics

def detect_visit_question(
    question_stem: str,
    round_item: dict[str, Any],
    dataset_name: str,
) -> list[dict[str, Any]]:
    events = []
    tool_calls = round_item.get("tool_calls") or round_item.get("tools") or []
    tool_responses = round_item.get("tool_responses", [])
    for idx, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        name = str(call.get("name", "")).strip().lower()
        if name == "visit" and idx < len(tool_responses):
            response_text = tool_responses[idx]
            if not response_text:
                continue
            # Question reconstruction signal
            question_text = question_stem[10:]
            event, _ = matching_metrics(
                question_text,
                response_text,
                dataset_name,
            )
            if event is not None:
                events.append(event)
                continue
    return events, 2

def type2_check(
    question_record: dict[str, Any],
    idx: int,
    dataset_name: str,
) -> dict[str, Any] | None:
    raw_question_text = str(question_record.get("question", ""))
    question_text, _ = split_question_and_options(raw_question_text)

    turns = question_record.get("new_messages", {}).get("turns", [])
    cheating_turns: list[dict[str, Any]] = []

    for turn_idx, turn in enumerate(turns, start=1):
        round_item = {**turn, "round_id": turn_idx}
        events, cheating_type = detect_visit_question(
            question_text,
            round_item,
            dataset_name,
        )
        if not events or cheating_type != 2:
            continue

        cheating_turns.append({
            "turn": turn_idx,
            "cheating_type": cheating_type,
            "cheating_reasons": events,
        })

    if not cheating_turns:
        return None

    return build_cheat_case_annotation(
        idx,
        question_text,
        turns,
        cheating_turns,
        question_record.get("prediction"),
        question_record.get("answer"),
    )

def analyze_dataset(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    dataset_name: str = "",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    questions = limit_questions_for_test(load_analysis_data(input_path))
    indexed_questions, question_ids = indexed_questions_by_id(questions)

    results: list[dict[str, Any]] = []
    for idx, question_record in tqdm(indexed_questions, desc=f"Type2 {dataset_name}"):
        result = type2_check(
            question_record,
            idx,
            dataset_name,
        )
        if result is not None:
            results.append(result)

    if output_path is not None:
        write_jsonl(output_path, results)

    accuracy_stats = compute_detection_accuracy(
        questions,
        results,
        "type2",
        question_ids,
    )
    return results, accuracy_stats

def main() -> None:
    dataset_names = DEFAULT_DATASETS
    results_root = DEFAULT_RESULTS_ROOT
    outputs_root = DEFAULT_OUTPUTS_ROOT
    summary: dict[str, Any] = {}

    for dataset_name in dataset_names:
        input_path = input_path_for_dataset(results_root, dataset_name)
        output_path = outputs_root / dataset_name / "Type2.jsonl"
        results, accuracy_stats = analyze_dataset(
            input_path,
            output_path,
            dataset_name,
        )
        summary[dataset_name] = {
            "num_detected_cases": len(results),
            "accuracy": accuracy_stats,
        }

    print(json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
