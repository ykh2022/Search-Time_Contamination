from __future__ import annotations

import json
import re
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


URL_RE = re.compile(r"https?://[^\s<>'\"`)\]}]+", re.I)
URL_SEP = r"(?:[-_./+]|%20)+"

COMMON_TYPE1_PATTERNS = [
    # datahosting platforms
    ("https://huggingface.co/datasets", r"https?://(?:www\.)?huggingface\.co/datasets\b"),
    ("https://github.com", r"https?://(?:www\.)?github\.com\b"),

    # medical exam related patterns
    ("https://www.coursehero.com", r"https?://(?:www\.)?coursehero\.com\b"),
    ("https://getoncourse.ai", r"https?://(?:www\.)?getoncourse\.ai\b"),
    ("https://medicoapps.org", r"https?://(?:www\.)?medicoapps\.org\b"),
    ("https://accessmedicine.mhmedical.com/ + Question/Quiz", r"https?://accessmedicine\.mhmedical\.com/\S*(?:question|quiz)\S*"),
    ("https://quizlet.com/", r"https?://(?:www\.)?quizlet\.com\b"),
    ("https://www.cram.com/flashcards/", r"https?://(?:www\.)?cram\.com/flashcards\b"),
    ("https://www.scribd.com/document", r"https?://(?:www\.)?scribd\.com/document\b"),
    ("https://www.orthobullets.com/testview", r"https?://(?:www\.)?orthobullets\.com/testview\b"),
    ("https://www.dentaldevotee.com/ + yyyy", r"https?://(?:www\.)?dentaldevotee\.com/\S*(?:19|20)\d{2}\S*"),
    ("https://www.transtutors.com/questions", r"https?://(?:www\.)?transtutors\.com/questions\b"),
    ("https://www.cliffsnotes.com/cliffs-questions", r"https?://(?:www\.)?cliffsnotes\.com/cliffs-questions\b"),
    ("https://homework.study.com/explanation", r"https?://homework\.study\.com/explanation\b"),
    ("aiims", r"\baiims\b"),
    ("dental pulse", rf"\bdental{URL_SEP}pulse\b"),
    ("SMART DENTAL REVISION", rf"\bsmart{URL_SEP}dental{URL_SEP}revision\b"),
    ("ORAL SURGERY LIVE Test", rf"\boral{URL_SEP}surgery{URL_SEP}live{URL_SEP}test\b"),
    ("USMLE", r"\busmle\b"),
    ("https://www.osmosis.org/blog/usmle", r"https?://(?:www\.)?osmosis\.org/blog/usmle\b"),
    ("passmed", r"\bpassmed\b"),
    ("mcq", r"\bmcqs?\b"),
    ("quiz", r"\bquiz(?:zes)?\b"),
    ("Question", r"\bquestions?\b"),
    ("Test", r"\btests?\b"),
]

DATASET_NAME_PATTERNS_BY_DATASET = {
    "MedBulltes5op": [
        ("https://step2.medbullets.com/testview", r"https?://step2\.medbullets\.com/testview\b"),
        ("https://www.facebook.com/medbullets", r"https?://(?:www\.)?facebook\.com/medbullets\b"),
        ("medbullets", r"\bmedbullets?\b"),
        ("medbullets_op[45]", r"\bmedbullets_op[45]\b"),
    ],
    "MedQA": [
        ("medqa", r"\bmedqa\b"),
        ("OpenMedQA", rf"\bopen(?:{URL_SEP})?medqa\b"),
    ],
    "MedMCQA": [
        ("https://dokumen.pub", r"https?://(?:www\.)?dokumen\.pub\b"),
        ("https://www.facebook.com", r"https?://(?:www\.)?facebook\.com\b"),
        ("medmcqa", r"\bmedmcqa\b"),
    ],
    "MMLU": [
        ("mmlu", r"\bmmlu\b"),
    ],
    "MedXpertQA": [
        ("https://arxiv.org/pdf/2505.18283", r"https?://arxiv\.org/pdf/2505\.18283\b"),
        ("MedxpertQA", rf"\bmedxpert(?:{URL_SEP})?qa\b"),
    ],
    "HLE149": [
        ("HLE", r"\bhle\b"),
        ("Human last exam", rf"\b(?:human|humanitys){URL_SEP}last{URL_SEP}exam\b"),
    ],
}

COMMON_TYPE1_REGEXES = [
    (label, re.compile(pattern, re.I))
    for label, pattern in COMMON_TYPE1_PATTERNS
]
DATASET_NAME_REGEXES_BY_DATASET = {
    dataset_name: [
        (label, re.compile(pattern, re.I))
        for label, pattern in patterns
    ]
    for dataset_name, patterns in DATASET_NAME_PATTERNS_BY_DATASET.items()
}


def type1_regexes_for_dataset(dataset_name: str) -> list[tuple[str, re.Pattern[str]]]:
    return [
        *COMMON_TYPE1_REGEXES,
        *DATASET_NAME_REGEXES_BY_DATASET.get(dataset_name, []),
    ]


def iter_strings(value: Any):
    if value is None:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)
    else:
        yield str(value)


def round_urls(round_item: dict[str, Any]) -> list[str]:
    tool_calls = round_item.get("tools") or []
    tool_responses = round_item.get("tool_responses") or []

    urls: list[str] = []
    for idx, call in enumerate(tool_calls):
        if not isinstance(call, dict) or str(call.get("name", "")).strip().lower() != "search":
            continue
        payloads = [call]
        if idx < len(tool_responses):
            payloads.append(tool_responses[idx])
        for payload in payloads:
            for text in iter_strings(payload):
                urls.extend(match.group(0).rstrip(".,;:") for match in URL_RE.finditer(text))
    return urls


def detect_type1_keywords(
    round_item: dict[str, Any],
    dataset_name: str,
) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for url in round_urls(round_item):
        for keyword, pattern in type1_regexes_for_dataset(dataset_name):
            match = pattern.search(url)
            if not match or (keyword, url.lower()) in seen:
                continue

            seen.add((keyword, url.lower()))
            events.append(
                {
                    "matched_keyword": keyword,
                    "matched_url": url,
                }
            )

    return events, 1


def type1_check(
    question_record: dict[str, Any],
    idx: int,
    dataset_name: str,
) -> dict[str, Any] | None:
    question_text, _ = split_question_and_options(question_record.get("question", ""))
    turns = question_record.get("new_messages", {}).get("turns", [])
    cheating_turns: list[dict[str, Any]] = []

    for turn_idx, turn in enumerate(turns, start=1):
        events, cheating_type = detect_type1_keywords(turn, dataset_name)
        if events:
            cheating_turns.append(
                {
                    "turn": turn_idx,
                    "cheating_type": cheating_type,
                    "cheating_reasons": events,
                }
            )

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
    input_path: str | Path,
    output_path: str | Path,
    dataset_name: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    questions = limit_questions_for_test(load_analysis_data(input_path))
    indexed_questions, question_ids = indexed_questions_by_id(questions)

    results = [
        result
        for idx, question_record in tqdm(indexed_questions, desc=f"Type1 {dataset_name}")
        if (result := type1_check(question_record, idx, dataset_name)) is not None
    ]

    write_jsonl(output_path, results)

    return results, compute_detection_accuracy(
        questions,
        results,
        "type1",
        question_ids,
    )


def main() -> None:
    dataset_names = DEFAULT_DATASETS
    results_root = DEFAULT_RESULTS_ROOT
    outputs_root = DEFAULT_OUTPUTS_ROOT
    summary: dict[str, Any] = {}

    for dataset_name in dataset_names:
        results, accuracy_stats = analyze_dataset(
            input_path_for_dataset(results_root, dataset_name),
            outputs_root / dataset_name / "Type1.jsonl",
            dataset_name,
        )
        summary[dataset_name] = {
            "num_detected_cases": len(results),
            "accuracy": accuracy_stats,
        }

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
