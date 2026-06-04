import json
import re
from pathlib import Path

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
ANY_WORD_RE = re.compile(r"[a-z0-9]+")
HALF_BY_QUESTION_LENGTH_DATASETS = {"mmlu", "medmcqa"}

def clean_text(text):
    return text.strip() if text else ""

def maybe_parse_json(text):
    text = clean_text(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text

def extract_tool_calls(content):
    matches = TOOL_CALL_RE.findall(content or "")
    if not matches:
        return []
    return [maybe_parse_json(match) for match in matches]

def extract_tool_responses(content):
    matches = TOOL_RESPONSE_RE.findall(content or "")
    if not matches:
        return []
    return [clean_text(match) for match in matches]

def extract_answer(content):
    matches = ANSWER_RE.findall(content or "")
    if not matches:
        return ""
    return clean_text(matches[-1])

def split_question_and_options(question_text):
    raw_text = str(question_text or "")
    normalized_text = raw_text.replace("\\n", "\n")
    parts = re.split(r"(?im)\n\s*options\s*:\s*", normalized_text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return normalized_text.strip(), ""

def question_word_count(record):
    question_text, _ = split_question_and_options(record.get("question", ""))
    normalized_text = re.sub(r"\s+", " ", question_text).strip().lower()
    return len(ANY_WORD_RE.findall(normalized_text))

def make_turn(assistant_content=""):
    return {
        "assistant": clean_text(assistant_content),
        "tools": extract_tool_calls(assistant_content),
        "answers": extract_answer(assistant_content),
        "tool_responses": [],
    }

def parse_messages(messages):
    parsed = {
        "system": "",
        "user": [],
        "turns": [],
    }
    current_turn = None
    awaiting_tool_response = False
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            parsed["system"] = clean_text(content)
            continue
        if role == "assistant":
            current_turn = make_turn(content)
            parsed["turns"].append(current_turn)
            awaiting_tool_response = bool(current_turn["tools"])
            continue
        if role == "user":
            tool_responses = extract_tool_responses(content)
            if current_turn is not None and awaiting_tool_response:
                payload = tool_responses or [clean_text(content)]
                current_turn["tool_responses"].extend(payload)
                awaiting_tool_response = False
            else:
                parsed["user"].append(clean_text(content))
            continue
    return parsed

def load_data(data_path):
    questions = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            item = json.loads(line)
            parsed = parse_messages(item.get("messages", []))
            item.pop("messages")
            item["id"] = i
            item["new_messages"] = parsed
            item["correct"] = item.get("prediction") == item.get("answer")
            questions.append(item)
    return questions

def load_judged_data(judged_path):
    judged_path = Path(judged_path)
    turns_path = judged_path.with_name(judged_path.name.replace("_judged.json", ".jsonl"))

    with judged_path.open("r", encoding="utf-8") as fh:
        judged_records = json.load(fh)
    turn_records = load_data(turns_path)

    questions = []
    for order_idx, judged_record in enumerate(judged_records):
        source_idx = judged_record.get("index")
        turn_idx = source_idx if isinstance(source_idx, int) else order_idx
        if turn_idx >= len(turn_records):
            continue

        question_record = dict(turn_records[turn_idx])
        question_record["id"] = turn_idx + 1
        judge_response = judged_record.get("judge_response") or {}
        question_record["question"] = judged_record.get(
            "question",
            question_record.get("question", ""),
        )
        question_record["answer"] = judged_record.get(
            "correct_answer",
            judge_response.get("correct_answer", question_record.get("answer", "")),
        )
        question_record["prediction"] = judge_response.get(
            "model_answer",
            judged_record.get("response", question_record.get("prediction")),
        )
        question_record["judge_response"] = judge_response
        question_record["correct"] = judge_response.get("correct") == "yes"
        questions.append(question_record)

    return questions

def load_analysis_data(data_path):
    path = Path(data_path)
    if path.name in {"iter1_judged.json", "iter2_judged.json"}:
        questions = load_judged_data(path)
    else:
        questions = load_data(path)

    path_text = str(path).lower()
    if any(dataset_name in path_text for dataset_name in HALF_BY_QUESTION_LENGTH_DATASETS):
        indexed_questions = list(enumerate(questions))
        indexed_questions.sort(
            key=lambda item: (
                question_word_count(item[1]),
                -int(item[1].get("id", item[0] + 1)),
            ),
            reverse=True,
        )
        selected_indices = {
            idx for idx, _ in indexed_questions[: len(indexed_questions) // 2]
        }
        questions = [
            question
            for idx, question in enumerate(questions)
            if idx in selected_indices
        ]

    return questions

if __name__ == "__main__":
    questions = load_analysis_data("../results/MedMCQA/iter1_extracted.jsonl")
    output_path = Path("outputs/MedMCQA_half.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in questions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
