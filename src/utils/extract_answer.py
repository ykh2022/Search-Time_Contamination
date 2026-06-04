from __future__ import annotations

import re
import string
from typing import Awaitable, Callable
from typing import Literal
from pydantic import BaseModel
from openai import AsyncOpenAI

from utils.llm import deepseek_completion_async


PROMPT_MCQ = """
You are given:
- Options
- An LLM reasoning process

Extract the final selected option from the reasoning.
If the reasoning clearly indicates an answer, return ONLY one uppercase letter (e.g. A).
If the reasoning does not clearly indicate any final answer, return None

Options: {option}
Reasoning: {reasoning}
"""

PROMPT_HLE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
"""

VALID_OPTIONS = set(string.ascii_uppercase[:5])
LLMFunc = Callable[[str, str | None], Awaitable[str]]


def parse_llm_results(text: str | None) -> str | None:
    if not text:
        return None

    s = str(text).strip().upper()
    first_line = s.splitlines()[0].strip() if s else ""
    if first_line in {"NONE"}:
        return "None"

    if first_line in VALID_OPTIONS:
        return first_line

    match = re.search(r"\b([A-E])\b", s)
    if match:
        return match.group(1)
    return None


async def extract_answer_from_text(
    text: str,
    options: str,
    question_id: int,
    turn_id: int,
    llm_func: LLMFunc = deepseek_completion_async,
) -> str | None:
    """Use an LLM to extract the final answer option from a reasoning string."""
    if not text:
        return None
    try:
        user_prompt = PROMPT_MCQ.format(option=options, reasoning=text)
        response = await llm_func(user_prompt, None)
        parsed = parse_llm_results(response)
        if parsed is not None:
            return parsed
    except Exception as e:
        print(f"Error processing question_id={question_id} turn_id={turn_id}: {e}", flush=True)
        raise

    return None

class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] # 100% reliability

client = AsyncOpenAI(timeout=300.0, max_retries=3)

async def extract_answer_hle(question, correct_answer, response):
    prompt = PROMPT_HLE.format(question=question, correct_answer=correct_answer, response=response)
    try:
        response = await client.beta.chat.completions.parse(
                model="o3-mini-2025-01-31",
                max_completion_tokens=4096, # overkill for judge
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format=ExtractedAnswer, 
            ) 
        content = response.choices[0].message.parsed
        return { 
            "correct_answer": correct_answer,
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct,
            "confidence": content.confidence
        }
    except Exception as e: # very, very rare
        print("Error type:", type(e).__name__, flush=True)
        print("Error repr:", repr(e), flush=True)
        cause = getattr(e, "__cause__", None)
        if cause is not None:
            print("Error cause:", repr(cause), flush=True)
        raise
