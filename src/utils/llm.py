# LLM calling functions and prompt
import os
from typing import List, Dict, Any

from openai import AsyncOpenAI, APIConnectionError, RateLimitError, Timeout
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


os.environ["TOKENIZERS_PARALLELISM"] = "false"

SYSTEM_PROMPTv1 = """
You are a strictly objective Data Contamination Auditor. Your task is to analyze 'Web Search Content' to determine if an AI system bypassed genuine reasoning through "Search-Time Contamination" by retrieving the exact benchmark question and its corresponding answer key directly from the web.

Inputs you will receive:
1. [Question]: The original benchmark question.
2. [Ground-Truth Answer]: The correct answer.
3. [Web Search Content]: The retrieved context to be evaluated.

Your goal is to distinguish between "Search-Time Contamination" (contaminated=true) and "Legitimate Knowledge Retrieval" (contaminated=false).

### Evaluation Criteria:
Set `contaminated = true` ONLY IF BOTH of the following strict conditions are simultaneously met:
1. EXACT Question Fingerprint Match: The search content contains extended, contiguous blocks of verbatim text from the original question.
   - EXCLUSION (Crucial): It MUST NOT be a merely "similar" or analogous question. If the content describes a parallel scenario but uses different variables, numbers, patient demographics (e.g., age, gender), or specific lab values, it is NOT an exact match. Mere keyword overlap is insufficient.

2. DIRECT Answer Leakage: The exact Ground-Truth Answer or correct option label MUST be explicitly paired with the matched question text in the search content (e.g., within the same Q&A block, flashcard format, or forum post).
   - EXCLUSION (Crucial): If the web content contains the exact question but DOES NOT explicitly provide the ground-truth answer key, you MUST set `contaminated = false`. Retrieving an unanswered question does not constitute answer leakage.

Set `contaminated = false` (Legitimate Knowledge Retrieval) in ALL other cases, strictly including when:
- The content is a similar practice question with altered details or different variables.
- The content contains the exact question but lacks the answer.
- The content provides general encyclopedic knowledge, literature snippets, or clinical guidelines that help *deduce* the answer, but lacks the exact Q&A pair.

### Output Format:
You must output a strictly valid JSON object with the following structure:
{
  "reasoning": "Step 1: Check for EXACT string matching (reject similar questions with different variables). Step 2: Check if the ground-truth answer is explicitly provided alongside the question. Step 3: Conclude contamination status based on the strict AND logic.",
  "contaminated": true or false
}
"""

USER_PROMPT = """
[QUESTION]: {question}
[GROUND-TRUTH ANSWER]: {answer_text}
[Web search Content]: {response_text}
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_completion_async(
    prompt,
    system_prompt=None,
) -> str:
    openai_client = AsyncOpenAI()
    input_items: List[Dict[str, Any]] = []
    input_items.append({"role": "user", "content": prompt})
    resp = await openai_client.responses.create(
        model="gpt-5-mini",
        input=input_items,
        instructions=system_prompt,
        reasoning={"effort": "medium"},
    )

    return resp.output_text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def deepseek_completion_async(
    prompt,
    system_prompt=None,
) -> str:
    ds_client = AsyncOpenAI(
        api_key="your_deepseek_api_key_here",
        base_url="https://api.deepseek.com",
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = await ds_client.chat.completions.create(
        model="deepseek-v4-pro", messages=messages
    )

    return response.choices[0].message.content
