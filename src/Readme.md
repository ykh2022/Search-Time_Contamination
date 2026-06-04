# Search-Time Contamination Detection

This directory contains the detection pipeline for Tongyi DeepResearch outputs. The pipeline detects three types of search-time contamination, removes overlapping Type2/Type3 turns, and annotates each detected turn with before/after answer correctness.

## Preparation

### Input and Output

Default input root:

```text
../output/Tongyi_DeepResearch
```

Each dataset is expected as one JSONL file:

```text
Tongyi_DeepResearch_<DATASET>.jsonl
```

Default datasets:

```text
HLE149
MedBulltes5op
MedMCQA
MedQA
MMLU
MedXpertQA
```

Default output root:

```text
../detection/Tongyi_DeepResearch
```

For each dataset, the pipeline writes:

```text
<OUTPUTS_ROOT>/<DATASET>/Type1.jsonl
<OUTPUTS_ROOT>/<DATASET>/Type2.jsonl
<OUTPUTS_ROOT>/<DATASET>/Type3.jsonl
```

For `HLE149`, answer extraction reasoning is also written to:

```text
<OUTPUTS_ROOT>/HLE149/hle_answer_extraction_reasons.jsonl
```

### API Keys

Type3 and answer extraction call external LLM APIs.

You can use either OpenAI or DeepSeek.

For OpenAI, put the API key in the environment:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

For DeepSeek, update the `api_key` argument inside `deepseek_completion_async()`:

```python
ds_client = AsyncOpenAI(
    api_key="your_deepseek_api_key",
    base_url="https://api.deepseek.com",
)
```

For safer local use, prefer reading the DeepSeek key from an environment variable, for example `DEEPSEEK_API_KEY`, instead of committing a real key into the source file.

## Run the Full Pipeline

Run the command from the `src` directory:

```bash
bash run.sh
```

Write outputs elsewhere:

```bash
bash run.sh --outputs-root ../detection/try/Tongyi_DeepResearch
```

Useful environment variables:

```bash
OUTPUTS_ROOT=../detection/Tongyi_DeepResearch
DETECTION_LIMIT=0
TYPE3_MODEL=deepseek
TYPE3_CONCURRENCY=20
ANSWER_CONCURRENCY=25
```

Environment variable meanings:

- `OUTPUTS_ROOT`: Root directory for detection outputs.
- `DETECTION_LIMIT`: Maximum number of loaded questions to process per dataset for each Type detector. `0` means no limit.
- `TYPE3_MODEL`: LLM provider used by Type3. Supported values are `gpt` and `deepseek`.
- `TYPE3_CONCURRENCY`: Number of concurrent Type3 LLM judge requests.
- `ANSWER_CONCURRENCY`: Number of concurrent before/after answer extraction tasks.

Example test run on the first 10 loaded questions:

```bash
DETECTION_LIMIT=10 bash run.sh --outputs-root ../detection/try/Tongyi_DeepResearch
```

**`run.sh` executes five steps**:

1. `type1_BML.py`
   Detects Type1 BML cases from search URL/domain/keyword patterns.

2. `type2_QCL.py`
   Detects Type2 QCL cases by matching visited page/tool-response text against the original question.

3. `type3_EAL.py`
   Detects Type3 EAL cases using LLM-as-a-judge.

4. `remove_overlap.py`
   Removes Type2 turns whose `(id, turn)` pair also appears in Type3. This edits `Type2.jsonl` in place.

5. `before_after_extraction.py`
   Use LLM-as-a-judge to detect answer before/after contamination cases. For `HLE149`, detailed LLM reasoning is saved separately and omitted from the main Type files.

## Output Record Format

Each detected case is a JSON object similar to:

```json
{
  "id": 7,
  "prefix": "...",
  "total_turn": 12,
  "cheating_turn": [
    {
      "turn": 4,
      "cheating_type": 1,
      "cheating_reasons": [],
      "before_cheat_answer": "A",
      "before_correct": true,
      "after_cheat_answer": "B",
      "after_correct": false
    }
  ],
  "final_answer": "A",
  "ground_truth": "A"
}
```

Field meanings:

- `id`: Question id from the loaded input file.
- `prefix`: A short slice of the question text, used for quick inspection.
- `total_turn`: Total number of assistant turns in this question's trajectory.
- `cheating_turn`: List of turns detected as contaminated. Each item corresponds to one detected turn.
- `turn`: The 1-based turn number in the trajectory.
- `cheating_type`: Detection type. `1` means BML, `2` means QCL, and `3` means EAL.
- `cheating_reasons`: Evidence for the detection. Type1 usually stores matched keywords/URLs; Type2 stores overlap/matching evidence; Type3 stores the LLM judge reason.
- `before_cheat_answer`: Extracted answer before the detected cheating turn takes effect.
- `before_correct`: Whether `before_cheat_answer` matches the ground truth. Can be `true`, `false`, or `null` when correctness cannot be determined.
- `after_cheat_answer`: Extracted answer after the detected cheating turn.
- `after_correct`: Whether `after_cheat_answer` matches the ground truth. Can be `true`, `false`, or `null` when correctness cannot be determined.
- `final_answer`: Final model answer from the original trajectory.
- `ground_truth`: Ground-truth answer for the question.
