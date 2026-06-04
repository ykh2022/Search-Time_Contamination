#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUTPUTS_ROOT="${OUTPUTS_ROOT:-../detection/Tongyi_DeepResearch}"
DETECTION_LIMIT="${DETECTION_LIMIT:-0}"
TYPE3_MODEL="${TYPE3_MODEL:-deepseek}"
TYPE3_CONCURRENCY="${TYPE3_CONCURRENCY:-20}"
ANSWER_CONCURRENCY="${ANSWER_CONCURRENCY:-25}"

usage() {
  cat <<'EOF'
Usage: bash run.bash [--outputs-root PATH]

Environment variables:
  OUTPUTS_ROOT           Detection output root (default: ../detection/Tongyi_DeepResearch)
  DETECTION_LIMIT        Max questions per dataset for each type; 0 means no limit (default: 0)
  TYPE3_MODEL            Type3 LLM provider: gpt or deepseek (default: deepseek)
  TYPE3_CONCURRENCY      Type3 concurrent LLM checks (default: 20)
  ANSWER_CONCURRENCY     Before/after answer extraction concurrency (default: 25)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--outputs-root)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      OUTPUTS_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

export OUTPUTS_ROOT
export DETECTION_LIMIT

echo "Using output root: ${OUTPUTS_ROOT}"
echo "Using per-dataset detection limit: ${DETECTION_LIMIT}"

echo "[1/5] Detecting Type1 cases"
python type1_BML.py

echo "[2/5] Detecting Type2 cases"
python type2_QCL.py

echo "[3/5] Detecting Type3 cases"
python type3_EAL.py --model "${TYPE3_MODEL}" --concurrency "${TYPE3_CONCURRENCY}"

echo "[4/5] Removing Type2 turns that overlap with Type3"
python remove_overlap.py

echo "[5/5] Annotating before/after answers"
python before_after_extraction.py --outputs-root "${OUTPUTS_ROOT}" --concurrency "${ANSWER_CONCURRENCY}"

echo "Done."
