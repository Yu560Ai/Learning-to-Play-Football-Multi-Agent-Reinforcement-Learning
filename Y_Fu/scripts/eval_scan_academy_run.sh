#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YFU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${YFU_DIR}/.." && pwd)"

PYTHON_BIN="${REPO_ROOT}/.venv_yfu_grf_sys/bin/python"
EVAL_PY="${REPO_ROOT}/Y_Fu/evaluate.py"

RUN_DIR_NAME="${1:-}"
if [[ -z "${RUN_DIR_NAME}" ]]; then
  echo "Usage: bash Y_Fu/scripts/eval_scan_academy_run.sh <checkpoint_dir_name>"
  exit 1
fi

CKPT_DIR="${REPO_ROOT}/Y_Fu/checkpoints/${RUN_DIR_NAME}"
STAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_DIR="${REPO_ROOT}/Y_Fu/eval_runs/${RUN_DIR_NAME}_scan_${STAMP}"
VIDEO_DIR="${REPO_ROOT}/Y_Fu/videos/${RUN_DIR_NAME}_scan_${STAMP}"
SUMMARY_FILE="${EVAL_DIR}/summary.tsv"

mkdir -p "${EVAL_DIR}" "${VIDEO_DIR}"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "ERROR: checkpoint directory not found: ${CKPT_DIR}"
  exit 1
fi

extract_final_line() {
  local logfile="$1"
  grep 'trained_policy:' "${logfile}" | tail -n 1
}

extract_metric() {
  local line="$1"
  local key="$2"
  echo "${line}" | sed -n "s/.*${key}=\([^ ]*\).*/\1/p"
}

printf "checkpoint\tavg_return\tavg_score_reward\tavg_goal_diff\twin_rate\tavg_length\tlogfile\n" > "${SUMMARY_FILE}"

SELECTED=()
for target in 10 20 30 50 75 100 125 150 175 200 230 260; do
  if [[ -f "${CKPT_DIR}/update_${target}.pt" ]]; then
    SELECTED+=("update_${target}.pt")
  fi
done
if [[ -f "${CKPT_DIR}/latest.pt" ]]; then
  SELECTED+=("latest.pt")
fi
if [[ "${#SELECTED[@]}" -eq 0 ]]; then
  mapfile -t SELECTED < <(cd "${CKPT_DIR}" && ls update_*.pt latest.pt 2>/dev/null | sed '/^$/d' | sort -V)
fi

BEST_CKPT=""
BEST_SCORE="-999"
BEST_WIN="-999"
BEST_GD="-999"

for ckpt in "${SELECTED[@]}"; do
  CKPT_PATH="${CKPT_DIR}/${ckpt}"
  LOGFILE="${EVAL_DIR}/${ckpt%.pt}.log"
  echo "Evaluating ${ckpt} ..."
  "${PYTHON_BIN}" "${EVAL_PY}" --checkpoint "${CKPT_PATH}" --episodes 20 --device cpu | tee "${LOGFILE}"
  LINE="$(extract_final_line "${LOGFILE}" || true)"
  if [[ -n "${LINE}" ]]; then
    AVG_RETURN="$(extract_metric "${LINE}" "avg_return")"
    AVG_SCORE="$(extract_metric "${LINE}" "avg_score_reward")"
    AVG_GD="$(extract_metric "${LINE}" "avg_goal_diff")"
    WIN_RATE="$(extract_metric "${LINE}" "win_rate")"
    AVG_LEN="$(extract_metric "${LINE}" "avg_length")"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${ckpt}" "${AVG_RETURN}" "${AVG_SCORE}" "${AVG_GD}" "${WIN_RATE}" "${AVG_LEN}" "${LOGFILE}" >> "${SUMMARY_FILE}"

    if awk "BEGIN {exit !(${AVG_SCORE} > ${BEST_SCORE})}"; then
      BEST_CKPT="${ckpt}"
      BEST_SCORE="${AVG_SCORE}"
      BEST_WIN="${WIN_RATE}"
      BEST_GD="${AVG_GD}"
    elif awk "BEGIN {exit !(${AVG_SCORE} == ${BEST_SCORE} && ${WIN_RATE} > ${BEST_WIN})}"; then
      BEST_CKPT="${ckpt}"
      BEST_SCORE="${AVG_SCORE}"
      BEST_WIN="${WIN_RATE}"
      BEST_GD="${AVG_GD}"
    elif awk "BEGIN {exit !(${AVG_SCORE} == ${BEST_SCORE} && ${WIN_RATE} == ${BEST_WIN} && ${AVG_GD} > ${BEST_GD})}"; then
      BEST_CKPT="${ckpt}"
      BEST_SCORE="${AVG_SCORE}"
      BEST_WIN="${WIN_RATE}"
      BEST_GD="${AVG_GD}"
    fi
  fi
done

echo
echo "Summary:"
column -t -s $'\t' "${SUMMARY_FILE}" || cat "${SUMMARY_FILE}"
echo

if [[ -z "${BEST_CKPT}" ]]; then
  echo "No best checkpoint selected."
  exit 1
fi

BEST_VIDEO_DIR="${VIDEO_DIR}/${BEST_CKPT%.pt}"
echo "Best checkpoint: ${BEST_CKPT}"
echo "Saving video to ${BEST_VIDEO_DIR}"
"${PYTHON_BIN}" "${EVAL_PY}" \
  --checkpoint "${CKPT_DIR}/${BEST_CKPT}" \
  --episodes 1 \
  --device cpu \
  --save-video \
  --video-dir "${BEST_VIDEO_DIR}" | tee "${EVAL_DIR}/${BEST_CKPT%.pt}_video.log"

echo
echo "Done."
echo "Summary file: ${SUMMARY_FILE}"
echo "Video dir: ${BEST_VIDEO_DIR}"
