#!/usr/bin/env bash
set -euo pipefail

# Batch evaluation script for academy_pass_reboot_v1 checkpoints.
# Purpose:
# 1) scan a useful subset of checkpoints
# 2) run stochastic and deterministic eval
# 3) save per-checkpoint logs
# 4) build a compact summary table for ranking
#
# Repo layout assumed:
# <repo_root>/
#   .venv_yfu_grf_sys/
#   Y_Fu/
#     evaluate.py
#     checkpoints/academy_pass_reboot_v1/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YFU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${YFU_DIR}/.." && pwd)"

PYTHON_BIN="${REPO_ROOT}/.venv_yfu_grf_sys/bin/python"
EVAL_PY="${REPO_ROOT}/Y_Fu/evaluate.py"
CKPT_DIR="${REPO_ROOT}/Y_Fu/checkpoints/academy_pass_reboot_v1"
OUT_ROOT="${REPO_ROOT}/Y_Fu/eval_runs"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/academy_pass_reboot_scan_${STAMP}"
SUMMARY_FILE="${RUN_DIR}/summary.tsv"

mkdir -p "${RUN_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python not found: ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -f "${EVAL_PY}" ]]; then
  echo "ERROR: evaluate.py not found: ${EVAL_PY}"
  exit 1
fi

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "ERROR: checkpoint dir not found: ${CKPT_DIR}"
  exit 1
fi

# Useful scan set:
# - early points
# - middle points
# - late points
# - latest
CHECKPOINTS=(
  update_10.pt
  update_20.pt
  update_30.pt
  update_50.pt
  update_75.pt
  update_100.pt
  update_125.pt
  update_150.pt
  update_175.pt
  update_200.pt
  update_230.pt
  update_260.pt
  latest.pt
)

EPISODES_STOCH="${EPISODES_STOCH:-20}"
EPISODES_DET="${EPISODES_DET:-10}"
SEED_DET="${SEED_DET:-123}"

extract_final_line() {
  local logfile="$1"
  grep 'trained_policy:' "${logfile}" | tail -n 1
}

extract_metric() {
  local line="$1"
  local key="$2"
  echo "${line}" | sed -n "s/.*${key}=\([^ ]*\).*/\1/p"
}

run_eval() {
  local ckpt_name="$1"
  local mode="$2"
  local logfile="$3"

  local ckpt_path="${CKPT_DIR}/${ckpt_name}"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "SKIP missing checkpoint: ${ckpt_path}"
    return 1
  fi

  echo
  echo "============================================================"
  echo "Checkpoint : ${ckpt_name}"
  echo "Mode       : ${mode}"
  echo "Log        : ${logfile}"
  echo "============================================================"

  if [[ "${mode}" == "stochastic" ]]; then
    "${PYTHON_BIN}" "${EVAL_PY}" \
      --checkpoint "${ckpt_path}" \
      --episodes "${EPISODES_STOCH}" \
      --device cpu | tee "${logfile}"
  else
    "${PYTHON_BIN}" "${EVAL_PY}" \
      --checkpoint "${ckpt_path}" \
      --episodes "${EPISODES_DET}" \
      --deterministic \
      --device cpu \
      --seed "${SEED_DET}" | tee "${logfile}"
  fi
}

printf "checkpoint\tmode\tavg_return\tavg_score_reward\tavg_goal_diff\twin_rate\tavg_length\tlogfile\n" > "${SUMMARY_FILE}"

echo "Run dir      : ${RUN_DIR}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Summary file : ${SUMMARY_FILE}"
echo "Stochastic episodes   : ${EPISODES_STOCH}"
echo "Deterministic episodes: ${EPISODES_DET}"
echo

for ckpt in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "${CKPT_DIR}/${ckpt}" ]]; then
    echo "Skipping missing ${ckpt}"
    continue
  fi

  # stochastic
  STOCH_LOG="${RUN_DIR}/${ckpt%.pt}_stochastic.log"
  run_eval "${ckpt}" "stochastic" "${STOCH_LOG}" || true
  STOCH_LINE="$(extract_final_line "${STOCH_LOG}" || true)"
  if [[ -n "${STOCH_LINE}" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${ckpt}" \
      "stochastic" \
      "$(extract_metric "${STOCH_LINE}" "avg_return")" \
      "$(extract_metric "${STOCH_LINE}" "avg_score_reward")" \
      "$(extract_metric "${STOCH_LINE}" "avg_goal_diff")" \
      "$(extract_metric "${STOCH_LINE}" "win_rate")" \
      "$(extract_metric "${STOCH_LINE}" "avg_length")" \
      "${STOCH_LOG}" >> "${SUMMARY_FILE}"
  fi

  # deterministic
  DET_LOG="${RUN_DIR}/${ckpt%.pt}_deterministic.log"
  run_eval "${ckpt}" "deterministic" "${DET_LOG}" || true
  DET_LINE="$(extract_final_line "${DET_LOG}" || true)"
  if [[ -n "${DET_LINE}" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${ckpt}" \
      "deterministic" \
      "$(extract_metric "${DET_LINE}" "avg_return")" \
      "$(extract_metric "${DET_LINE}" "avg_score_reward")" \
      "$(extract_metric "${DET_LINE}" "avg_goal_diff")" \
      "$(extract_metric "${DET_LINE}" "win_rate")" \
      "$(extract_metric "${DET_LINE}" "avg_length")" \
      "${DET_LOG}" >> "${SUMMARY_FILE}"
  fi
done

echo
echo "============================================================"
echo "DONE"
echo "Summary saved to: ${SUMMARY_FILE}"
echo "Quick view:"
echo "------------------------------------------------------------"
column -t -s $'\t' "${SUMMARY_FILE}" || cat "${SUMMARY_FILE}"
echo "------------------------------------------------------------"
echo
echo "Recommended next manual step:"
echo "1) inspect the stochastic rows first"
echo "2) pick the top 2-3 checkpoints by avg_score_reward, win_rate, and avg_goal_diff"
echo "3) record one video for each top candidate"
