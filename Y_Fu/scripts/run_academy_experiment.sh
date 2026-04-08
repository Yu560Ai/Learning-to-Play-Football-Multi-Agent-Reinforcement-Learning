#!/usr/bin/env bash
set -euo pipefail

# Multi-run launcher for Academy experiments with isolated output folders.
# Supports separate save/log/monitor dirs per run so multiple strategies can coexist.
#
# IMPORTANT:
# Running multiple rollout-heavy GRF jobs on the SAME CPU machine will slow all runs.
# Use parallel only if you accept slower/noisier throughput or have separate machines.
#
# Examples:
#   bash Y_Fu/scripts/run_academy_experiment.sh shoot_v1
#   bash Y_Fu/scripts/run_academy_experiment.sh passshoot_v1 800000
#   bash Y_Fu/scripts/run_academy_experiment.sh 3v1_v1 600000
#
# For background parallel runs:
#   nohup bash Y_Fu/scripts/run_academy_experiment.sh shoot_v1 800000 > Y_Fu/launch_logs/shoot_v1.out 2>&1 &
#   nohup bash Y_Fu/scripts/run_academy_experiment.sh passshoot_v1 800000 > Y_Fu/launch_logs/passshoot_v1.out 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YFU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${YFU_DIR}/.." && pwd)"

PYTHON_BIN="${REPO_ROOT}/.venv_yfu_grf_sys/bin/python"
TRAIN_PY="${REPO_ROOT}/Y_Fu/train.py"

MODE="${1:-shoot_v1}"
TOTAL_STEPS="${2:-800000}"
SEED="${3:-42}"
STAMP="$(date +%Y%m%d_%H%M%S)"

BASE_CKPT_DIR="${REPO_ROOT}/Y_Fu/checkpoints"
BASE_LOG_DIR="${REPO_ROOT}/Y_Fu/logs"
BASE_MONITOR_DIR="${REPO_ROOT}/Y_Fu/monitor_logs"
BASE_LAUNCH_LOG_DIR="${REPO_ROOT}/Y_Fu/launch_logs"

mkdir -p "${BASE_CKPT_DIR}" "${BASE_LOG_DIR}" "${BASE_MONITOR_DIR}" "${BASE_LAUNCH_LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python not found at ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "ERROR: train.py not found at ${TRAIN_PY}"
  exit 1
fi

NUM_ENVS=4
ROLLOUT_STEPS=192
UPDATE_EPOCHS=4
NUM_MINIBATCHES=1
SAVE_INTERVAL=5
DEVICE="cpu"
USE_PLAYER_ID=1
PRESET=""
RUN_NAME=""
INIT_CHECKPOINT=""

case "${MODE}" in
  shoot_v1)
    PRESET="academy_run_to_score_with_keeper"
    RUN_NAME="academy_shoot_v1_s${SEED}_${STAMP}"
    ;;
  shoot_v2_long)
    PRESET="academy_run_to_score_with_keeper"
    RUN_NAME="academy_shoot_v2_long_s${SEED}_${STAMP}"
    TOTAL_STEPS="${2:-1200000}"
    ;;
  passshoot_v1)
    PRESET="academy_pass_and_shoot_with_keeper"
    RUN_NAME="academy_passshoot_v1_s${SEED}_${STAMP}"
    ;;
  passshoot_v2_long)
    PRESET="academy_pass_and_shoot_with_keeper"
    RUN_NAME="academy_passshoot_v2_long_s${SEED}_${STAMP}"
    TOTAL_STEPS="${2:-1200000}"
    ;;
  threev1_v1|3v1_v1)
    PRESET="academy_3_vs_1_with_keeper"
    RUN_NAME="academy_3v1_v1_s${SEED}_${STAMP}"
    TOTAL_STEPS="${2:-600000}"
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Use one of:"
    echo "  shoot_v1"
    echo "  shoot_v2_long"
    echo "  passshoot_v1"
    echo "  passshoot_v2_long"
    echo "  threev1_v1"
    exit 1
    ;;
esac

SAVE_DIR="${BASE_CKPT_DIR}/${RUN_NAME}"
LOGDIR="${BASE_LOG_DIR}/${RUN_NAME}"
MONITOR_LOG="${BASE_MONITOR_DIR}/${RUN_NAME}.log"

mkdir -p "${SAVE_DIR}" "${LOGDIR}"

format_time() {
  local secs="${1}"
  if [[ "${secs}" -lt 0 ]]; then secs=0; fi
  printf "%02d:%02d:%02d" "$((secs/3600))" "$(((secs%3600)/60))" "$((secs%60))"
}

extract_step() {
  local logfile="${1}"
  if [[ ! -f "${logfile}" ]]; then
    echo 0
    return
  fi
  local step=""
  step="$(
    grep -Eo 'total_agent_steps[=: ]+[0-9]+' "${logfile}" 2>/dev/null | tail -n1 | grep -Eo '[0-9]+' | tail -n1
  )"
  if [[ -z "${step}" ]]; then
    step="$(
      grep -Eo 'agent_steps[=: ]+[0-9]+' "${logfile}" 2>/dev/null | tail -n1 | grep -Eo '[0-9]+' | tail -n1
    )"
  fi
  if [[ -z "${step}" ]]; then
    step="$(
      grep -Eo 'step[=: ]+[0-9]+' "${logfile}" 2>/dev/null | tail -n1 | grep -Eo '[0-9]+' | tail -n1
    )"
  fi
  echo "${step:-0}"
}

draw_bar() {
  local current="$1"
  local total="$2"
  local width=28
  local filled=0
  local pct=0
  if [[ "${total}" -gt 0 ]]; then
    filled=$(( current * width / total ))
    pct=$(( current * 100 / total ))
    [[ "${filled}" -gt "${width}" ]] && filled="${width}"
    [[ "${pct}" -gt 100 ]] && pct=100
  fi
  local empty=$(( width - filled ))
  printf "["
  printf "%0.s#" $(seq 1 "${filled}" 2>/dev/null)
  printf "%0.s-" $(seq 1 "${empty}" 2>/dev/null)
  printf "] %3d%%" "${pct}"
}

monitor_loop() {
  local start_ts
  start_ts="$(date +%s)"
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    local now elapsed current_step speed remaining
    now="$(date +%s)"
    elapsed=$(( now - start_ts ))
    current_step="$(extract_step "${MONITOR_LOG}")"
    if [[ "${current_step}" -gt 0 && "${elapsed}" -gt 0 ]]; then
      speed="$(awk "BEGIN { printf \"%.1f\", ${current_step}/${elapsed} }")"
      remaining="$(awk "BEGIN { if (${current_step}>0) printf \"%d\", ((${TOTAL_STEPS}-${current_step})/(${current_step}/${elapsed})); else print -1 }")"
      if [[ "${current_step}" -ge "${TOTAL_STEPS}" ]]; then
        remaining=0
      fi
      draw_bar "${current_step}" "${TOTAL_STEPS}"
      printf " | elapsed %s | left %s | step %s/%s | %s step/s\r" \
        "$(format_time "${elapsed}")" \
        "$(format_time "${remaining}")" \
        "${current_step}" "${TOTAL_STEPS}" "${speed}"
    else
      draw_bar 0 "${TOTAL_STEPS}"
      printf " | elapsed %s | left --:--:-- | step 0/%s | warming-up\r" \
        "$(format_time "${elapsed}")" "${TOTAL_STEPS}"
    fi
    sleep 2
  done
  echo
}

CMD=(
  "${PYTHON_BIN}" -u "${TRAIN_PY}"
  --preset "${PRESET}"
  --num-envs "${NUM_ENVS}"
  --rollout-steps "${ROLLOUT_STEPS}"
  --total-timesteps "${TOTAL_STEPS}"
  --save-interval "${SAVE_INTERVAL}"
  --update-epochs "${UPDATE_EPOCHS}"
  --num-minibatches "${NUM_MINIBATCHES}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --save-dir "${SAVE_DIR}"
  --logdir "${LOGDIR}"
)

if [[ "${USE_PLAYER_ID}" -eq 1 ]]; then
  CMD+=(--use-player-id)
fi

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  CMD+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

echo "Mode        : ${MODE}"
echo "Preset      : ${PRESET}"
echo "Run name    : ${RUN_NAME}"
echo "Repo root   : ${REPO_ROOT}"
echo "Save dir    : ${SAVE_DIR}"
echo "Log dir     : ${LOGDIR}"
echo "Monitor log : ${MONITOR_LOG}"
echo "Seed        : ${SEED}"
echo "Timesteps   : ${TOTAL_STEPS}"
printf "Command     : "
printf "%q " "${CMD[@]}"
echo
echo

(
  cd "${REPO_ROOT}" || exit 1
  "${CMD[@]}" > "${MONITOR_LOG}" 2>&1
) &
TRAIN_PID=$!

monitor_loop
wait "${TRAIN_PID}"
RC=$?

echo
echo "Training finished with exit code: ${RC}"
echo "Monitor log saved to: ${MONITOR_LOG}"
echo "Checkpoint dir: ${SAVE_DIR}"

if [[ "${RC}" -ne 0 ]]; then
  echo
  echo "Last 40 log lines:"
  tail -n 40 "${MONITOR_LOG}" 2>/dev/null || true
fi

exit "${RC}"
