#!/usr/bin/env bash
set -u

# Robust launcher for the repo layout:
# <repo_root>/
#   .venv_yfu_grf_sys/
#   Y_Fu/
#     train.py
#     scripts/
#       run_training_with_eta.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YFU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${YFU_DIR}/.." && pwd)"

PYTHON_BIN="${REPO_ROOT}/.venv_yfu_grf_sys/bin/python"
TRAIN_PY="${REPO_ROOT}/Y_Fu/train.py"
MONITOR_DIR="${REPO_ROOT}/Y_Fu/monitor_logs"

mkdir -p "${MONITOR_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python not found at:"
  echo "  ${PYTHON_BIN}"
  echo
  echo "Your repo layout suggests the venv should be at repo root:"
  echo "  ${REPO_ROOT}/.venv_yfu_grf_sys"
  exit 1
fi

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "ERROR: train.py not found at:"
  echo "  ${TRAIN_PY}"
  exit 1
fi

MODE="${1:-academy_reboot}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE=""

build_command() {
  case "${MODE}" in
    academy_reboot)
      TOTAL_STEPS=400000
      RUN_NAME="academy_pass_reboot_v1_${STAMP}"
      LOG_FILE="${MONITOR_DIR}/${RUN_NAME}.log"
      CMD=(
        "${PYTHON_BIN}" -u "${TRAIN_PY}"
        --preset academy_pass_and_shoot_with_keeper
        --use-player-id
        --num-envs 4
        --rollout-steps 192
        --total-timesteps 400000
        --save-interval 5
        --update-epochs 4
        --num-minibatches 1
        --device cpu
        --seed 42
        --save-dir "${REPO_ROOT}/Y_Fu/checkpoints/academy_pass_reboot_v1"
        --logdir "${REPO_ROOT}/Y_Fu/logs/academy_pass_reboot_v1"
        --init-checkpoint "${REPO_ROOT}/Y_Fu/checkpoints/academy_pass_reboot_v1/update_5.pt"
      )
      ;;
    fivev5_v2)
      TOTAL_STEPS=2000000
      RUN_NAME="fivev5_v2_${STAMP}"
      LOG_FILE="${MONITOR_DIR}/${RUN_NAME}.log"
      CMD=(
        "${PYTHON_BIN}" -u "${TRAIN_PY}"
        --preset five_vs_five_reward_v2
        --device cpu
        --num-envs 4
        --rollout-steps 256
        --update-epochs 4
        --num-minibatches 1
        --total-timesteps 2000000
        --init-checkpoint "${REPO_ROOT}/Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_10.pt"
        --seed 42
      )
      ;;
    fivev5_v2b)
      TOTAL_STEPS=2000000
      RUN_NAME="fivev5_v2b_transition_${STAMP}"
      LOG_FILE="${MONITOR_DIR}/${RUN_NAME}.log"
      CMD=(
        "${PYTHON_BIN}" -u "${TRAIN_PY}"
        --preset five_vs_five_reward_v2b_transition
        --device cpu
        --num-envs 4
        --rollout-steps 256
        --update-epochs 4
        --num-minibatches 1
        --total-timesteps 2000000
        --init-checkpoint "${REPO_ROOT}/Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_10.pt"
        --seed 42
      )
      ;;
    fivev5_v2c)
      TOTAL_STEPS=2000000
      RUN_NAME="fivev5_v2c_progression_${STAMP}"
      LOG_FILE="${MONITOR_DIR}/${RUN_NAME}.log"
      CMD=(
        "${PYTHON_BIN}" -u "${TRAIN_PY}"
        --preset five_vs_five_reward_v2c_progression
        --device cpu
        --num-envs 4
        --rollout-steps 256
        --update-epochs 4
        --num-minibatches 1
        --total-timesteps 2000000
        --init-checkpoint "${REPO_ROOT}/Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_10.pt"
        --seed 42
      )
      ;;
    *)
      echo "Unknown mode: ${MODE}"
      echo "Use one of: academy_reboot | fivev5_v2 | fivev5_v2b | fivev5_v2c"
      exit 1
      ;;
  esac
}

format_time() {
  local secs="${1}"
  if [[ "${secs}" -lt 0 ]]; then
    secs=0
  fi
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
    current_step="$(extract_step "${LOG_FILE}")"

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

build_command

echo "Mode      : ${MODE}"
echo "Script dir: ${SCRIPT_DIR}"
echo "Y_Fu dir  : ${YFU_DIR}"
echo "Repo root : ${REPO_ROOT}"
echo "Python    : ${PYTHON_BIN}"
echo "Train py  : ${TRAIN_PY}"
echo "Log file  : ${LOG_FILE}"
printf "Command   : "
printf "%q " "${CMD[@]}"
echo
echo

(
  cd "${REPO_ROOT}" || exit 1
  "${CMD[@]}" > "${LOG_FILE}" 2>&1
) &
TRAIN_PID=$!

monitor_loop
wait "${TRAIN_PID}"
RC=$?

echo
echo "Training finished with exit code: ${RC}"
echo "Log saved to: ${LOG_FILE}"

if [[ "${RC}" -ne 0 ]]; then
  echo
  echo "Last 40 log lines:"
  tail -n 40 "${LOG_FILE}" 2>/dev/null || true
fi

exit "${RC}"
