#!/usr/bin/env bash
set -euo pipefail

# Iterative Academy loop:
# 1) run one short training iteration
# 2) evaluate selected checkpoints
# 3) pick best checkpoint by stochastic avg_score_reward, then win_rate, then avg_goal_diff
# 4) save one video for the best checkpoint
# 5) STOP and wait for human decision
#
# It does NOT start the next iteration automatically.
#
# Usage examples:
#   bash Y_Fu/scripts/iterative_academy_loop.sh
#   bash Y_Fu/scripts/iterative_academy_loop.sh --timesteps 800000 --tag v3
#
# Assumed repo layout:
# <repo_root>/
#   .venv_yfu_grf_sys/
#   Y_Fu/
#     train.py
#     evaluate.py
#     checkpoints/
#     logs/
#     videos/
#     monitor_logs/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YFU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${YFU_DIR}/.." && pwd)"

PYTHON_BIN="${REPO_ROOT}/.venv_yfu_grf_sys/bin/python"
TRAIN_PY="${REPO_ROOT}/Y_Fu/train.py"
EVAL_PY="${REPO_ROOT}/Y_Fu/evaluate.py"

TIMESTEPS=400000
TAG="iter"
PRESET="academy_pass_and_shoot_with_keeper"
SEED=42
NUM_ENVS=4
ROLLOUT_STEPS=192
UPDATE_EPOCHS=4
NUM_MINIBATCHES=1
SAVE_INTERVAL=5
DEVICE="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timesteps) TIMESTEPS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python not found at ${PYTHON_BIN}"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="academy_${TAG}_${STAMP}"
SAVE_DIR="${REPO_ROOT}/Y_Fu/checkpoints/${RUN_NAME}"
LOGDIR="${REPO_ROOT}/Y_Fu/logs/${RUN_NAME}"
MONITOR_DIR="${REPO_ROOT}/Y_Fu/monitor_logs"
EVAL_DIR="${REPO_ROOT}/Y_Fu/eval_runs/${RUN_NAME}"
VIDEO_DIR="${REPO_ROOT}/Y_Fu/videos/${RUN_NAME}"

mkdir -p "${SAVE_DIR}" "${LOGDIR}" "${MONITOR_DIR}" "${EVAL_DIR}" "${VIDEO_DIR}"

TRAIN_LOG="${MONITOR_DIR}/${RUN_NAME}.log"
SUMMARY_FILE="${EVAL_DIR}/summary.tsv"

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
    current_step="$(extract_step "${TRAIN_LOG}")"
    if [[ "${current_step}" -gt 0 && "${elapsed}" -gt 0 ]]; then
      speed="$(awk "BEGIN { printf \"%.1f\", ${current_step}/${elapsed} }")"
      remaining="$(awk "BEGIN { if (${current_step}>0) printf \"%d\", ((${TIMESTEPS}-${current_step})/(${current_step}/${elapsed})); else print -1 }")"
      if [[ "${current_step}" -ge "${TIMESTEPS}" ]]; then
        remaining=0
      fi
      draw_bar "${current_step}" "${TIMESTEPS}"
      printf " | elapsed %s | left %s | step %s/%s | %s step/s\r" \
        "$(format_time "${elapsed}")" \
        "$(format_time "${remaining}")" \
        "${current_step}" "${TIMESTEPS}" "${speed}"
    else
      draw_bar 0 "${TIMESTEPS}"
      printf " | elapsed %s | left --:--:-- | step 0/%s | warming-up\r" \
        "$(format_time "${elapsed}")" "${TIMESTEPS}"
    fi
    sleep 2
  done
  echo
}

printf "checkpoint\tmode\tavg_return\tavg_score_reward\tavg_goal_diff\twin_rate\tavg_length\tlogfile\n" > "${SUMMARY_FILE}"

echo "Run name   : ${RUN_NAME}"
echo "Save dir   : ${SAVE_DIR}"
echo "Log dir    : ${LOGDIR}"
echo "Train log  : ${TRAIN_LOG}"
echo "Eval dir   : ${EVAL_DIR}"
echo "Video dir  : ${VIDEO_DIR}"
echo

CMD=(
  "${PYTHON_BIN}" -u "${TRAIN_PY}"
  --preset "${PRESET}"
  --use-player-id
  --num-envs "${NUM_ENVS}"
  --rollout-steps "${ROLLOUT_STEPS}"
  --total-timesteps "${TIMESTEPS}"
  --save-interval "${SAVE_INTERVAL}"
  --update-epochs "${UPDATE_EPOCHS}"
  --num-minibatches "${NUM_MINIBATCHES}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --save-dir "${SAVE_DIR}"
  --logdir "${LOGDIR}"
)

printf "Command    : "
printf "%q " "${CMD[@]}"
echo
echo

(
  cd "${REPO_ROOT}" || exit 1
  "${CMD[@]}" > "${TRAIN_LOG}" 2>&1
) &
TRAIN_PID=$!
monitor_loop
wait "${TRAIN_PID}"
RC=$?

echo
echo "Training finished with exit code: ${RC}"
if [[ "${RC}" -ne 0 ]]; then
  echo "Last 40 training log lines:"
  tail -n 40 "${TRAIN_LOG}" || true
  exit "${RC}"
fi

# Build a practical checkpoint subset from what's actually saved.
mapfile -t CKPTS < <(
  cd "${SAVE_DIR}" && ls update_*.pt latest.pt 2>/dev/null | sed '/^$/d' | sort -V
)

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "ERROR: no checkpoints found in ${SAVE_DIR}"
  exit 1
fi

# Select a sparse subset plus latest
SELECTED=()
for target in 10 20 30 50 75 100 125 150 175 200 230 260; do
  if [[ -f "${SAVE_DIR}/update_${target}.pt" ]]; then
    SELECTED+=("update_${target}.pt")
  fi
done
if [[ -f "${SAVE_DIR}/latest.pt" ]]; then
  SELECTED+=("latest.pt")
fi
if [[ "${#SELECTED[@]}" -eq 0 ]]; then
  SELECTED=("${CKPTS[@]}")
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

echo
echo "Evaluating checkpoints..."
BEST_CKPT=""
BEST_SCORE="-999"
BEST_WIN="-999"
BEST_GD="-999"

for ckpt in "${SELECTED[@]}"; do
  CKPT_PATH="${SAVE_DIR}/${ckpt}"
  STOCH_LOG="${EVAL_DIR}/${ckpt%.pt}_stochastic.log"

  echo "  stochastic eval: ${ckpt}"
  "${PYTHON_BIN}" "${EVAL_PY}" \
    --checkpoint "${CKPT_PATH}" \
    --episodes 20 \
    --device cpu | tee "${STOCH_LOG}"

  LINE="$(extract_final_line "${STOCH_LOG}" || true)"
  if [[ -n "${LINE}" ]]; then
    AVG_RETURN="$(extract_metric "${LINE}" "avg_return")"
    AVG_SCORE="$(extract_metric "${LINE}" "avg_score_reward")"
    AVG_GD="$(extract_metric "${LINE}" "avg_goal_diff")"
    WIN_RATE="$(extract_metric "${LINE}" "win_rate")"
    AVG_LEN="$(extract_metric "${LINE}" "avg_length")"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${ckpt}" "stochastic" "${AVG_RETURN}" "${AVG_SCORE}" "${AVG_GD}" "${WIN_RATE}" "${AVG_LEN}" "${STOCH_LOG}" \
      >> "${SUMMARY_FILE}"

    # Pick best by score_reward, then win_rate, then goal_diff.
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

echo "Best checkpoint: ${BEST_CKPT}"
echo "Best stochastic avg_score_reward: ${BEST_SCORE}"
echo "Best stochastic win_rate: ${BEST_WIN}"
echo "Best stochastic avg_goal_diff: ${BEST_GD}"
echo

BEST_VIDEO_DIR="${VIDEO_DIR}/${BEST_CKPT%.pt}"
echo "Saving one video for best checkpoint to ${BEST_VIDEO_DIR}"
"${PYTHON_BIN}" "${EVAL_PY}" \
  --checkpoint "${SAVE_DIR}/${BEST_CKPT}" \
  --episodes 1 \
  --device cpu \
  --save-video \
  --video-dir "${BEST_VIDEO_DIR}" | tee "${EVAL_DIR}/${BEST_CKPT%.pt}_video.log"

echo
echo "DONE."
echo "Next action:"
echo "  1) upload the saved video here if you want me to assess the behavior directly"
echo "  2) use the summary table + video to decide whether to patch reward again or scale timesteps"
