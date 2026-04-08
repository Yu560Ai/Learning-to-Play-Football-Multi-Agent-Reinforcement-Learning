#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YFU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${YFU_DIR}/.." && pwd)"

PYTHON_BIN="${REPO_ROOT}/.venv_yfu_grf_sys/bin/python"
TRAIN_PY="${REPO_ROOT}/Y_Fu/train.py"
PPO_PY="${REPO_ROOT}/Y_Fu/yfu_football/ppo.py"

TIMESTEPS="${1:-800000}"
SEED="${2:-42}"
TAG="${3:-shootfirst}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="academy_${TAG}_s${SEED}_${STAMP}"

SAVE_DIR="${REPO_ROOT}/Y_Fu/checkpoints/${RUN_NAME}"
LOGDIR="${REPO_ROOT}/Y_Fu/logs/${RUN_NAME}"
MONITOR_LOG="${REPO_ROOT}/Y_Fu/monitor_logs/${RUN_NAME}.log"
PATCH_LOG="${REPO_ROOT}/Y_Fu/monitor_logs/${RUN_NAME}_patch.log"

mkdir -p "${SAVE_DIR}" "${LOGDIR}" "${REPO_ROOT}/Y_Fu/monitor_logs"

cp "${PPO_PY}" "${PPO_PY}.bak_${STAMP}"

"${PYTHON_BIN}" - <<'PY' "${PPO_PY}" "${PATCH_LOG}"
from pathlib import Path
import re
import sys

ppo_path = Path(sys.argv[1])
patch_log = Path(sys.argv[2])
text = ppo_path.read_text()

anchor = "academy_run_to_score_with_keeper"
start = text.find(anchor)
if start == -1:
    raise SystemExit(f"Could not find preset anchor: {anchor}")

candidates = []
for pat in [
    r'(?m)^[ \t]*["\']academy_[^"\']+["\']\s*[:=]',
    r'(?m)^[ \t]*["\']five_vs_five[^"\']*["\']\s*[:=]',
    r'(?m)^[A-Z_][A-Z0-9_]*\s*=',
]:
    for m in re.finditer(pat, text[start+1:]):
        candidates.append(start + 1 + m.start())
end = min(candidates) if candidates else len(text)
block = text[start:end]

changes = {
    "shot_attempt_reward": "0.60",
    "final_third_entry_reward": "0.10",
    "pass_success_reward": "0.00",
    "pass_failure_penalty": "0.00",
    "pass_progress_reward_scale": "0.00",
    "attacking_possession_reward": "0.00",
    "possession_retention_reward": "0.00",
    "possession_recovery_reward": "0.00",
    "defensive_third_recovery_reward": "0.00",
    "own_half_turnover_penalty": "0.10",
    "opponent_attacking_possession_penalty": "0.00",
}

new_block = block
log_lines = []
for key, value in changes.items():
    patterns = [
        rf'({re.escape(key)}\s*=\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
        rf'(["\']{re.escape(key)}["\']\s*:\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
    ]
    replaced = False
    for pat in patterns:
        newer, n = re.subn(pat, rf'\g<1>{value}', new_block, count=1)
        if n:
            new_block = newer
            replaced = True
            log_lines.append(f"UPDATED {key} -> {value}")
            break
    if not replaced:
        log_lines.append(f"SKIPPED {key}")

if new_block == block:
    raise SystemExit("No reward keys changed. Inspect ppo.py format.")

text = text[:start] + new_block + text[end:]
ppo_path.write_text(text)
patch_log.write_text("\n".join(log_lines) + "\n")
print("Patch applied.")
for line in log_lines:
    print(line)
PY

format_time() {
  local secs="${1}"
  if [[ "${secs}" -lt 0 ]]; then secs=0; fi
  printf "%02d:%02d:%02d" "$((secs/3600))" "$(((secs%3600)/60))" "$((secs%60))"
}

extract_step() {
  local logfile="${1}"
  if [[ ! -f "${logfile}" ]]; then echo 0; return; fi
  local step=""
  step="$(grep -Eo 'total_agent_steps[=: ]+[0-9]+' "${logfile}" 2>/dev/null | tail -n1 | grep -Eo '[0-9]+' | tail -n1)"
  if [[ -z "${step}" ]]; then
    step="$(grep -Eo 'agent_steps[=: ]+[0-9]+' "${logfile}" 2>/dev/null | tail -n1 | grep -Eo '[0-9]+' | tail -n1)"
  fi
  if [[ -z "${step}" ]]; then
    step="$(grep -Eo 'step[=: ]+[0-9]+' "${logfile}" 2>/dev/null | tail -n1 | grep -Eo '[0-9]+' | tail -n1)"
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
      remaining="$(awk "BEGIN { if (${current_step}>0) printf \"%d\", ((${TIMESTEPS}-${current_step})/(${current_step}/${elapsed})); else print -1 }")"
      if [[ "${current_step}" -ge "${TIMESTEPS}" ]]; then remaining=0; fi
      draw_bar "${current_step}" "${TIMESTEPS}"
      printf " | elapsed %s | left %s | step %s/%s | %s step/s\r" \
        "$(format_time "${elapsed}")" "$(format_time "${remaining}")" "${current_step}" "${TIMESTEPS}" "${speed}"
    else
      draw_bar 0 "${TIMESTEPS}"
      printf " | elapsed %s | left --:--:-- | step 0/%s | warming-up\r" "$(format_time "${elapsed}")" "${TIMESTEPS}"
    fi
    sleep 2
  done
  echo
}

CMD=(
  "${PYTHON_BIN}" -u "${TRAIN_PY}"
  --preset academy_run_to_score_with_keeper
  --use-player-id
  --num-envs 4
  --rollout-steps 192
  --total-timesteps "${TIMESTEPS}"
  --save-interval 5
  --update-epochs 4
  --num-minibatches 1
  --device cpu
  --seed "${SEED}"
  --save-dir "${SAVE_DIR}"
  --logdir "${LOGDIR}"
)

echo "Patch log : ${PATCH_LOG}"
echo "Run name  : ${RUN_NAME}"
echo "Save dir  : ${SAVE_DIR}"
echo "Log dir   : ${LOGDIR}"
echo "Monitor   : ${MONITOR_LOG}"
printf "Command   : "
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
echo "Patch log: ${PATCH_LOG}"
echo "Monitor log: ${MONITOR_LOG}"
echo "Checkpoint dir: ${SAVE_DIR}"

if [[ "${RC}" -ne 0 ]]; then
  tail -n 40 "${MONITOR_LOG}" || true
fi

exit "${RC}"
