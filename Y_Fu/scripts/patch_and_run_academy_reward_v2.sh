#!/usr/bin/env bash
set -euo pipefail

# This script does two things:
# 1) patches Y_Fu/yfu_football/ppo.py for a more "completion-first" Academy reward
# 2) launches the next Academy run
#
# Assumptions:
# - repo root contains .venv_yfu_grf_sys and Y_Fu/
# - reward preset block contains the string academy_pass_and_shoot_with_keeper
# - inside that block, keys like shot_attempt_reward / pass_success_reward / ... exist
#
# It is intentionally conservative:
# - backs up ppo.py first
# - only edits inside the academy_pass_and_shoot_with_keeper preset block
# - only updates keys that actually exist

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YFU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${YFU_DIR}/.." && pwd)"

PYTHON_BIN="${REPO_ROOT}/.venv_yfu_grf_sys/bin/python"
TRAIN_PY="${REPO_ROOT}/Y_Fu/train.py"
PPO_PY="${REPO_ROOT}/Y_Fu/yfu_football/ppo.py"
MONITOR_DIR="${REPO_ROOT}/Y_Fu/monitor_logs"
SAVE_DIR="${REPO_ROOT}/Y_Fu/checkpoints/academy_pass_reboot_v2"
LOGDIR="${REPO_ROOT}/Y_Fu/logs/academy_pass_reboot_v2"

mkdir -p "${MONITOR_DIR}" "${SAVE_DIR}" "${LOGDIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python not found at ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -f "${PPO_PY}" ]]; then
  echo "ERROR: ppo.py not found at ${PPO_PY}"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_PPO="${PPO_PY}.bak_${STAMP}"
PATCH_LOG="${MONITOR_DIR}/academy_reward_patch_${STAMP}.log"
RUN_LOG="${MONITOR_DIR}/academy_pass_reboot_v2_${STAMP}.log"

cp "${PPO_PY}" "${BACKUP_PPO}"

echo "Backup created:"
echo "  ${BACKUP_PPO}"
echo

# Patch ppo.py using embedded Python so the edit is easier to control.
"${PYTHON_BIN}" - <<'PY' "${PPO_PY}" "${PATCH_LOG}"
from pathlib import Path
import re
import sys

ppo_path = Path(sys.argv[1])
patch_log = Path(sys.argv[2])

text = ppo_path.read_text()
original_text = text

# Target preset block:
# from the occurrence of academy_pass_and_shoot_with_keeper
# up to the next preset-like marker or a double newline followed by a top-level identifier.
anchor = "academy_pass_and_shoot_with_keeper"
start = text.find(anchor)
if start == -1:
    raise SystemExit(f"Could not find preset anchor: {anchor}")

# Heuristic end: next occurrence of another preset name line beginning with quotes/backticks
# or next major top-level assignment containing academy_/five_vs_five after current block.
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
    # Strongly prefer completion over empty progression
    "shot_attempt_reward": "0.30",
    "final_third_entry_reward": "0.05",
    "pass_success_reward": "0.03",
    "pass_failure_penalty": "0.03",
    "pass_progress_reward_scale": "0.00",
    "attacking_possession_reward": "0.00",
    "possession_retention_reward": "0.00",
    "possession_recovery_reward": "0.01",
    "defensive_third_recovery_reward": "0.01",
    "own_half_turnover_penalty": "0.08",
    "opponent_attacking_possession_penalty": "0.00",
}

log_lines = []
new_block = block

for key, value in changes.items():
    # Match both:
    # key = 0.1
    # "key": 0.1
    # 'key': 0.1
    patterns = [
        rf'({re.escape(key)}\s*=\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
        rf'(["\']{re.escape(key)}["\']\s*:\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
    ]
    replaced = False
    for pattern in patterns:
        newer, n = re.subn(pattern, rf'\g<1>{value}', new_block, count=1)
        if n:
            new_block = newer
            replaced = True
            log_lines.append(f"UPDATED {key} -> {value}")
            break
    if not replaced:
        log_lines.append(f"SKIPPED {key} (not found in preset block)")

# Safety check: ensure anchor block still exists and something changed
if new_block == block:
    raise SystemExit(
        "No reward keys were modified inside academy_pass_and_shoot_with_keeper block. "
        "Please inspect ppo.py format."
    )

text = text[:start] + new_block + text[end:]
ppo_path.write_text(text)
patch_log.write_text("\n".join(log_lines) + "\n")
print("Patch applied.")
print("Patch log:")
print(patch_log)
print("\nSummary:")
for line in log_lines:
    print(" ", line)
PY

echo
echo "Last patch log lines:"
tail -n 20 "${PATCH_LOG}" || true
echo

# Run next Academy attempt from scratch (new folder) after patching.
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
  --save-dir "${SAVE_DIR}"
  --logdir "${LOGDIR}"
)

TOTAL_STEPS=400000

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
    current_step="$(extract_step "${RUN_LOG}")"
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

echo "Patch log : ${PATCH_LOG}"
echo "Run log   : ${RUN_LOG}"
printf "Command   : "
printf "%q " "${CMD[@]}"
echo
echo

(
  cd "${REPO_ROOT}" || exit 1
  "${CMD[@]}" > "${RUN_LOG}" 2>&1
) &
TRAIN_PID=$!

monitor_loop
wait "${TRAIN_PID}"
RC=$?

echo
echo "Training finished with exit code: ${RC}"
echo "Run log saved to: ${RUN_LOG}"
echo "Patched ppo backup: ${BACKUP_PPO}"

if [[ "${RC}" -ne 0 ]]; then
  echo
  echo "Last 40 log lines:"
  tail -n 40 "${RUN_LOG}" 2>/dev/null || true
fi

exit "${RC}"
