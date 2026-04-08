#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv_yfu_grf_sys/bin/python"

CKPT_DIR="${REPO_ROOT}/Y_Fu/checkpoints/academy_pass_reboot_v2"
EVAL_DIR="${REPO_ROOT}/Y_Fu/eval_runs/auto_eval_$(date +%H%M%S)"
VIDEO_DIR="${REPO_ROOT}/Y_Fu/videos/auto_eval_$(date +%H%M%S)"

mkdir -p "$EVAL_DIR" "$VIDEO_DIR"

echo "=== WAITING FOR TRAINING TO FINISH ==="
while pgrep -f "train.py" > /dev/null; do
    sleep 10
done

echo "=== TRAINING DONE ==="

CHECKPOINTS=(
  update_10.pt
  update_50.pt
  update_100.pt
  update_200.pt
  latest.pt
)

BEST_SCORE=0
BEST_CKPT=""

for ckpt in "${CHECKPOINTS[@]}"; do
    PATH_CKPT="${CKPT_DIR}/${ckpt}"
    if [[ ! -f "$PATH_CKPT" ]]; then
        continue
    fi

    LOG="${EVAL_DIR}/${ckpt}.log"

    echo "Evaluating $ckpt..."

    $PYTHON Y_Fu/evaluate.py \
        --checkpoint "$PATH_CKPT" \
        --episodes 20 \
        --device cpu > "$LOG"

    SCORE=$(grep "avg_score_reward" "$LOG" | awk -F= '{print $3}' | awk '{print $1}')

    echo "Score reward: $SCORE"

    if (( $(echo "$SCORE > $BEST_SCORE" | bc -l) )); then
        BEST_SCORE=$SCORE
        BEST_CKPT=$ckpt
    fi
done

echo "=== BEST CHECKPOINT ==="
echo "$BEST_CKPT with score $BEST_SCORE"

if [[ "$BEST_CKPT" == "" ]]; then
    echo "No valid checkpoint found"
    exit 0
fi

echo "=== GENERATING VIDEO ==="

$PYTHON Y_Fu/evaluate.py \
    --checkpoint "${CKPT_DIR}/${BEST_CKPT}" \
    --episodes 1 \
    --device cpu \
    --save-video \
    --video-dir "$VIDEO_DIR"

echo "Video saved to $VIDEO_DIR"

echo "=== DECISION ==="

if (( $(echo "$BEST_SCORE > 0.3" | bc -l) )); then
    echo "GOOD: shooting emerging → go 5v5 next"
elif (( $(echo "$BEST_SCORE > 0.05" | bc -l) )); then
    echo "WEAK: borderline → maybe another Academy tweak"
else
    echo "FAIL: still no shooting → reward still wrong"
fi