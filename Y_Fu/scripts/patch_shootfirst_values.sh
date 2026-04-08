#!/usr/bin/env bash
set -e

FILE="Y_Fu/yfu_football/ppo.py"

echo "Backing up..."
cp $FILE ${FILE}.bak_$(date +%s)

echo "Patching shoot-first reward..."

# Replace ONLY inside this preset region
sed -i '/academy_run_to_score_with_keeper/,/}/ s/"shot_attempt_reward": [0-9.]\+/"shot_attempt_reward": 0.60/' $FILE
sed -i '/academy_run_to_score_with_keeper/,/}/ s/"final_third_entry_reward": [0-9.]\+/"final_third_entry_reward": 0.10/' $FILE
sed -i '/academy_run_to_score_with_keeper/,/}/ s/"pass_progress_reward_scale": [0-9.]\+/"pass_progress_reward_scale": 0.00/' $FILE
sed -i '/academy_run_to_score_with_keeper/,/}/ s/"pass_success_reward": [0-9.]\+/"pass_success_reward": 0.00/' $FILE
sed -i '/academy_run_to_score_with_keeper/,/}/ s/"pass_failure_penalty": [0-9.]\+/"pass_failure_penalty": 0.00/' $FILE
sed -i '/academy_run_to_score_with_keeper/,/}/ s/"attacking_possession_reward": [0-9.]\+/"attacking_possession_reward": 0.00/' $FILE
sed -i '/academy_run_to_score_with_keeper/,/}/ s/"possession_retention_reward": [0-9.]\+/"possession_retention_reward": 0.00/' $FILE
sed -i '/academy_run_to_score_with_keeper/,/}/ s/"own_half_turnover_penalty": [0-9.]\+/"own_half_turnover_penalty": 0.10/' $FILE

echo "Patch done."