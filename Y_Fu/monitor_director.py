from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
Y_FU_ROOT = PROJECT_ROOT / "Y_Fu"
DEFAULT_PYTHON = PROJECT_ROOT / "football-master" / "football-env" / "bin" / "python3"
TRAINING_STAGE_LOG = Y_FU_ROOT / "TRAINING_STAGE_LOG.md"


@dataclass(frozen=True)
class EvalResult:
    checkpoint: Path
    avg_return: float
    avg_score_reward: float
    avg_goal_diff: float
    win_rate: float
    avg_length: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor markdown progress and continue the Y_Fu training pipeline.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    parser.add_argument("--log-dir", default="Y_Fu/monitor_logs")
    parser.add_argument("--state-path", default="Y_Fu/monitor_state.json")
    parser.add_argument("--status-md-path", default="Y_Fu/MONITOR_STATUS.md")
    parser.add_argument("--plan", choices=("pilot", "full"), default="full")
    return parser.parse_args()


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_status(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _parse_current_status_rows(training_stage_log: Path) -> list[dict[str, str]]:
    lines = training_stage_log.read_text(encoding="utf-8").splitlines()
    header_index = None
    for index, line in enumerate(lines):
        if line.strip().startswith("| Stage | Preset | Run Status | Pass Status | Evidence Window | Preferred Checkpoint |"):
            header_index = index
            break
    if header_index is None:
        return []

    rows: list[dict[str, str]] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) != 6:
            continue
        rows.append(
            {
                "stage": columns[0],
                "preset": columns[1].strip("`"),
                "run_status": columns[2],
                "pass_status": columns[3],
                "evidence_window": columns[4],
                "preferred_checkpoint": columns[5].strip("`"),
            }
        )
    return rows


def _find_active_jobs() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid,cmd"],
        capture_output=True,
        text=True,
        check=True,
    )
    active: list[str] = []
    patterns = (
        "Y_Fu/train.py",
        "Y_Fu/collect_offline_data.py",
        "Y_Fu/train_iql.py",
        "Y_Fu/evaluate.py",
        "Y_Fu/evaluate_iql.py",
    )
    current_pid = str(Path("/proc/self").resolve().name)
    for line in result.stdout.splitlines():
        if "Y_Fu/monitor_director.py" in line:
            continue
        if any(pattern in line for pattern in patterns):
            if f" {current_pid} " in f" {line} ":
                continue
            active.append(line.strip())
    return active


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    if path.name == "latest.pt":
        return (10**9, path.name)
    match = re.search(r"update_(\d+)\.pt$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (-1, path.name)


def _five_vs_five_checkpoint_candidates() -> list[Path]:
    checkpoint_dir = Y_FU_ROOT / "checkpoints" / "five_vs_five"
    candidates = [path for path in checkpoint_dir.glob("*.pt") if path.is_file()]
    candidates.sort(key=_checkpoint_sort_key, reverse=True)
    preferred: list[Path] = []
    for path in candidates:
        if path.name == "latest.pt":
            preferred.append(path)
            break
    preferred.extend([path for path in candidates if path.name != "latest.pt"][:3])
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in preferred:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def _parse_eval_metrics(output: str) -> EvalResult | None:
    match = re.search(
        r"trained_policy:\s+avg_return=([-+0-9.]+)\s+avg_score_reward=([-+0-9.]+)\s+avg_goal_diff=([-+0-9.]+)\s+win_rate=([-+0-9.]+)\s+avg_length=([-+0-9.]+)",
        output,
    )
    if match is None:
        return None
    return EvalResult(
        checkpoint=Path(),
        avg_return=float(match.group(1)),
        avg_score_reward=float(match.group(2)),
        avg_goal_diff=float(match.group(3)),
        win_rate=float(match.group(4)),
        avg_length=float(match.group(5)),
    )


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    execute: bool,
) -> subprocess.CompletedProcess[str] | None:
    _append_log(log_path, f"$ {' '.join(command)}")
    if not execute:
        return None
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        _append_log(log_path, result.stdout)
    if result.stderr:
        _append_log(log_path, result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )
    return result


def _evaluate_candidates(
    *,
    python_bin: str,
    candidates: list[Path],
    log_path: Path,
    execute: bool,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for checkpoint in candidates:
        command = [
            python_bin,
            "Y_Fu/evaluate.py",
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            "20",
            "--deterministic",
            "--device",
            "cpu",
            "--seed",
            "123",
        ]
        completed = _run_command(command, cwd=PROJECT_ROOT, log_path=log_path, execute=execute)
        if completed is None:
            continue
        metrics = _parse_eval_metrics(completed.stdout)
        if metrics is None:
            raise RuntimeError(f"Could not parse evaluation metrics for checkpoint {checkpoint}")
        results.append(
            EvalResult(
                checkpoint=checkpoint,
                avg_return=metrics.avg_return,
                avg_score_reward=metrics.avg_score_reward,
                avg_goal_diff=metrics.avg_goal_diff,
                win_rate=metrics.win_rate,
                avg_length=metrics.avg_length,
            )
        )
    results.sort(
        key=lambda item: (item.win_rate, item.avg_goal_diff, item.avg_score_reward),
        reverse=True,
    )
    return results


def _all_manifest_paths_exist(paths: list[Path]) -> bool:
    return all((path / "manifest.json").exists() for path in paths)


def _summarize_eval_results(eval_results: list[EvalResult]) -> list[dict[str, Any]]:
    return [
        {
            "checkpoint": str(item.checkpoint),
            "win_rate": item.win_rate,
            "avg_goal_diff": item.avg_goal_diff,
            "avg_score_reward": item.avg_score_reward,
            "avg_return": item.avg_return,
            "avg_length": item.avg_length,
        }
        for item in eval_results
    ]


def _status_lines(*, state: dict[str, Any], rows: list[dict[str, str]], active_jobs: list[str]) -> list[str]:
    lines = [
        "# Monitor Status",
        "",
        "## Current Summary",
        "",
        f"- monitor_phase: `{state.get('phase', 'unknown')}`",
        f"- last_action: `{state.get('last_action', 'none')}`",
        f"- last_decision: `{state.get('last_decision', 'none')}`",
        f"- active_job_count: `{len(active_jobs)}`",
        "",
        "## Current Status Table",
        "",
    ]
    if rows:
        lines.append("| Stage | Preset | Run Status | Pass Status | Preferred Checkpoint |")
        lines.append("|---|---|---|---|---|")
        for row in rows:
            lines.append(
                f"| {row['stage']} | `{row['preset']}` | {row['run_status']} | {row['pass_status']} | {row['preferred_checkpoint']} |"
            )
    else:
        lines.append("- could not parse `TRAINING_STAGE_LOG.md` status table")

    lines.extend(["", "## Active Jobs", ""])
    if active_jobs:
        lines.extend([f"- `{job}`" for job in active_jobs])
    else:
        lines.append("- none")

    if "ppo_eval_results" in state:
        lines.extend(["", "## PPO Eval Results", ""])
        for item in state["ppo_eval_results"]:
            lines.append(
                f"- `{item['checkpoint']}`: win_rate={item['win_rate']:.3f} avg_goal_diff={item['avg_goal_diff']:.3f} avg_score_reward={item['avg_score_reward']:.3f}"
            )
    return lines


def main() -> None:
    args = parse_args()
    log_dir = (PROJECT_ROOT / args.log_dir).resolve()
    log_path = log_dir / "monitor.log"
    state_path = (PROJECT_ROOT / args.state_path).resolve()
    status_md_path = (PROJECT_ROOT / args.status_md_path).resolve()

    pilot_dirs = [
        Y_FU_ROOT / "offline_data" / "pilot_5v5_best_eps0",
        Y_FU_ROOT / "offline_data" / "pilot_5v5_best_eps15",
        Y_FU_ROOT / "offline_data" / "pilot_5v5_weaker_eps005",
        Y_FU_ROOT / "offline_data" / "pilot_5v5_random",
    ]
    full_dirs = [
        Y_FU_ROOT / "offline_data" / "run_5v5_best_eps0",
        Y_FU_ROOT / "offline_data" / "run_5v5_best_eps15",
        Y_FU_ROOT / "offline_data" / "run_5v5_weaker_eps005",
        Y_FU_ROOT / "offline_data" / "run_5v5_random",
    ]

    while True:
        state = _load_state(state_path)
        rows = _parse_current_status_rows(TRAINING_STAGE_LOG)
        active_jobs = _find_active_jobs()
        state.setdefault("phase", "waiting_for_md")

        any_in_progress = any("In progress" in row["run_status"] for row in rows)
        if any_in_progress:
            state["phase"] = "waiting_for_stage_completion"
            state["last_decision"] = "md still shows an in-progress stage; monitor will wait"
        elif any("collect_offline_data.py" in job or "train_iql.py" in job or "Y_Fu/train.py" in job for job in active_jobs):
            state["phase"] = "waiting_for_active_job"
            state["last_decision"] = "a heavy job is currently active; monitor will not schedule another one"
        else:
            best_checkpoint = Path(state["best_checkpoint"]) if "best_checkpoint" in state else None
            weaker_checkpoint = Path(state["weaker_checkpoint"]) if "weaker_checkpoint" in state else None

            if best_checkpoint is None or weaker_checkpoint is None:
                candidates = _five_vs_five_checkpoint_candidates()
                if not candidates:
                    state["phase"] = "waiting_for_checkpoints"
                    state["last_decision"] = "no five_vs_five checkpoints found yet"
                else:
                    state["phase"] = "evaluating_ppo_candidates"
                    state["last_action"] = "evaluate_ppo_candidates"
                    eval_results = _evaluate_candidates(
                        python_bin=args.python_bin,
                        candidates=candidates,
                        log_path=log_path,
                        execute=args.execute,
                    )
                    if eval_results:
                        state["ppo_eval_results"] = _summarize_eval_results(eval_results)
                        state["best_checkpoint"] = str(eval_results[0].checkpoint)
                        state["weaker_checkpoint"] = str(eval_results[1].checkpoint if len(eval_results) > 1 else eval_results[0].checkpoint)
                        state["last_decision"] = (
                            f"selected best checkpoint {state['best_checkpoint']} and weaker checkpoint {state['weaker_checkpoint']}"
                        )
                        best_checkpoint = Path(state["best_checkpoint"])
                        weaker_checkpoint = Path(state["weaker_checkpoint"])
                    else:
                        state["last_decision"] = "evaluation was skipped because execute mode is disabled"

            elif not _all_manifest_paths_exist(pilot_dirs):
                state["phase"] = "collecting_offline_pilot"
                state["last_action"] = "offline_pilot_collection"
                pilot_commands = [
                    [
                        args.python_bin, "Y_Fu/collect_offline_data.py",
                        "--checkpoint", str(best_checkpoint),
                        "--policy", "checkpoint",
                        "--num-envs", "6",
                        "--total-env-steps", "200000",
                        "--epsilon", "0.0",
                        "--chunk-size", "500000",
                        "--save-dir", "Y_Fu/offline_data/pilot_5v5_best_eps0",
                        "--seed", "123",
                        "--obs-dtype", "float16",
                        "--checkpoint-id", "1",
                    ],
                    [
                        args.python_bin, "Y_Fu/collect_offline_data.py",
                        "--checkpoint", str(best_checkpoint),
                        "--policy", "checkpoint",
                        "--num-envs", "6",
                        "--total-env-steps", "100000",
                        "--epsilon", "0.15",
                        "--chunk-size", "500000",
                        "--save-dir", "Y_Fu/offline_data/pilot_5v5_best_eps15",
                        "--seed", "124",
                        "--obs-dtype", "float16",
                        "--checkpoint-id", "1",
                    ],
                    [
                        args.python_bin, "Y_Fu/collect_offline_data.py",
                        "--checkpoint", str(weaker_checkpoint),
                        "--policy", "checkpoint",
                        "--num-envs", "6",
                        "--total-env-steps", "100000",
                        "--epsilon", "0.05",
                        "--chunk-size", "500000",
                        "--save-dir", "Y_Fu/offline_data/pilot_5v5_weaker_eps005",
                        "--seed", "125",
                        "--obs-dtype", "float16",
                        "--checkpoint-id", "2",
                    ],
                    [
                        args.python_bin, "Y_Fu/collect_offline_data.py",
                        "--policy", "random",
                        "--num-envs", "6",
                        "--total-env-steps", "100000",
                        "--epsilon", "0.0",
                        "--chunk-size", "500000",
                        "--save-dir", "Y_Fu/offline_data/pilot_5v5_random",
                        "--seed", "126",
                        "--obs-dtype", "float16",
                        "--checkpoint-id", "3",
                    ],
                ]
                for command, manifest_dir in zip(pilot_commands, pilot_dirs, strict=True):
                    if (manifest_dir / "manifest.json").exists():
                        continue
                    _run_command(command, cwd=PROJECT_ROOT, log_path=log_path, execute=args.execute)
                state["last_decision"] = "pilot collection commands completed or were queued"

            elif not (Y_FU_ROOT / "checkpoints" / "iql_5v5_pilot" / "best.pt").exists():
                state["phase"] = "training_iql_pilot"
                state["last_action"] = "pilot_iql_training"
                command = [
                    args.python_bin, "Y_Fu/train_iql.py",
                    "--dataset-dirs",
                    "Y_Fu/offline_data/pilot_5v5_best_eps0",
                    "Y_Fu/offline_data/pilot_5v5_best_eps15",
                    "Y_Fu/offline_data/pilot_5v5_weaker_eps005",
                    "Y_Fu/offline_data/pilot_5v5_random",
                    "--reward-key", "reward",
                    "--device", "cuda",
                    "--save-dir", "Y_Fu/checkpoints/iql_5v5_pilot",
                    "--batch-size", "4096",
                    "--chunk-reuse-batches", "32",
                    "--learning-rate", "3e-4",
                    "--gamma", "0.993",
                    "--expectile", "0.7",
                    "--temperature", "3.0",
                    "--tau", "0.005",
                    "--normalize-rewards",
                    "--total-gradient-steps", "20000",
                    "--eval-interval", "5000",
                    "--save-interval", "10000",
                    "--eval-episodes", "20",
                    "--seed", "123",
                ]
                _run_command(command, cwd=PROJECT_ROOT, log_path=log_path, execute=args.execute)
                state["last_decision"] = "pilot IQL training completed or was queued"

            elif args.plan == "full" and not _all_manifest_paths_exist(full_dirs):
                state["phase"] = "collecting_offline_full"
                state["last_action"] = "full_offline_collection"
                full_commands = [
                    [
                        args.python_bin, "Y_Fu/collect_offline_data.py",
                        "--checkpoint", str(best_checkpoint),
                        "--policy", "checkpoint",
                        "--num-envs", "6",
                        "--total-env-steps", "20000000",
                        "--epsilon", "0.0",
                        "--chunk-size", "500000",
                        "--save-dir", "Y_Fu/offline_data/run_5v5_best_eps0",
                        "--seed", "223",
                        "--obs-dtype", "float16",
                        "--checkpoint-id", "1",
                    ],
                    [
                        args.python_bin, "Y_Fu/collect_offline_data.py",
                        "--checkpoint", str(best_checkpoint),
                        "--policy", "checkpoint",
                        "--num-envs", "6",
                        "--total-env-steps", "12500000",
                        "--epsilon", "0.15",
                        "--chunk-size", "500000",
                        "--save-dir", "Y_Fu/offline_data/run_5v5_best_eps15",
                        "--seed", "224",
                        "--obs-dtype", "float16",
                        "--checkpoint-id", "1",
                    ],
                    [
                        args.python_bin, "Y_Fu/collect_offline_data.py",
                        "--checkpoint", str(weaker_checkpoint),
                        "--policy", "checkpoint",
                        "--num-envs", "6",
                        "--total-env-steps", "10000000",
                        "--epsilon", "0.05",
                        "--chunk-size", "500000",
                        "--save-dir", "Y_Fu/offline_data/run_5v5_weaker_eps005",
                        "--seed", "225",
                        "--obs-dtype", "float16",
                        "--checkpoint-id", "2",
                    ],
                    [
                        args.python_bin, "Y_Fu/collect_offline_data.py",
                        "--policy", "random",
                        "--num-envs", "6",
                        "--total-env-steps", "7500000",
                        "--epsilon", "0.0",
                        "--chunk-size", "500000",
                        "--save-dir", "Y_Fu/offline_data/run_5v5_random",
                        "--seed", "226",
                        "--obs-dtype", "float16",
                        "--checkpoint-id", "3",
                    ],
                ]
                for command, manifest_dir in zip(full_commands, full_dirs, strict=True):
                    if (manifest_dir / "manifest.json").exists():
                        continue
                    _run_command(command, cwd=PROJECT_ROOT, log_path=log_path, execute=args.execute)
                state["last_decision"] = "full offline collection commands completed or were queued"

            elif args.plan == "full" and not (Y_FU_ROOT / "checkpoints" / "iql_5v5_iter0" / "best.pt").exists():
                state["phase"] = "training_iql_full"
                state["last_action"] = "full_iql_training"
                command = [
                    args.python_bin, "Y_Fu/train_iql.py",
                    "--dataset-dirs",
                    "Y_Fu/offline_data/run_5v5_best_eps0",
                    "Y_Fu/offline_data/run_5v5_best_eps15",
                    "Y_Fu/offline_data/run_5v5_weaker_eps005",
                    "Y_Fu/offline_data/run_5v5_random",
                    "--reward-key", "reward",
                    "--device", "cuda",
                    "--save-dir", "Y_Fu/checkpoints/iql_5v5_iter0",
                    "--batch-size", "4096",
                    "--chunk-reuse-batches", "32",
                    "--learning-rate", "3e-4",
                    "--gamma", "0.993",
                    "--expectile", "0.7",
                    "--temperature", "3.0",
                    "--tau", "0.005",
                    "--normalize-rewards",
                    "--total-gradient-steps", "1000000",
                    "--eval-interval", "10000",
                    "--save-interval", "50000",
                    "--eval-episodes", "20",
                    "--seed", "323",
                ]
                _run_command(command, cwd=PROJECT_ROOT, log_path=log_path, execute=args.execute)
                state["last_decision"] = "full IQL training completed or was queued"

            else:
                state["phase"] = "idle_done"
                state["last_decision"] = "no pending automated phase remains under the current monitor plan"

        _save_state(state_path, state)
        _write_status(status_md_path, _status_lines(state=state, rows=rows, active_jobs=active_jobs))

        if args.once:
            return
        time.sleep(max(10, int(args.poll_seconds)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
