from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import torch


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tikick_root() -> Path:
    return _project_root() / "Two_V_Two" / "third_party" / "tikick"


def _bootstrap_paths() -> None:
    for path in (_project_root(), _tikick_root()):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_paths()

from Two_V_Two.training.basic_shared_ppo import MAPPOModule
from Two_V_Two.env.grf_simple_env import TwoVTwoFootballEnv


VARIANTS = [
    "r1_scoring",
    "r2_progress",
    "r3_assist",
    "r4_anti_selfish",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved Phase 1 shared-PPO checkpoints with deterministic actions.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=VARIANTS,
        help="Reward variants to evaluate.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="Two_V_Two/results/phase1",
        help="Root directory containing per-variant run directories.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="latest.pt",
        help="Checkpoint filename inside each variant checkpoint directory.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of deterministic evaluation episodes per variant.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="Two_V_Two/results/phase1/eval_summary.json",
        help="Path to write the evaluation summary JSON.",
    )
    return parser


def load_checkpoint_args(run_dir: Path, checkpoint_name: str) -> tuple[Namespace, dict]:
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    payload = torch.load(checkpoint_path, map_location="cpu")
    args = Namespace(**payload["args"])
    args.disable_cuda = True
    args.render = False
    args.save_replay = False
    args.n_rollout_threads = 1
    return args, payload


def evaluate_variant(run_dir: Path, checkpoint_name: str, episodes: int) -> dict[str, float | int | str]:
    args, payload = load_checkpoint_args(run_dir, checkpoint_name)
    env = TwoVTwoFootballEnv(args, rank=0, log_dir=str(run_dir / "eval_replays"), is_eval=True)
    obs_space = env.observation_space[0]
    share_obs_space = env.share_observation_space[0]
    act_space = env.action_space[0]

    module = MAPPOModule(args, obs_space, share_obs_space, act_space, device=torch.device("cpu"))
    module.actor.load_state_dict(payload["actor_state_dict"])
    module.actor.eval()

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_goals: list[float] = []
    episode_passes: list[float] = []
    episode_assists: list[float] = []
    episode_pass_to_shot: list[float] = []
    episode_possession_means: list[float] = []

    try:
        for _ in range(episodes):
            obs, _share_obs, available_actions = env.reset()
            rnn_states = np.zeros((args.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
            masks = np.ones((args.num_agents, 1), dtype=np.float32)
            done = False
            total_return = 0.0
            total_length = 0
            final_episode_metrics: dict[str, float] = {}

            while not done:
                with torch.no_grad():
                    actions, _action_log_probs, next_rnn_states = module.actor(
                        obs.reshape(-1, obs.shape[-1]),
                        rnn_states.reshape(-1, *rnn_states.shape[1:]),
                        masks.reshape(-1, 1),
                        available_actions.reshape(-1, available_actions.shape[-1]),
                        deterministic=True,
                    )

                env_actions = actions.detach().cpu().numpy().reshape(args.num_agents, -1).squeeze(-1).astype(np.int64)
                next_obs, _next_share_obs, rewards, dones, infos, next_available_actions = env.step(env_actions)
                total_return += float(np.mean(rewards))
                total_length += 1
                done = bool(np.all(dones))

                if done:
                    final_episode_metrics = dict(infos[0].get("episode_metrics", {}))
                    masks = np.zeros((args.num_agents, 1), dtype=np.float32)
                else:
                    masks = np.ones((args.num_agents, 1), dtype=np.float32)

                rnn_states = next_rnn_states.detach().cpu().numpy().reshape(args.num_agents, args.recurrent_N, args.hidden_size)
                rnn_states[masks.squeeze(-1) == 0.0] = 0.0
                obs = next_obs
                available_actions = next_available_actions

            episode_returns.append(total_return)
            episode_lengths.append(total_length)
            episode_goals.append(float(final_episode_metrics.get("goal_count", 0.0)))
            episode_passes.append(float(final_episode_metrics.get("pass_count", 0.0)))
            episode_assists.append(float(final_episode_metrics.get("assist_count", 0.0)))
            episode_pass_to_shot.append(float(final_episode_metrics.get("pass_to_shot_count", 0.0)))
            episode_possession_means.append(float(final_episode_metrics.get("mean_same_owner_possession_length", 0.0)))
    finally:
        env.close()

    return {
        "reward_variant": str(args.reward_variant),
        "structure_variant": str(getattr(args, "structure_variant", "shared_ppo")),
        "episodes": int(episodes),
        "checkpoint": checkpoint_name,
        "checkpoint_update": int(payload["update"]),
        "checkpoint_env_steps": int(payload["total_env_steps"]),
        "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else float("nan"),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
        "goal_rate": float(np.mean(np.asarray(episode_goals) > 0.0)) if episode_goals else float("nan"),
        "mean_goal_count": float(np.mean(episode_goals)) if episode_goals else float("nan"),
        "mean_pass_count": float(np.mean(episode_passes)) if episode_passes else float("nan"),
        "mean_assist_count": float(np.mean(episode_assists)) if episode_assists else float("nan"),
        "mean_pass_to_shot_count": float(np.mean(episode_pass_to_shot)) if episode_pass_to_shot else float("nan"),
        "mean_same_owner_possession_length": float(np.mean(episode_possession_means))
        if episode_possession_means
        else float("nan"),
    }


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root)
    summaries: dict[str, dict[str, float | int | str]] = {}

    for variant in args.variants:
        run_dir = results_root / variant
        summary = evaluate_variant(run_dir, args.checkpoint_name, args.episodes)
        summaries[variant] = summary
        print(
            "[eval] "
            f"variant={variant} "
            f"episodes={summary['episodes']} "
            f"goal_rate={summary['goal_rate']:.3f} "
            f"return={summary['mean_episode_return']:.3f} "
            f"goals={summary['mean_goal_count']:.3f} "
            f"passes={summary['mean_pass_count']:.3f} "
            f"assists={summary['mean_assist_count']:.3f} "
            f"pass_to_shot={summary['mean_pass_to_shot_count']:.3f} "
            f"possession={summary['mean_same_owner_possession_length']:.2f}",
            flush=True,
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[eval] summary_json={output_path}", flush=True)


if __name__ == "__main__":
    main()
