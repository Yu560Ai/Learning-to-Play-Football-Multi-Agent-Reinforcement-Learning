from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tikick_root() -> Path:
    return _project_root() / "Two_V_Two" / "third_party" / "tikick"


def _bootstrap_paths() -> None:
    repo_root = _project_root()
    for path in (repo_root, _tikick_root()):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_paths()

from tmarl.configs.config import get_config

from Two_V_Two.training.basic_shared_ppo import SharedPolicyPPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = get_config()
    parser.description = "Basic shared-policy PPO trainer for the Two_V_Two baseline."

    parser.add_argument(
        "--scenario_name",
        type=str,
        default="two_v_two_plus_goalkeepers",
        help="GRF scenario name for the main Two_V_Two run.",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=2,
        help="Number of controlled left-side players.",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="simple115v2",
        help="GRF observation representation.",
    )
    parser.add_argument(
        "--rewards",
        type=str,
        default="scoring",
        help="Underlying GRF reward string. Phase 1 reward shaping uses local reward_variant logic.",
    )
    parser.add_argument(
        "--action_set",
        type=str,
        default="default",
        help="GRF action set to request from the environment.",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=20000,
        help="Total environment steps across rollout threads.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="Two_V_Two/runs/basic_shared_ppo",
        help="Directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render the first environment.",
    )
    parser.add_argument(
        "--save_replay",
        action="store_true",
        default=False,
        help="Save replay dumps for the first environment.",
    )
    parser.add_argument(
        "--game_engine_random_seed",
        type=int,
        default=None,
        help="Optional GRF engine seed base.",
    )
    parser.add_argument(
        "--reward_variant",
        type=str,
        default="r1_scoring",
        choices=["r1_scoring", "r2_progress", "r3_assist", "r4_anti_selfish"],
        help="Phase 1 reward variant to run on top of the shared-policy PPO baseline.",
    )
    parser.add_argument(
        "--structure_variant",
        type=str,
        default="shared_ppo",
        choices=["shared_ppo", "shared_ppo_id", "mappo_id_cc"],
        help="Phase 2 structure variant: baseline shared PPO, shared PPO with identity, or MAPPO-style centralized critic.",
    )
    parser.add_argument(
        "--progress_reward_coef",
        type=float,
        default=0.05,
        help="Coefficient alpha for positive forward ball progress reward.",
    )
    parser.add_argument(
        "--assist_reward",
        type=float,
        default=0.5,
        help="Assist reward beta when a controlled pass leads to a teammate goal within the assist window.",
    )
    parser.add_argument(
        "--assist_window",
        type=int,
        default=25,
        help="Assist window K in environment steps.",
    )
    parser.add_argument(
        "--selfish_possession_threshold",
        type=int,
        default=12,
        help="Possession streak threshold N before the anti-selfish penalty applies.",
    )
    parser.add_argument(
        "--selfish_penalty",
        type=float,
        default=0.02,
        help="Penalty gamma applied each step after the selfish possession threshold is exceeded.",
    )
    parser.add_argument(
        "--pass_to_shot_window",
        type=int,
        default=10,
        help="Diagnostic window in steps for counting a teammate shot shortly after a successful controlled pass.",
    )

    parser.set_defaults(
        env_name="football",
        episode_length=400,
        n_rollout_threads=1,
        use_recurrent_policy=False,
        use_naive_recurrent_policy=False,
        use_valuenorm=False,
        use_popart=False,
        use_policy_vhead=True,
        use_eval=False,
        num_mini_batch=4,
        ppo_epoch=4,
        lr=3e-4,
        critic_lr=3e-4,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        clip_param=0.2,
        max_grad_norm=0.5,
        save_interval=10,
        log_interval=1,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trainer = SharedPolicyPPOTrainer(args)
    checkpoint = trainer.train()
    print(f"[done] latest_checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
