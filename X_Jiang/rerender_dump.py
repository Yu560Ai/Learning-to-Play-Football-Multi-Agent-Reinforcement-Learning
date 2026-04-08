from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import types
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_gfootball_modules(repo_root: Path):
    gfootball_root = repo_root / "football-master" / "gfootball"

    # Minimal fake engine enum surface needed by football_action_set and labels.
    engine = types.ModuleType("gfootball_engine")

    class _BackendAction:
        idle = 0
        builtin_ai = 1
        left = 2
        top_left = 3
        top = 4
        top_right = 5
        right = 6
        bottom_right = 7
        bottom = 8
        bottom_left = 9
        long_pass = 10
        high_pass = 11
        short_pass = 12
        shot = 13
        keeper_rush = 14
        sliding = 15
        pressure = 16
        team_pressure = 17
        switch = 18
        sprint = 19
        dribble = 20
        release_direction = 21
        release_long_pass = 22
        release_high_pass = 23
        release_short_pass = 24
        release_shot = 25
        release_keeper_rush = 26
        release_sliding = 27
        release_pressure = 28
        release_team_pressure = 29
        release_switch = 30
        release_sprint = 31
        release_dribble = 32

    class _PlayerRole:
        e_PlayerRole_GK = 0

    engine.e_BackendAction = _BackendAction
    engine.e_PlayerRole = _PlayerRole
    sys.modules["gfootball_engine"] = engine

    gfootball_pkg = types.ModuleType("gfootball")
    gfootball_pkg.__path__ = [str(gfootball_root)]
    sys.modules["gfootball"] = gfootball_pkg

    env_pkg = types.ModuleType("gfootball.env")
    env_pkg.__path__ = [str(gfootball_root / "env")]
    sys.modules["gfootball.env"] = env_pkg

    scenarios_mod = types.ModuleType("gfootball.scenarios")
    scenarios_mod.e_PlayerRole_GK = 0
    sys.modules["gfootball.scenarios"] = scenarios_mod

    _load_module("gfootball.env.constants", gfootball_root / "env" / "constants.py")
    _load_module(
        "gfootball.env.football_action_set",
        gfootball_root / "env" / "football_action_set.py",
    )
    return _load_module(
        "gfootball.env.observation_processor",
        gfootball_root / "env" / "observation_processor.py",
    )


class _DummyPickledObject:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "_gameplayfootball":
            return _DummyPickledObject


class _ConfigWrapper(dict):
    def get_dictionary(self):
        return dict(self)
        try:
            __import__(module)
            mod = sys.modules[module]
            return getattr(mod, name)
        except Exception:
            return _DummyPickledObject


def rerender_dump(trace_file: Path, output_base: Path, repo_root: Path) -> Path:
    observation_processor = _bootstrap_gfootball_modules(repo_root)

    traces = []
    with trace_file.open("rb") as f:
        while True:
            try:
                traces.append(_SafeUnpickler(f).load())
            except EOFError:
                break
    if not traces:
        raise ValueError(f"No frames found in dump: {trace_file}")

    cfg = _ConfigWrapper(traces[0]["debug"]["config"])
    cfg["dump_full_episodes"] = True
    cfg["write_video"] = True
    cfg["display_game_stats"] = True
    cfg["tracesdir"] = str(output_base.parent)

    dump = observation_processor.ActiveDump(str(output_base), finish_step=len(traces) + 1, config=cfg)
    for trace in traces:
        dump.add_step(observation_processor.ObservationState(trace))
    dump.finalize()
    return output_base.with_suffix(f".{cfg['video_format']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-render a GRF dump with local label fixes.")
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--output-base", required=True, help="Path without extension for new .avi/.dump outputs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output = rerender_dump(Path(args.trace_file), Path(args.output_base), repo_root)
    print(f"[done] wrote video to {output}", flush=True)


if __name__ == "__main__":
    main()
