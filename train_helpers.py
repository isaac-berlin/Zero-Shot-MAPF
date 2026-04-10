import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from MAPF import MAPF


WAREHOUSE_MAP_PATH = Path("maps") / "warehouse16.domain" / "maps" / "warehouse_16x16.map"
WAREHOUSE_ONEWIDE_MAP_PATH = (
    Path("maps") / "warehouse16.domain" / "maps" / "warehouse_16x16_onewide.map"
)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stack_global_state(env) -> np.ndarray:
    """
    Global CTDE state for MAPF as a 3-channel grid tensor:
      channel 0: blocked cells
      channel 1: agent locations encoded by heading
      channel 2: goal locations encoded by agent id
    """
    h, w = env.grid_h, env.grid_w
    state = np.zeros((3, h, w), dtype=np.float32)

    state[0, :, :] = env._blocked_grid.astype(np.float32)

    num_agents = len(env.possible_agents)
    for i, agent in enumerate(env.possible_agents):
        agent_id = (i + 1) / num_agents

        ax, ay = env.agent_location[agent]
        state[1, ay, ax] = (env.agent_dir[agent] + 1) / 4.0

        gx, gy = env.goal_locations[agent]
        state[2, gy, gx] = agent_id

    return state


def _load_random_layout(entry: dict, layout_dir: Path) -> dict:
    json_name = entry["json"]
    json_path = layout_dir / json_name
    if not json_path.exists():
        raise FileNotFoundError(f"Missing random layout file: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        layout = json.load(f)

    blocked_cells = layout.get("blocked_cells", [])
    if not isinstance(blocked_cells, list):
        raise ValueError(f"{json_path}: blocked_cells must be a list.")

    grid_shape = layout.get("grid_shape")
    if not isinstance(grid_shape, list) or len(grid_shape) != 2:
        raise ValueError(f"{json_path}: grid_shape must be a two-item list.")

    return {
        "blocked_cells": tuple(tuple(cell) for cell in blocked_cells),
        "obstacle_density": float(layout.get("obstacle_density", 0.0)),
        "grid_shape": tuple(grid_shape),
        "num_agents": int(layout.get("num_agents", 0)),
        "name": entry.get("name", json_path.stem),
    }


def build_random_layout_pool_from_dir(layout_dir: Path, pool_size: int) -> List[dict]:
    manifest_path = layout_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Expected random environment manifest at {manifest_path}. Run generate_random_envs.py first."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"{manifest_path}: manifest must contain a non-empty list of layouts.")

    selected_entries = manifest[:pool_size]
    return [_load_random_layout(entry, layout_dir) for entry in selected_entries]


def build_episode_env(
    scenario,
    obs_radius,
    num_agents,
    grid_shape,
    random_layout: Optional[dict] = None,
):
    if scenario == "warehouse":
        return MAPF(
            obs_mode="hybrid",
            obs_radius=obs_radius,
            map_path=str(WAREHOUSE_MAP_PATH),
            num_agents=num_agents,
            grid_shape=grid_shape,
        ), {
            "scenario": "warehouse",
            "obstacle_density": 0.0,
            "map_path": str(WAREHOUSE_MAP_PATH),
        }

    if scenario == "warehouse_onewide":
        return MAPF(
            obs_mode="hybrid",
            obs_radius=obs_radius,
            map_path=str(WAREHOUSE_ONEWIDE_MAP_PATH),
            num_agents=num_agents,
            grid_shape=grid_shape,
        ), {
            "scenario": "warehouse_onewide",
            "obstacle_density": 0.0,
            "map_path": str(WAREHOUSE_ONEWIDE_MAP_PATH),
        }

    if scenario == "random":
        if random_layout is None:
            raise ValueError("random_layout is required when using disk-backed random environments.")

        env = MAPF(
            obs_mode="hybrid",
            obs_radius=obs_radius,
            num_agents=num_agents,
            grid_shape=grid_shape,
            blocked_cells=set(random_layout["blocked_cells"]),
        )

        return env, {
            "scenario": "random",
            "obstacle_density": float(random_layout["obstacle_density"]),
            "blocked_cell_count": len(random_layout["blocked_cells"]),
        }

    raise ValueError(f"Unknown training scenario: {scenario!r}")
