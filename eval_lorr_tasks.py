import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from MAPF import MAPF
from run_mapf import load_actor_for_mode


# ============================================================
# Editable hardcoded config
# ============================================================
ACTOR_PATH = "mappo_hybrid_random_32_32_20_10_actor.pth"
OBS_MODE = "hybrid"  # vector | window | knn | hybrid
EPISODES = 1
STOCHASTIC = True
OBS_RADIUS = 10
K_AGENTS = 5
SEED = 0
DEVICE = None  # None => auto-select cuda if available, else cpu
OUTPUT_DIR = Path("logs")
SHOW_TQDM = True

# Hardcoded LORR scenarios. Edit this list directly if needed.
LORR_SCENARIOS = [
    Path("LORR_eval/city.domain/MR23-I-01.json"),
    Path("LORR_eval/city.domain/MR23-I-02.json"),
    Path("LORR_eval/game.domain/MR23-I-09.json"),
    Path("LORR_eval/random.domain/MR23-I-03.json"),
    Path("LORR_eval/random.domain/MR23-I-04.json"),
    Path("LORR_eval/random.domain/MR23-I-05.json"),
    Path("LORR_eval/random.domain/MR23-I-07.json"),
    Path("LORR_eval/random.domain/MR23-I-08.json"),
    Path("LORR_eval/warehouse.domain/MR23-I-06.json"),
    Path("LORR_eval/warehouse.domain/MR23-I-10.json"),
]


def select_actions_batch(
    actor: torch.nn.Module,
    obs_dict: Dict,
    agent_order: List[str],
    obs_mode: str,
    device: str,
    stochastic: bool,
) -> Dict[str, int]:
    if obs_mode == "hybrid":
        obs_t = {
            "vector": torch.tensor(
                np.stack([obs_dict[a]["vector"] for a in agent_order]),
                dtype=torch.float32,
                device=device,
            ),
            "window": torch.tensor(
                np.stack([obs_dict[a]["window"] for a in agent_order]),
                dtype=torch.float32,
                device=device,
            ),
        }
    else:
        obs_t = torch.tensor(
            np.stack([obs_dict[a] for a in agent_order]),
            dtype=torch.float32,
            device=device,
        )

    logits = actor(obs_t)
    dist = Categorical(logits=logits)

    if stochastic:
        actions_t = dist.sample()
    else:
        actions_t = torch.argmax(logits, dim=-1)

    return {a: int(actions_t[i].item()) for i, a in enumerate(agent_order)}


def run_scenario(
    scenario_path: Path,
    actor_path: str,
    obs_mode: str,
    device: str,
    stochastic: bool,
    episodes: int,
    obs_radius: int,
    k_agents: int,
    seed: int,
) -> Tuple[List[Dict], Dict]:
    env = MAPF(
        obs_mode=obs_mode,
        map_path=str(scenario_path),
        obs_radius=obs_radius,
        k_agents=k_agents,
    )

    try:
        obs, _ = env.reset(seed=seed)
        agent_order = env.possible_agents[:]
        n_actions = env.action_space(agent_order[0]).n

        sample_obs = obs[agent_order[0]]
        actor = load_actor_for_mode(obs_mode, sample_obs, n_actions, device)
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        actor.eval()

        per_episode_rows: List[Dict] = []
        aggregate_counts = {a: 0 for a in agent_order}
        aggregate_steps = 0

        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done_flags = {a: False for a in agent_order}
            episode_counts = {a: 0 for a in agent_order}
            steps = 0
            pbar = None

            if SHOW_TQDM and tqdm is not None:
                pbar = tqdm(
                    total=env.max_steps,
                    desc=f"{scenario_path.name} | ep {ep + 1}/{episodes}",
                    unit="step",
                    leave=False,
                )

            try:
                while env.agents and not all(done_flags.values()):
                    with torch.no_grad():
                        actions = select_actions_batch(
                            actor=actor,
                            obs_dict=obs,
                            agent_order=agent_order,
                            obs_mode=obs_mode,
                            device=device,
                            stochastic=stochastic,
                        )

                    previous_goals = {a: env.goal_locations[a] for a in agent_order}
                    obs, rewards, dones, truncs, infos = env.step(actions)
                    del rewards, infos

                    for a in agent_order:
                        if env.goal_locations[a] != previous_goals[a]:
                            episode_counts[a] += 1
                            aggregate_counts[a] += 1

                    steps += 1
                    if pbar is not None:
                        pbar.update(1)
                    done_flags = {a: (dones.get(a, False) or truncs.get(a, False)) for a in agent_order}
            finally:
                if pbar is not None:
                    pbar.close()

            aggregate_steps += steps
            per_episode_rows.append(
                {
                    "scenario": str(scenario_path).replace("\\", "/"),
                    "episode": ep,
                    "steps": steps,
                    "total_reaches": int(sum(episode_counts.values())),
                    "per_agent_reaches": episode_counts,
                }
            )

        summary = {
            "scenario": str(scenario_path).replace("\\", "/"),
            "episodes": episodes,
            "steps_total": aggregate_steps,
            "total_reaches": int(sum(aggregate_counts.values())),
            "per_agent_reaches": aggregate_counts,
            "num_agents": len(agent_order),
        }

        return per_episode_rows, summary
    finally:
        env.close()


def write_logs(
    episode_rows: List[Dict],
    scenario_summaries: List[Dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_out = output_dir / "lorr_task_reaches.json"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario_summaries": scenario_summaries,
                "episode_rows": episode_rows,
            },
            f,
            indent=2,
        )

    max_agents = 0
    for row in episode_rows:
        max_agents = max(max_agents, len(row["per_agent_reaches"]))

    csv_out = output_dir / "lorr_task_reaches.csv"
    fieldnames = ["scenario", "episode", "steps", "total_reaches"] + [
        f"agent_{i}_reaches" for i in range(max_agents)
    ]

    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in episode_rows:
            flat = {
                "scenario": row["scenario"],
                "episode": row["episode"],
                "steps": row["steps"],
                "total_reaches": row["total_reaches"],
            }
            for i in range(max_agents):
                flat[f"agent_{i}_reaches"] = row["per_agent_reaches"].get(f"agent_{i}", "")
            writer.writerow(flat)


def main() -> None:
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    scenarios = [Path(p) for p in LORR_SCENARIOS]

    if not scenarios:
        raise RuntimeError("LORR_SCENARIOS is empty. Add at least one scenario path.")

    missing = [str(p) for p in scenarios if not p.exists()]
    if missing:
        raise RuntimeError(
            "These hardcoded scenario files do not exist:\n"
            + "\n".join(missing)
        )

    print(f"Using {len(scenarios)} hardcoded LORR scenarios")
    print(f"Running actor: {ACTOR_PATH}")
    print(f"obs_mode={OBS_MODE}, episodes={EPISODES}, stochastic={STOCHASTIC}, device={device}")

    all_episode_rows: List[Dict] = []
    all_summaries: List[Dict] = []

    for idx, scenario in enumerate(scenarios, start=1):
        print(f"[{idx}/{len(scenarios)}] Running scenario: {scenario}")
        try:
            episode_rows, summary = run_scenario(
                scenario_path=scenario,
                actor_path=ACTOR_PATH,
                obs_mode=OBS_MODE,
                device=device,
                stochastic=STOCHASTIC,
                episodes=EPISODES,
                obs_radius=OBS_RADIUS,
                k_agents=K_AGENTS,
                seed=SEED,
            )
            all_episode_rows.extend(episode_rows)
            all_summaries.append(summary)
            print(
                f"  total_reaches={summary['total_reaches']}, "
                f"steps_total={summary['steps_total']}, agents={summary['num_agents']}"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            all_summaries.append(
                {
                    "scenario": str(scenario).replace("\\", "/"),
                    "failed": True,
                    "error": str(e),
                }
            )

    write_logs(all_episode_rows, all_summaries, OUTPUT_DIR)

    print("\nWrote logs:")
    print(f"  {(OUTPUT_DIR / 'lorr_task_reaches.json').as_posix()}")
    print(f"  {(OUTPUT_DIR / 'lorr_task_reaches.csv').as_posix()}")


if __name__ == "__main__":
    main()
