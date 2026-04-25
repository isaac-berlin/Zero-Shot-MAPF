"""
Evaluate a trained model on Moving AI MAPF benchmark scenarios.

This script:
1. Finds all .scen files in MovingAI_eval directory
2. For each scenario file:
    - Ignores bucket ids in the .scen rows
    - Uses each feasible team size from {5, 10, 15, 20, 25}
    - Runs 10 random samples per team size from scenario rows
    - Runs one multi-agent traditional MAPF episode per sampled set
    - Runs the model for up to 5000 timesteps
    - Tracks completion status and number of steps
3. If exactly one selected scenario is run, optional live visualization is enabled
4. Outputs results to JSON and CSV logs
"""

import csv
import json
import random
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
from map_loader import MovingAIScenarioCase, parse_moving_ai_scenario
from run_mapf import load_actor_for_mode


# ============================================================
# Editable hardcoded config
# ============================================================
ACTOR_PATH = "mappo_hybrid_agents_mix_v3_actor.pth"
OBS_MODE = "hybrid"
STOCHASTIC = True
OBS_RADIUS = 5  # Must match training obs_radius
SEED = 0
DEVICE = None  # None => auto-select cuda if available, else cpu
OUTPUT_DIR = Path("logs")
SHOW_TQDM = True
MAX_STEPS = 5000
MOVING_AI_DIR = Path("MovingAI_eval")
SELECTED_SCENARIOS: List[str] = []  # e.g. ["Berlin_1_256-even-1.scen"]; empty => all
SAMPLED_AGENT_COUNTS: Tuple[int, ...] = (5, 10, 15, 20, 25)
RUNS_PER_TEAM_SIZE = 10
VISUALIZE_SINGLE_SELECTED_SCENARIO = True


def select_actions_batch(
    actor: torch.nn.Module,
    obs_dict: Dict,
    agent_order: List[str],
    obs_mode: str,
    device: str,
    stochastic: bool,
) -> Dict[str, int]:
    if obs_mode != "hybrid":
        raise ValueError("This evaluator now only supports obs_mode='hybrid'.")

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

    logits = actor(obs_t)
    dist = Categorical(logits=logits)

    if stochastic:
        actions_t = dist.sample()
    else:
        actions_t = torch.argmax(logits, dim=-1)

    return {a: int(actions_t[i].item()) for i, a in enumerate(agent_order)}


def run_moving_ai_sampled_test(
    test_key: str,
    sampled_cases: List[MovingAIScenarioCase],
    actor_path: str,
    obs_mode: str,
    device: str,
    stochastic: bool,
    obs_radius: int,
    seed: int,
    visualize: bool = False,
) -> Dict:
    """Run one traditional MAPF test for sampled rows from a scenario file."""
    if not sampled_cases:
        return {
            "test_case": test_key,
            "completed": False,
            "steps": 0,
            "num_agents": 0,
            "error": "No sampled cases.",
        }

    map_paths = {case.map_file for case in sampled_cases}
    if len(map_paths) != 1:
        return {
            "test_case": test_key,
            "completed": False,
            "steps": 0,
            "num_agents": len(sampled_cases),
            "error": "Sampled rows contain multiple map files.",
        }

    map_path = sampled_cases[0].map_file
    num_agents = len(sampled_cases)

    try:
        env = MAPF(
            obs_mode=obs_mode,
            map_path=map_path,
            obs_radius=obs_radius,
            num_agents=num_agents,
            lifelong=False,  # Traditional MAPF: no goal resampling
        )

        obs, _ = env.reset(seed=seed)
        agent_order = env.possible_agents[:]

        # Override starts/goals from sampled scenario rows.
        for idx, case in enumerate(sampled_cases):
            agent_name = agent_order[idx]
            env.agent_location[agent_name] = (case.start_x, case.start_y)
            env.goal_locations[agent_name] = (case.goal_x, case.goal_y)

        obs = env._get_observations()
        n_actions = env.action_space(agent_order[0]).n

        sample_obs = obs[agent_order[0]]
        actor = load_actor_for_mode(obs_mode, sample_obs, n_actions, device)
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        actor.eval()

        steps = 0
        completed = False

        pbar = None
        if SHOW_TQDM and tqdm is not None:
            pbar = tqdm(total=MAX_STEPS, desc=test_key, unit="step", leave=False)

        try:
            if visualize:
                env.render("human")

            while env.agents and steps < MAX_STEPS:
                active_agents = env.agents[:]
                with torch.no_grad():
                    actions = select_actions_batch(
                        actor=actor,
                        obs_dict=obs,
                        agent_order=active_agents,
                        obs_mode=obs_mode,
                        device=device,
                        stochastic=stochastic,
                    )

                obs, rewards, dones, truncs, infos = env.step(actions)
                del rewards, dones, truncs, infos
                steps += 1
                if pbar is not None:
                    pbar.update(1)
                if visualize:
                    env.render("human")

                # Traditional MAPF success: all agents finished before timeout.
                if not env.agents and steps < MAX_STEPS:
                    completed = True
                    break
        finally:
            if pbar is not None:
                pbar.close()
            env.close()

        return {
            "test_case": test_key,
            "completed": completed,
            "steps": steps,
            "max_steps": MAX_STEPS,
            "num_agents": num_agents,
            "map_file": Path(map_path).name,
        }

    except Exception as e:
        return {
            "test_case": test_key,
            "completed": False,
            "steps": 0,
            "num_agents": num_agents,
            "error": str(e),
        }


def run_scenario_file(
    scenario_path: Path,
    actor_path: str,
    obs_mode: str,
    device: str,
    stochastic: bool,
    obs_radius: int,
    seed: int,
    sampled_agent_counts: Tuple[int, ...],
    runs_per_team_size: int,
    visualize: bool = False,
) -> Tuple[List[Dict], Dict]:
    """
    Run repeated sampled-row tests from a Moving AI scenario file.
    
    Args:
        scenario_path: Path to the .scen file
        actor_path: Path to actor model
        obs_mode: Observation mode
        device: Device to run on
        stochastic: Whether to sample actions
        obs_radius: Observation radius
        seed: Random seed
        sampled_agent_counts: Candidate team sizes to sample from.
        runs_per_team_size: Number of random samples per feasible team size.
        visualize: Render the episode live if True.
        
    Returns:
        Tuple of (list of per-case results, summary dict)
    """
    try:
        cases = parse_moving_ai_scenario(str(scenario_path))
        if not cases:
            raise RuntimeError("Scenario file has no test rows.")

        rng = random.Random(seed)
        feasible_sizes = [n for n in sampled_agent_counts if n <= len(cases)]
        if not feasible_sizes:
            raise RuntimeError(
                f"Scenario has {len(cases)} rows, but no requested sample size fits: {sampled_agent_counts}."
            )

        results: List[Dict] = []
        completed_count = 0
        total_steps = 0

        total_runs = len(feasible_sizes) * runs_per_team_size
        run_index = 0

        for team_size in feasible_sizes:
            for rep in range(1, runs_per_team_size + 1):
                run_index += 1
                sampled_cases = rng.sample(cases, team_size)
                test_case_key = f"{scenario_path.stem}_n{team_size}_run{rep:02d}"

                print(
                    f"  [{run_index}/{total_runs}] {test_case_key}... ",
                    end="",
                    flush=True,
                )

                result = run_moving_ai_sampled_test(
                    test_key=test_case_key,
                    sampled_cases=sampled_cases,
                    actor_path=actor_path,
                    obs_mode=obs_mode,
                    device=device,
                    stochastic=stochastic,
                    obs_radius=obs_radius,
                    seed=seed + run_index,
                    visualize=visualize and total_runs == 1,
                )

                result["sampled_agents"] = team_size
                result["replicate"] = rep
                results.append(result)

                if result.get("completed", False):
                    completed_count += 1
                    total_steps += int(result.get("steps", 0))
                    print(f"✓ ({result['steps']} steps)")
                elif "error" in result:
                    print(f"✗ ERROR: {result['error']}")
                else:
                    total_steps += int(result.get("steps", 0))
                    print(f"✗ ({result['steps']} steps, timeout)")

        summary = {
            "scenario": str(scenario_path).replace("\\", "/"),
            "total_rows": len(cases),
            "team_sizes": ",".join(str(n) for n in feasible_sizes),
            "runs_per_team_size": runs_per_team_size,
            "runs": total_runs,
            "completed": completed_count,
            "success_rate": completed_count / total_runs if total_runs > 0 else 0.0,
            "total_steps": total_steps,
            "avg_steps_completed": total_steps / completed_count if completed_count > 0 else 0,
        }
        
        return results, summary
    
    except Exception as e:
        print(f"  FAILED to parse scenario: {e}")
        return [], {
            "scenario": str(scenario_path).replace("\\", "/"),
            "failed": True,
            "error": str(e),
        }


def write_logs(
    all_test_results: List[Dict],
    scenario_summaries: List[Dict],
    output_dir: Path,
) -> None:
    """Write results to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON output
    json_out = output_dir / "moving_ai_results.json"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario_summaries": scenario_summaries,
                "test_results": all_test_results,
            },
            f,
            indent=2,
        )
    
    # CSV output
    csv_out = output_dir / "moving_ai_results.csv"
    fieldnames = [
        "test_case",
        "sampled_agents",
        "replicate",
        "num_agents",
        "map_file",
        "completed",
        "steps",
        "max_steps",
        "error",
    ]
    
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_test_results:
            writer.writerow(row)
    
    # Scenario summary CSV
    summary_csv = output_dir / "moving_ai_summary.csv"
    summary_fieldnames = [
        "scenario",
        "total_rows",
        "team_sizes",
        "runs_per_team_size",
        "runs",
        "completed",
        "success_rate",
        "total_steps",
        "avg_steps_completed",
    ]
    
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for summary in scenario_summaries:
            if "failed" not in summary:
                writer.writerow(summary)


def find_moving_ai_scenarios(base_dir: Path) -> List[Path]:
    """Find all .scen files in the MovingAI_eval directory."""
    if not base_dir.exists():
        raise RuntimeError(f"Moving AI eval directory not found: {base_dir}")
    
    scen_files = sorted(base_dir.glob("*.scen"))
    
    if not scen_files:
        raise RuntimeError(f"No .scen files found in {base_dir}")
    
    return scen_files


def _filter_scenarios(all_scenarios: List[Path], selected_scenarios: List[str]) -> List[Path]:
    if not selected_scenarios:
        return all_scenarios

    wanted = set(selected_scenarios)
    filtered = [s for s in all_scenarios if s.name in wanted]
    missing = sorted(wanted - {s.name for s in filtered})
    if missing:
        raise RuntimeError("Selected scenario files not found: " + ", ".join(missing))
    return filtered


def main() -> None:
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        moving_ai_scenarios = find_moving_ai_scenarios(MOVING_AI_DIR)
        moving_ai_scenarios = _filter_scenarios(moving_ai_scenarios, SELECTED_SCENARIOS)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return
    
    print(f"Found {len(moving_ai_scenarios)} Moving AI scenario files")
    print(f"Running actor: {ACTOR_PATH}")
    print(
        f"obs_mode={OBS_MODE}, stochastic={STOCHASTIC}, "
        f"max_steps={MAX_STEPS}, device={device}"
    )
    print(f"Sample sizes: {list(SAMPLED_AGENT_COUNTS)}")
    print(f"Runs per team size: {RUNS_PER_TEAM_SIZE}")
    if SELECTED_SCENARIOS:
        print(f"Selected scenarios: {SELECTED_SCENARIOS}")
    print()
    
    all_test_results: List[Dict] = []
    all_summaries: List[Dict] = []
    
    for idx, scenario_path in enumerate(moving_ai_scenarios, start=1):
        print(f"[{idx}/{len(moving_ai_scenarios)}] {scenario_path.name}")
        
        test_results, summary = run_scenario_file(
            scenario_path=scenario_path,
            actor_path=ACTOR_PATH,
            obs_mode=OBS_MODE,
            device=device,
            stochastic=STOCHASTIC,
            obs_radius=OBS_RADIUS,
            seed=SEED + idx,
            sampled_agent_counts=SAMPLED_AGENT_COUNTS,
            runs_per_team_size=RUNS_PER_TEAM_SIZE,
            visualize=VISUALIZE_SINGLE_SELECTED_SCENARIO and len(moving_ai_scenarios) == 1,
        )
        
        all_test_results.extend(test_results)
        all_summaries.append(summary)
        
        if "failed" not in summary:
            print(
                f"  Summary: {summary['completed']}/{summary['runs']} completed "
                f"(team_sizes={summary['team_sizes']}), "
                f"success_rate={summary['success_rate']:.2%}"
            )
        print()
    
    write_logs(all_test_results, all_summaries, OUTPUT_DIR)
    
    print("Wrote logs:")
    print(f"  {(OUTPUT_DIR / 'moving_ai_results.json').as_posix()}")
    print(f"  {(OUTPUT_DIR / 'moving_ai_results.csv').as_posix()}")
    print(f"  {(OUTPUT_DIR / 'moving_ai_summary.csv').as_posix()}")


if __name__ == "__main__":
    main()
