"""
Evaluate a trained model on Moving AI MAPF benchmark scenarios.

This script:
1. Finds all .scen files in MovingAI_eval directory
2. For each scenario and bucket group:
    - Groups rows by the first column (bucket id)
    - Runs one multi-agent traditional MAPF episode per bucket
    - Sets starts/goals from all rows in that bucket
    - Runs the model for up to 5000 timesteps
    - Tracks completion status and number of steps
3. If exactly one selected bucket is run, optional live visualization is enabled
4. Outputs results to JSON and CSV logs
"""

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
from map_loader import MovingAIScenarioCase, parse_moving_ai_scenario
from run_mapf import load_actor_for_mode


# ============================================================
# Editable hardcoded config
# ============================================================
ACTOR_PATH = "mappo_hybrid_16x16_10agents_mix_actor.pth"
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
SELECTED_BUCKETS: List[int] = []  # e.g. [11]; empty => all
VISUALIZE_SINGLE_SELECTED_BUCKET = True


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


def _group_cases_by_bucket(cases: List[MovingAIScenarioCase]) -> List[Tuple[int, List[MovingAIScenarioCase]]]:
    grouped: Dict[int, List[MovingAIScenarioCase]] = {}
    ordered_buckets: List[int] = []
    for case in cases:
        if case.bucket not in grouped:
            grouped[case.bucket] = []
            ordered_buckets.append(case.bucket)
        grouped[case.bucket].append(case)
    return [(bucket, grouped[bucket]) for bucket in ordered_buckets]


def run_moving_ai_bucket_test(
    test_key: str,
    bucket: int,
    bucket_cases: List[MovingAIScenarioCase],
    actor_path: str,
    obs_mode: str,
    device: str,
    stochastic: bool,
    obs_radius: int,
    seed: int,
    visualize: bool = False,
) -> Dict:
    """Run one traditional MAPF test for all rows in a single bucket."""
    if not bucket_cases:
        return {
            "test_case": test_key,
            "bucket": bucket,
            "completed": False,
            "steps": 0,
            "num_agents": 0,
            "error": "Empty bucket.",
        }

    map_paths = {case.map_file for case in bucket_cases}
    if len(map_paths) != 1:
        return {
            "test_case": test_key,
            "bucket": bucket,
            "completed": False,
            "steps": 0,
            "num_agents": len(bucket_cases),
            "error": "Bucket contains multiple map files.",
        }

    map_path = bucket_cases[0].map_file
    num_agents = len(bucket_cases)

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

        # Override starts/goals from this bucket.
        for idx, case in enumerate(bucket_cases):
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
            "bucket": bucket,
            "completed": completed,
            "steps": steps,
            "max_steps": MAX_STEPS,
            "num_agents": num_agents,
            "map_file": Path(map_path).name,
        }

    except Exception as e:
        return {
            "test_case": test_key,
            "bucket": bucket,
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
    max_buckets: int = None,
    selected_buckets: List[int] = None,
    visualize_single_selected_bucket: bool = False,
) -> Tuple[List[Dict], Dict]:
    """
    Run all bucket tests from a Moving AI scenario file.
    
    Args:
        scenario_path: Path to the .scen file
        actor_path: Path to actor model
        obs_mode: Observation mode
        device: Device to run on
        stochastic: Whether to sample actions
        obs_radius: Observation radius
        seed: Random seed
        max_buckets: Maximum number of buckets to run (None = all)
        selected_buckets: Optional explicit bucket ids to evaluate
        visualize_single_selected_bucket: Render live if exactly one bucket is selected
        
    Returns:
        Tuple of (list of per-case results, summary dict)
    """
    try:
        cases = parse_moving_ai_scenario(str(scenario_path))
        bucket_groups = _group_cases_by_bucket(cases)

        if selected_buckets:
            selected = set(selected_buckets)
            bucket_groups = [pair for pair in bucket_groups if pair[0] in selected]

        if max_buckets is not None:
            bucket_groups = bucket_groups[:max_buckets]

        should_visualize = visualize_single_selected_bucket and len(bucket_groups) == 1
        
        results = []
        completed_count = 0
        total_steps = 0

        for idx, (bucket, bucket_cases) in enumerate(bucket_groups, start=1):
            test_case_key = f"{scenario_path.stem}_bucket_{bucket}"

            print(
                f"  [{idx}/{len(bucket_groups)}] {test_case_key} "
                f"(agents={len(bucket_cases)})... ",
                end="",
                flush=True,
            )

            result = run_moving_ai_bucket_test(
                test_key=test_case_key,
                bucket=bucket,
                bucket_cases=bucket_cases,
                actor_path=actor_path,
                obs_mode=obs_mode,
                device=device,
                stochastic=stochastic,
                obs_radius=obs_radius,
                seed=seed + idx,
                visualize=should_visualize,
            )
            
            results.append(result)
            
            if result.get("completed", False):
                completed_count += 1
                total_steps += result["steps"]
                print(f"✓ ({result['steps']} steps)")
            elif "error" in result:
                print(f"✗ ERROR: {result['error']}")
            else:
                total_steps += result["steps"]
                print(f"✗ ({result['steps']} steps, timeout)")
        
        summary = {
            "scenario": str(scenario_path).replace("\\", "/"),
            "total_buckets": len(bucket_groups),
            "total_rows": len(cases),
            "completed": completed_count,
            "success_rate": completed_count / len(bucket_groups) if bucket_groups else 0,
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
    fieldnames = ["test_case", "bucket", "num_agents", "map_file", "completed", "steps", "max_steps", "error"]
    
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_test_results:
            writer.writerow(row)
    
    # Scenario summary CSV
    summary_csv = output_dir / "moving_ai_summary.csv"
    summary_fieldnames = [
        "scenario",
        "total_buckets",
        "total_rows",
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
    if SELECTED_SCENARIOS:
        print(f"Selected scenarios: {SELECTED_SCENARIOS}")
    if SELECTED_BUCKETS:
        print(f"Selected buckets: {sorted(SELECTED_BUCKETS)}")
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
            seed=SEED,
            selected_buckets=SELECTED_BUCKETS,
            visualize_single_selected_bucket=VISUALIZE_SINGLE_SELECTED_BUCKET,
        )
        
        all_test_results.extend(test_results)
        all_summaries.append(summary)
        
        if "failed" not in summary:
            print(
                f"  Summary: {summary['completed']}/{summary['total_buckets']} completed, "
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
