# Zero-Shot-MAPF

Zero-shot generalization in multi-agent pathfinding. This repository trains and evaluates a MAPPO policy that is designed to transfer across different map layouts and domains.

## What This Repo Contains

This codebase includes:

- A PettingZoo parallel MAPF environment with hybrid observations.
- A dedicated map-loading module that supports multiple map formats.
- MAPPO training over mixed scenarios (warehouse plus random blocked layouts).
- Interactive rollout playback for trained actors.
- Batch evaluation on hardcoded LORR benchmark scenarios.
- A utility to generate random environment files and a manifest.

## Script Overview

### MAPF.py

Environment implementation (PettingZoo ParallelEnv) for cooperative MAPF.

Main behavior:

- Observation mode: hybrid only.
	- vector: goal direction in agent-relative coordinates.
	- window: heading-aligned local tensor (rotated into agent frame).
		- local +y is forward, local +x is right.
		- channel 0: traversability map plus ego-center marker (1.0 at center).
		- channel 1 shared encoding for non-ego entities:
			- 0.5 other-agent goal
			- 1.0 other-agent position
			- 1.5 overlap of other-agent position and other-agent goal
		- channel 2: own goal indicator.
- Action space (Discrete(4)):
	- 0 forward
	- 1 turn right
	- 2 turn left
	- 3 wait
- Collision handling:
	- vertex collisions are canceled and penalized.
	- edge-swap collisions are canceled and penalized.
- Reward shaping:
	- small step penalty
	- wall-bump penalty
	- collision penalty
	- positive shaping for reducing Manhattan distance to goal
	- goal completion bonus (+10)

Task modes:
	- `lifelong=True` (default for training): immediate goal resampling on completion
	- `lifelong=False` (for Moving AI evaluation): agents marked as done when goal is reached
	- For benchmark/domain configs, goal resampling is constrained to task-defined goal locations only.
- Episode termination:
	- truncation by max step count only
- Rendering:
	- pygame human rendering and rgb_array frame export

Map content is no longer parsed directly in this file. It is delegated to map_loader.py.

### map_loader.py

Standalone map loading and parsing module used by MAPF.py.

Main entrypoint:

- load_map_configuration(path, requested_agents, default_grid_shape)

Supported map inputs:

1. .domain directory
- Finds JSON scenario configs in that directory.
- Selection strategy when multiple JSON files exist:
	- exact teamSize match to requested_agents if possible
	- otherwise smallest teamSize >= requested_agents
	- otherwise largest known teamSize
	- otherwise lexicographically first JSON

2. Benchmark JSON scenario
- Reads mapFile, agentFile, taskFile.
- Strict consistency checks:
	- teamSize must match the number of parsed agent starts from agentFile.
	- numTasksReveal must be > 0 and cannot exceed parsed task locations.
- agentFile format:
	- first line is number of agents n
	- next n lines each contain one integer start location index
- taskFile format:
	- first line is number of tasks m
	- next m lines each contain one or more integer location indices
	- order on each line is preserved
- Parses optional metadata:
	- teamSize
	- numTasksReveal
	- agentSize
	- maxCounter or agentCounter
	- delayConfig (object or relative JSON file path)

3. Moving AI octile format (.map files)
- Standard header: type, height, width, map
- ASCII grid with characters:
	- . (passable terrain)
	- G (passable terrain)
	- @ (out of bounds)
	- O (out of bounds)
	- T (trees, unpassable)
	- S (swamp, passable from regular terrain)
	- W (water, traversable but not passable from terrain)
- Blocked cells: @, O, T
- Note: When loading a .map file directly, no spawn/goal points are defined; they must be set externally or via the scenario parser

4. Legacy text map format
- Supports directives:
	- GRID H W
	- BLOCK x y
	- BLOCK_RECT x1 y1 x2 y2
	- SPAWN x y
	- GOAL x y

Returned object:

- MapLoadResult dataclass with grid shape, blocked cells, spawn/goal pools, domain metadata, and parsing mode flag.

Validation:

- Bounds checks for blocked cells, spawns, and goals.
- Spawn/goal overlap checks against blocked cells.
- Robust parsing errors for malformed files.

### train_mapf.py

Top-level training orchestration entrypoint for MAPPO.

This file now focuses on episode rollout orchestration, metrics/logging, checkpointing, and wiring the training modules listed below.

### train_core.py

Training core components extracted from train_mapf.py.

Contains:

- Transition dataclass
- RolloutBuffer with GAE and flattening utilities
- MAPPO algorithm class (policy/value update loop)

### train_models.py

Neural network model definitions used by training and inference.

Contains:

- ActorHybrid
- ActorMLP (legacy compatibility)
- ActorCNN (legacy compatibility)
- CentralCritic

### train_helpers.py

Environment/data helper functions used by train_mapf.py.

Contains:

- set_seed
- stack_global_state
- random layout manifest loading
- scenario-specific environment construction

Key components (imported from split modules):

- ActorHybrid: CNN over local window plus MLP over goal-direction vector.
- CentralCritic: CNN-based value function over global state tensor.
- RolloutBuffer with GAE.
- MAPPO update loop with clipping, entropy regularization, value loss, and grad clipping.

Training setup:

- Hybrid observation mode only.
- Mixed scenario training:
	- warehouse map from maps/warehouse16.domain/maps/warehouse_16x16.map
	- random blocked layouts loaded from generated_random_envs/manifest.json
- Random environments are sampled from a pool and reused for multiple episodes to reduce env rebuild overhead.
- TensorBoard metrics for episode-level stats, update-level stats, timing, and action fractions.
- Periodic checkpoint saving and final actor checkpoint save.

Outputs:

- TensorBoard logs in runs/
- Actor checkpoints (*.pth) in repo root by default

### run_mapf.py

Interactive/continuous rollout runner for a trained actor.

What it does:

- Creates MAPF environment from map_path.
- Infers observation spec from reset output.
- Loads ActorHybrid weights.
- Selects actions (stochastic sample or argmax).
- Renders environment each step.
- Prints per-episode return, goals reached, step count, and optional timing breakdown.

### eval_lorr_tasks.py

Batch evaluator for hardcoded LORR scenario JSON files.

What it does:

- Uses a hardcoded list of LORR scenarios.
- Loads actor and runs fixed number of episodes per scenario.
- Tracks task reaches by detecting goal changes per agent.
- Writes:
	- logs/lorr_task_reaches.json
	- logs/lorr_task_reaches.csv
- Records per-episode rows and per-scenario summary aggregates.

### eval_moving_ai.py

Batch evaluator for Moving AI MAPF benchmark scenarios.

What it does:

- Discovers all .scen files in MovingAI_eval/ directory.
- Groups each scenario by bucket (first column in .scen file).
- Runs one multi-agent traditional MAPF episode per bucket for up to 5000 timesteps.
- Uses all rows in a bucket as that episode's agents (start/goal pairs).
- Tracks whether all agents in the bucket finished before timeout and the number of steps taken.
- Outputs three files:
	- logs/moving_ai_results.json (all test results + scenario summaries)
	- logs/moving_ai_results.csv (per-test-case results)
	- logs/moving_ai_summary.csv (per-scenario success rates and statistics)

Configuration (edit at top of file):
- `ACTOR_PATH`: Path to trained actor model (.pth file)
- `OBS_RADIUS`: Must match training value (default 5 for mappo_hybrid_16x16_10agents_mix_actor.pth)
- `MAX_STEPS`: Episode length limit (default 5000)
- `MOVING_AI_DIR`: Directory containing .scen and .map files (default "MovingAI_eval")
- `SELECTED_SCENARIOS`: Optional scenario filename filter
- `SELECTED_BUCKETS`: Optional bucket-id filter
- `VISUALIZE_SINGLE_SELECTED_BUCKET`: If true, live-render when exactly one selected bucket is run

### generate_random_envs.py

Utility to pre-generate random blocked layouts on disk.

What it does:

- Samples obstacle density from a clipped normal distribution over a given range.
- Builds blocked-cell layouts while preserving enough free space for agents.
- Writes for each layout:
	- JSON layout file
	- legacy .map file (GRID/BLOCK lines)
- Writes manifest.json used by training.

## Data and Folder Roles

- maps/: Static map assets and domain files.
- generated_random_envs/: Pre-generated random layouts plus manifest.
- LORR_eval/: LORR benchmark scenarios used by eval_lorr_tasks.py.
- MovingAI_eval/: Additional benchmark assets.
- runs/: TensorBoard event logs from training.

## Typical Workflow

1. Generate random layout pool

```bash
python generate_random_envs.py --count 1000 --width 16 --height 16 --num-agents 10
```

2. Train policy

```bash
python train_mapf.py
```

3. Run trained policy with rendering

```bash
python run_mapf.py
```

4. Evaluate on LORR scenarios

```bash
python eval_lorr_tasks.py
```

5. Evaluate on Moving AI benchmarks

```bash
python eval_moving_ai.py
```

## Map Path Notes

MAPF accepts map_path values that match map_loader.py formats:

- .domain directory (expects JSON scenario files directly inside that directory)
- benchmark JSON scenario file
- Moving AI octile format .map file
- legacy text map file with GRID/BLOCK style directives

If a .domain directory has no scenario JSON files, loading will fail with a clear error.

## Environment Configuration

Key parameters for MAPF initialization:

- `grid_shape`: Environment dimensions (default 7x7 or inferred from map_path)
- `num_agents`: Number of agents (default 2 or inferred from domain config)
- `obs_radius`: Observation window radius (default 3; **must be 5 for eval_moving_ai.py**)
- `lifelong`: Task mode
	- `True` (default): Agents receive new goals on reaching current ones (training)
	- `False`: Agents marked as done when reaching goal (traditional MAPF, used for Moving AI evaluation)

## Main Artifacts Produced

- Trained actor weights: mappo_hybrid_..._actor.pth
- Intermediate checkpoints: mappo_hybrid_..._actor_epN.pth
- TensorBoard logs: runs/...
- LORR evaluation logs: logs/lorr_task_reaches.json and logs/lorr_task_reaches.csv
- Moving AI evaluation logs: logs/moving_ai_results.json, logs/moving_ai_results.csv, logs/moving_ai_summary.csv
- Random layout corpus: generated_random_envs/*.json, generated_random_envs/*.map, generated_random_envs/manifest.json
