from typing import Dict, Optional, List, Tuple, Set
import json
import os
import re
import random
import heapq
import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import pygame


class MAPF(ParallelEnv):
    """
    Unified Multi-Agent Pathfinding Environment.

    Observation mode:
        - "hybrid": local window (3 channels) plus a 2D goal-direction vector

    Task: Cooperative MAPF with per-agent assigned goals.

    Actions (Discrete(4)):
        0: forward
        1: turn right
        2: turn left
        3: wait

    Headings:
        0 = North (+y)
        1 = East  (+x)
        2 = South (-y)
        3 = West  (-x)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "Pathfinding-v0",
    }

    def __init__(
        self,
        grid_shape=(7, 7),
        num_agents: Optional[int] = None,
        obs_mode="hybrid",
        obs_radius=3,
        map_path=None,       # optional .json config or .domain directory
    ):

        if obs_mode != "hybrid":
            raise ValueError("This environment now only supports obs_mode='hybrid'.")
        self.obs_mode = obs_mode
        self.map_path = map_path

        self.grid_h = grid_shape[0]
        self.grid_w = grid_shape[1]
        
        self.n_agents = int(num_agents) if num_agents is not None else 2
        self.obs_radius = obs_radius
        self.max_steps = 5000
        self.timestep = 0
        self._using_domain_config = False

        # Optional metadata loaded from benchmark JSON configs.
        self.team_size: Optional[int] = None
        self.num_tasks_reveal: Optional[float] = None
        self.agent_size: Optional[float] = None
        self.max_counter: Optional[int] = None
        self.delay_config: Dict = {}
        
        # Map configuration
        self.blocked: Set[Tuple[int, int]] = set()
        self._blocked_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.bool_)
        self.spawn_points: List[Tuple[int, int]] = []
        self.goal_points: List[Tuple[int, int]] = []
        
        if map_path is not None:
            self._load_map_file(map_path)
            if num_agents is None and self.team_size is not None:
                self.n_agents = self.team_size
        self._refresh_blocked_grid()

        # Agents
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = self.possible_agents[:]
        self.agent_location = {a: (0, 0) for a in self.agents}

        # Agent headings (0=N,1=E,2=S,3=W)
        self.agent_dir = {a: 0 for a in self.agents}

        # Goals (one per agent)
        self.goal_locations = {a: (0, 0) for a in self.agents}

        # Action space: forward, turn right, turn left, wait
        self.action_spaces = {
            agent: spaces.Discrete(4)
            for agent in self.possible_agents
        }

        # Observation spaces depend on mode
        self.observation_spaces = {
            agent: self._build_observation_space()
            for agent in self.possible_agents
        }

        # Rendering
        self.render_mode = "human"
        self._pygame_initialized = False
        
        # Rendering sizing limits (tweak as you like)
        self._max_window_w = 2000 
        self._max_window_h = 1500
        self._min_cell_size = 2
        self._max_cell_size = 64

        # Default; will be overwritten dynamically in _init_pygame()
        self._cell_size = 32
        self._margin = 20
        
        self._screen = None
        self._clock = None
        self._font = None
        self._static_surface = None
        self._label_cache: Dict[Tuple[str, Tuple[int, int, int]], pygame.Surface] = {}

    def _refresh_blocked_grid(self) -> None:
        self._blocked_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.bool_)
        for bx, by in self.blocked:
            if 0 <= bx < self.grid_w and 0 <= by < self.grid_h:
                self._blocked_grid[by, bx] = True
        
    # ============================================================
    # Map file parsing
    # ============================================================
    def _load_map_file(self, path: str) -> None:
        """
        Load either:
        - .domain directory containing one or more benchmark JSON configs,
        - benchmark JSON config (.json) that points to .map/.agents/.tasks files, or
        - legacy text directive format.
        """
        resolved_path = path
        norm = os.path.normpath(path)
        if os.path.isdir(norm) or norm.lower().endswith(".domain"):
            resolved_path = self._resolve_domain_config_json(norm)

        if resolved_path.lower().endswith(".json"):
            self._load_benchmark_json(resolved_path)
        else:
            self._load_legacy_map_file(resolved_path)

    def _resolve_domain_config_json(self, domain_dir: str) -> str:
        """
        Resolve a .domain directory into a concrete benchmark JSON config file.

        Selection strategy:
        - if only one JSON exists: use it
        - if multiple JSON files exist and num_agents is provided: prefer exact teamSize,
          otherwise the smallest teamSize >= num_agents, otherwise the largest teamSize
        - if no teamSize metadata can be used: pick lexicographically first file
        """
        if not os.path.isdir(domain_dir):
            raise ValueError(f"{domain_dir}: expected an existing .domain directory.")

        json_candidates = [
            os.path.join(domain_dir, name)
            for name in os.listdir(domain_dir)
            if name.lower().endswith(".json")
        ]
        json_candidates.sort()

        if not json_candidates:
            raise ValueError(f"{domain_dir}: no JSON scenario files found in .domain directory.")
        if len(json_candidates) == 1:
            return json_candidates[0]

        # Try selecting by teamSize when multiple scenarios are present.
        annotated: List[Tuple[str, Optional[int]]] = []
        for cfg_path in json_candidates:
            team_size = None
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "teamSize" in cfg:
                    team_size = int(cfg["teamSize"])
            except Exception:
                team_size = None
            annotated.append((cfg_path, team_size))

        if self.n_agents is not None:
            exact = [p for (p, ts) in annotated if ts == self.n_agents]
            if exact:
                return sorted(exact)[0]

            larger_or_equal = sorted(
                [(p, ts) for (p, ts) in annotated if ts is not None and ts >= self.n_agents],
                key=lambda item: (item[1], item[0]),
            )
            if larger_or_equal:
                return larger_or_equal[0][0]

            known_sizes = sorted(
                [(p, ts) for (p, ts) in annotated if ts is not None],
                key=lambda item: (item[1], item[0]),
            )
            if known_sizes:
                return known_sizes[-1][0]

        return annotated[0][0]

    def _load_benchmark_json(self, path: str) -> None:
        """Load benchmark-style config with mapFile/agentFile/taskFile fields."""
        self._using_domain_config = True
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        base_dir = os.path.dirname(path)

        def resolve_rel(rel_path: str) -> str:
            return os.path.normpath(os.path.join(base_dir, rel_path))

        try:
            map_file = resolve_rel(cfg["mapFile"])
            agent_file = resolve_rel(cfg["agentFile"])
            task_file = resolve_rel(cfg["taskFile"])
        except KeyError as e:
            raise ValueError(f"{path}: missing required key {e.args[0]!r}") from e

        width, height, blocked = self._parse_octile_map_file(map_file)
        self.grid_w = width
        self.grid_h = height
        self.blocked = blocked

        self.spawn_points = self._parse_agents_file(agent_file)
        self.goal_points = self._parse_tasks_file(task_file)

        if "teamSize" in cfg:
            self.team_size = int(cfg["teamSize"])
        if "numTasksReveal" in cfg:
            self.num_tasks_reveal = float(cfg["numTasksReveal"])
        if "agentSize" in cfg:
            self.agent_size = float(cfg["agentSize"])
        # Some files use "agentCounter" while docs call it "maxCounter".
        if "maxCounter" in cfg:
            self.max_counter = int(cfg["maxCounter"])
        elif "agentCounter" in cfg:
            self.max_counter = int(cfg["agentCounter"])

        delay_cfg = cfg.get("delayConfig", {})
        if isinstance(delay_cfg, str):
            delay_path = resolve_rel(delay_cfg)
            with open(delay_path, "r", encoding="utf-8") as f:
                self.delay_config = json.load(f)
        elif isinstance(delay_cfg, dict):
            self.delay_config = delay_cfg
        else:
            raise ValueError(f"{path}: delayConfig must be an object or relative json path.")

        self._validate_points(path, self.blocked, self.spawn_points, self.goal_points)
        self.spawn_points = self._dedup_points(self.spawn_points)
        self.goal_points = self._dedup_points(self.goal_points)
        self._refresh_blocked_grid()

    def _parse_octile_map_file(self, path: str) -> Tuple[int, int, Set[Tuple[int, int]]]:
        """
        Parse a map in movingai "octile" text format and return (width, height, blocked).

        Coordinates are converted into this env's (x, y) with y growing upward.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw_lines = [line.rstrip("\n") for line in f]

        if len(raw_lines) < 4:
            raise ValueError(f"{path}: invalid map file, expected at least 4 header lines.")

        height = None
        width = None
        map_start = None

        for i, line in enumerate(raw_lines):
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower.startswith("height"):
                parts = stripped.split()
                if len(parts) < 2:
                    raise ValueError(f"{path}:{i + 1}: malformed height line.")
                height = int(parts[1])
            elif lower.startswith("width"):
                parts = stripped.split()
                if len(parts) < 2:
                    raise ValueError(f"{path}:{i + 1}: malformed width line.")
                width = int(parts[1])
            elif lower == "map":
                map_start = i + 1
                break

        if height is None or width is None or map_start is None:
            raise ValueError(f"{path}: expected header fields: height, width, and map.")
        if height <= 1 or width <= 1:
            raise ValueError(f"{path}: height and width must be > 1.")

        if len(raw_lines) < map_start + height:
            raise ValueError(f"{path}: expected {height} map rows, got fewer.")

        blocked: Set[Tuple[int, int]] = set()
        obstacle_symbols = {"@", "T"}

        for row in range(height):
            map_row = raw_lines[map_start + row]
            if len(map_row) < width:
                raise ValueError(
                    f"{path}:{map_start + row + 1}: map row too short (expected {width}, got {len(map_row)})."
                )
            y = height - 1 - row
            for x, symbol in enumerate(map_row[:width]):
                if symbol in obstacle_symbols:
                    blocked.add((x, y))

        return width, height, blocked

    def _linear_to_xy(self, loc: int, width: int, height: int, source: str) -> Tuple[int, int]:
        if loc < 0:
            raise ValueError(f"{source}: negative location index {loc}.")
        row = loc // width
        col = loc % width
        if row >= height:
            raise ValueError(f"{source}: location index {loc} is out of bounds for {height}x{width}.")
        # Input row increases downward; env y increases upward.
        return col, (height - 1 - row)

    def _parse_agents_file(self, path: str) -> List[Tuple[int, int]]:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        if not lines:
            raise ValueError(f"{path}: empty agents file.")

        try:
            n_agents = int(lines[0])
        except ValueError as e:
            raise ValueError(f"{path}: first non-comment line must be integer agent count.") from e

        if len(lines) < 1 + n_agents:
            raise ValueError(f"{path}: expected {n_agents} agent locations, found {len(lines) - 1}.")

        starts: List[Tuple[int, int]] = []
        for i in range(n_agents):
            line = lines[1 + i]
            nums = re.findall(r"-?\d+", line)
            if not nums:
                raise ValueError(f"{path}:{i + 2}: missing location index.")
            loc = int(nums[0])
            starts.append(self._linear_to_xy(loc, self.grid_w, self.grid_h, f"{path}:{i + 2}"))

        return starts

    def _parse_tasks_file(self, path: str) -> List[Tuple[int, int]]:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        if not lines:
            raise ValueError(f"{path}: empty tasks file.")

        try:
            n_tasks = int(lines[0])
        except ValueError as e:
            raise ValueError(f"{path}: first non-comment line must be integer task count.") from e

        if len(lines) < 1 + n_tasks:
            raise ValueError(f"{path}: expected {n_tasks} task lines, found {len(lines) - 1}.")

        points: List[Tuple[int, int]] = []
        for i in range(n_tasks):
            line = lines[1 + i]
            nums = re.findall(r"-?\d+", line)
            if not nums:
                raise ValueError(f"{path}:{i + 2}: task line has no locations.")
            for token in nums:
                loc = int(token)
                points.append(self._linear_to_xy(loc, self.grid_w, self.grid_h, f"{path}:{i + 2}"))

        return points

    @staticmethod
    def _dedup_points(seq: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        seen = set()
        out = []
        for item in seq:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _validate_points(
        self,
        source: str,
        blocked: Set[Tuple[int, int]],
        spawns: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
    ) -> None:
        def in_bounds(p: Tuple[int, int]) -> bool:
            x, y = p
            return 0 <= x < self.grid_w and 0 <= y < self.grid_h

        for p in blocked:
            if not in_bounds(p):
                raise ValueError(f"{source}: blocked cell out of bounds: {p} for GRID {self.grid_h}x{self.grid_w}")
        for p in spawns:
            if not in_bounds(p):
                raise ValueError(f"{source}: spawn out of bounds: {p} for GRID {self.grid_h}x{self.grid_w}")
            if p in blocked:
                raise ValueError(f"{source}: spawn on blocked cell: {p}")
        for p in goals:
            if not in_bounds(p):
                raise ValueError(f"{source}: goal out of bounds: {p} for GRID {self.grid_h}x{self.grid_w}")
            if p in blocked:
                raise ValueError(f"{source}: goal on blocked cell: {p}")

    def _load_legacy_map_file(self, path: str) -> None:
        """
        Load a text map specification.

        Supported directives (case-insensitive), one per line:

            GRID <H> <W>
                - sets grid shape to (H, W) (optional; overrides constructor arg)

            BLOCK <x> <y>
                - marks a single blocked cell

            BLOCK_RECT <x1> <y1> <x2> <y2>
                - marks a rectangle of blocked cells (inclusive)

            SPAWN <x> <y>
                - adds a candidate spawn location

            GOAL <x> <y>
                - adds a candidate goal location

        Notes:
        - Coordinates are 0-indexed and use the same (x, y) convention as the env.
        - Lines starting with '#' or empty lines are ignored.
        """
        blocked: Set[Tuple[int, int]] = set()
        spawns: List[Tuple[int, int]] = []
        goals: List[Tuple[int, int]] = []

        with open(path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                key = parts[0].upper()

                def parse_ints(n: int) -> List[int]:
                    if len(parts) != n + 1:
                        raise ValueError(f"{path}:{lineno}: expected {n} ints after {parts[0]!r}, got: {parts[1:]}")
                    try:
                        return [int(v) for v in parts[1:]]
                    except ValueError as e:
                        raise ValueError(f"{path}:{lineno}: non-integer value in: {parts}") from e

                if key == "GRID":
                    h, w = parse_ints(2)
                    if h <= 1 or w <= 1:
                        raise ValueError(f"{path}:{lineno}: GRID dimensions must be > 1")
                    self.grid_h = h
                    self.grid_w = w

                elif key == "BLOCK":
                    x, y = parse_ints(2)
                    blocked.add((x, y))

                elif key == "BLOCK_RECT":
                    x1, y1, x2, y2 = parse_ints(4)
                    xa, xb = sorted((x1, x2))
                    ya, yb = sorted((y1, y2))
                    for x in range(xa, xb + 1):
                        for y in range(ya, yb + 1):
                            blocked.add((x, y))

                elif key == "SPAWN":
                    x, y = parse_ints(2)
                    spawns.append((x, y))

                elif key == "GOAL":
                    x, y = parse_ints(2)
                    goals.append((x, y))

                else:
                    raise ValueError(f"{path}:{lineno}: unknown directive {parts[0]!r}")

        self._validate_points(path, blocked, spawns, goals)

        self.blocked = set(blocked)
        self.spawn_points = self._dedup_points(spawns)
        self.goal_points = self._dedup_points(goals)
        self._refresh_blocked_grid()

    def _random_free_cell(self, occupied: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Random free (non-blocked) cell not in occupied."""
        for _ in range(2000):
            p = (random.randint(0, self.grid_w - 1), random.randint(0, self.grid_h - 1))
            if p not in self.blocked and p not in occupied:
                return p
        # fallback: brute force search
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                p = (x, y)
                if p not in self.blocked and p not in occupied:
                    return p
        raise RuntimeError("No free cells available (grid may be fully blocked/occupied).")

    # ============================================================
    # Spaces
    # ============================================================
    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def _build_observation_space(self):
        w = 2 * self.obs_radius + 1
        return spaces.Dict({
            "vector": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "window": spaces.Box(low=-1.0, high=1.0, shape=(w, w, 3), dtype=np.float32),
        })
    # ============================================================
    # Reset
    # ============================================================
    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.timestep = 0
        self.agents = self.possible_agents[:]

        # randomize headings
        for a in self.agents:
            self.agent_dir[a] = random.randint(0, 3)

        # place agents and their goals
        # Priority:
        #   - if spawn_points / goal_points are provided via map file, sample from those sets
        #   - otherwise fall back to uniform random sampling over free cells

        # -----------------------------
        # Choose spawn locations
        # -----------------------------
        occupied: Set[Tuple[int, int]] = set()

        if self.spawn_points:
            if len(self.spawn_points) < self.n_agents:
                raise ValueError(
                    f"Not enough SPAWN points for {self.n_agents} agents (have {len(self.spawn_points)})."
                )
            if self._using_domain_config:
                # Domain configs already provide explicit starts in agentFile.
                spawns = self.spawn_points[:self.n_agents]
            else:
                spawns = random.sample(self.spawn_points, self.n_agents)
        else:
            spawns = []
            for _ in range(self.n_agents):
                p = self._random_free_cell(occupied)
                spawns.append(p)
                occupied.add(p)

        # mark occupied by spawns
        occupied |= set(spawns)

        # -----------------------------
        # Choose goal locations
        # -----------------------------
        if self.goal_points:
            # avoid picking a goal that collides with any spawn
            candidates = [p for p in self.goal_points if p not in occupied]
            if len(candidates) < self.n_agents:
                raise ValueError(
                    f"Not enough GOAL points that are distinct from chosen spawns "
                    f"for {self.n_agents} agents (need {self.n_agents}, have {len(candidates)})."
                )
            if self._using_domain_config:
                # Domain taskFile defines the target pool; preserve file order.
                goals = candidates[:self.n_agents]
            else:
                goals = random.sample(candidates, self.n_agents)
        else:
            goals = []
            for _ in range(self.n_agents):
                p = self._random_free_cell(occupied)
                goals.append(p)
                occupied.add(p)

        # Assign per-agent
        for i, agent in enumerate(self.agents):
            self.goal_locations[agent] = goals[i]
            self.agent_location[agent] = spawns[i]

        return self._get_observations(), {a: {} for a in self.agents}


    # ============================================================
    # Step
    # ============================================================
    def step(self, actions):
        self.timestep += 1

        rewards = {agent: -0.001 for agent in self.agents}
        collision_penalty = -0.1

        # Save originals for collision checks
        orig_pos = {a: self.agent_location[a] for a in self.agents}

        # 1) Apply turns immediately; compute proposed moves for forward
        proposed_pos = {a: orig_pos[a] for a in self.agents}
        moved = {a: False for a in self.agents}

        for agent, action in actions.items():
            if agent not in self.agents:
                continue

            if action == 1:  # turn right
                self.agent_dir[agent] = (self.agent_dir[agent] + 1) % 4
            elif action == 2:  # turn left
                self.agent_dir[agent] = (self.agent_dir[agent] - 1) % 4
            elif action == 0:  # forward
                proposed_pos[agent] = self._forward(orig_pos[agent], self.agent_dir[agent])
                moved[agent] = True
            elif action == 3:  # wait
                pass

        # 2) Vertex collisions: same proposed cell => cancel all involved
        cell_to_agents = {}
        for a in self.agents:
            cell_to_agents.setdefault(proposed_pos[a], []).append(a)

        collided = set()
        for cell, agents_here in cell_to_agents.items():
            if len(agents_here) > 1:
                for a in agents_here:
                    proposed_pos[a] = orig_pos[a]
                    collided.add(a)

        # 3) Edge collisions: swaps => cancel both.
        # Build reverse-transition lookup to avoid O(A^2) pair scans.
        active_movers = [a for a in self.agents if moved[a] and a not in collided]
        transitions = {
            (orig_pos[a], proposed_pos[a]): a
            for a in active_movers
        }
        for a in active_movers:
            reverse = (proposed_pos[a], orig_pos[a])
            b = transitions.get(reverse)
            if b is not None and b != a:
                proposed_pos[a] = orig_pos[a]
                proposed_pos[b] = orig_pos[b]
                collided.add(a)
                collided.add(b)

        # Apply collision penalties
        for a in collided:
            rewards[a] += collision_penalty

        # 4) Commit positions
        for a in self.agents:
            self.agent_location[a] = proposed_pos[a]

        for a in self.agents:
            if self.agent_location[a] == self.goal_locations[a]:
                rewards[a] += 10.0
                self._sample_new_goal(a)

        # Only truncate by time (no "all goals reached" terminal condition now)
        truncated = self.timestep >= self.max_steps
        dones = {a: truncated for a in self.agents}
        truncs = {a: truncated for a in self.agents}

        if truncated:
            self.agents = []

        return (
            self._get_observations(),
            rewards,
            dones,
            truncs,
            {a: {} for a in dones},
        )

    def _sample_new_goal(self, agent):
        """
        Sample a new goal location for `agent` that doesn't overlap with:
        - any agent position
        - any other agent's goal position
        - any blocked cell

        If goal points were provided in a map file, sample from that set.
        Otherwise, sample uniformly at random over free cells.
        """
        occupied = set(self.agent_location[a] for a in self.possible_agents)  # agent cells
        occupied |= set(self.goal_locations[a] for a in self.possible_agents if a != agent)  # other goals
        occupied |= set(self.blocked)

        if self.goal_points:
            candidates = [p for p in self.goal_points if p not in occupied]
            if candidates:
                self.goal_locations[agent] = random.choice(candidates)
                return
            # fall through to random if candidates exhausted

        self.goal_locations[agent] = self._random_free_cell(occupied)
    # ============================================================
    # Movement
    # ============================================================
    def _forward(self, loc, heading):
        x, y = loc
        # 0=N,1=E,2=S,3=W
        if heading == 0:
            y = min(self.grid_h - 1, y + 1)
        elif heading == 1:
            x = min(self.grid_w - 1, x + 1)
        elif heading == 2:
            y = max(0, y - 1)
        elif heading == 3:
            x = max(0, x - 1)
            
        cand = (x, y)
        # blocked cells are impassable: treat as "bump into wall" (stay put)
        if cand in self.blocked:
            return loc
        return cand

    # ============================================================
    # Observation dispatcher
    # ============================================================
    def _get_observations(self):
        return {a: self._single_obs(a) for a in self.agents}

    def _single_obs(self, agent):
        return {
            "vector": self._obs_goal_vector(agent),
            "window": self._obs_window(agent),
        }

    # ============================================================
    # Goal-direction Vector Observation
    # ============================================================
    def _obs_goal_vector(self, agent):
        ax, ay = self.agent_location[agent]
        gx, gy = self.goal_locations[agent]
        dx = float(gx - ax)
        dy = float(gy - ay)
        norm = float(np.hypot(dx, dy))
        if norm > 0.0:
            dx /= norm
            dy /= norm
        else:
            dx = 0.0
            dy = 0.0
        return np.array([dx, dy], dtype=np.float32)

    # ============================================================
    # WINDOW Observation
    # ============================================================
    def _obs_window(self, agent):
        ax, ay = self.agent_location[agent]
        R = self.obs_radius
        W = 2 * R + 1

        obs = np.full((W, W, 3), -1.0, dtype=np.float32)

        # Fill in-bounds window area with vectorized blocked lookup.
        x0 = max(0, ax - R)
        x1 = min(self.grid_w, ax + R + 1)
        y0 = max(0, ay - R)
        y1 = min(self.grid_h, ay + R + 1)

        ox0 = x0 - (ax - R)
        ox1 = ox0 + (x1 - x0)
        oy0 = y0 - (ay - R)
        oy1 = oy0 + (y1 - y0)

        obs[ox0:ox1, oy0:oy1, :] = 0.0
        local_blocked = self._blocked_grid[y0:y1, x0:x1].T
        obs[ox0:ox1, oy0:oy1, 0] = np.where(local_blocked, -1.0, 0.0)

        # ego (channel 0 encodes heading at center cell)
        # heading: 0=N,1=E,2=S,3=W -> 0.25,0.5,0.75,1.0
        obs[R, R, 0] = (self.agent_dir[agent] + 1) / 4.0

        # other agents (channel 1)
        for other in self.agents:
            if other == agent:
                continue
            ox, oy = self.agent_location[other]
            dx, dy = ox - ax, oy - ay
            if -R <= dx <= R and -R <= dy <= R:
                obs[R + dx, R + dy, 1] = 1.0

        # own goal only (channel 2)
        gx, gy = self.goal_locations[agent]
        dx, dy = gx - ax, gy - ay
        if -R <= dx <= R and -R <= dy <= R:
            obs[R + dx, R + dy, 2] = 1.0

        return obs

    # ============================================================
    # Rendering
    # ============================================================
    def _agent_color(self, agent):
        """Deterministic per-agent color (RGB)."""
        # 10 distinct-ish colors; cycles if you have >10 agents
        palette = [
            (255, 99, 71),    # tomato
            (54, 162, 235),   # blue
            (255, 205, 86),   # yellow
            (75, 192, 192),   # teal
            (153, 102, 255),  # purple
            (255, 159, 64),   # orange
            (46, 204, 113),   # green
            (231, 76, 60),    # red
            (52, 73, 94),     # slate
            (241, 196, 15),   # gold
        ]
        idx = int(agent.split("_")[-1]) if "_" in agent else 0
        return palette[idx % len(palette)]

    def _heading_to_triangle(self, cx, cy, heading, size):
        """
        Return 3 points for a triangle arrow centered at (cx, cy),
        pointing along heading: 0=N,1=E,2=S,3=W.
        """
        # Tip points in heading direction; base is opposite.
        if heading == 0:  # N
            tip = (cx, cy - size)
            left = (cx - size * 0.6, cy + size * 0.7)
            right = (cx + size * 0.6, cy + size * 0.7)
        elif heading == 1:  # E
            tip = (cx + size, cy)
            left = (cx - size * 0.7, cy - size * 0.6)
            right = (cx - size * 0.7, cy + size * 0.6)
        elif heading == 2:  # S
            tip = (cx, cy + size)
            left = (cx - size * 0.6, cy - size * 0.7)
            right = (cx + size * 0.6, cy - size * 0.7)
        else:  # 3=W
            tip = (cx - size, cy)
            left = (cx + size * 0.7, cy - size * 0.6)
            right = (cx + size * 0.7, cy + size * 0.6)

        # pygame wants ints
        return [(int(tip[0]), int(tip[1])),
                (int(left[0]), int(left[1])),
                (int(right[0]), int(right[1]))]

    
    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode

        if mode not in ["human", "rgb_array"]:
            raise ValueError(f"Unsupported render mode: {mode}")

        self._init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame_initialized = False
                return None

        self._screen.blit(self._static_surface, (0, 0))

        # goals/targets: orange squares with agent number
        for agent in self.possible_agents:
            gx, gy = self.goal_locations[agent]
            sx, sy = self._grid_to_screen(gx, gy)
            goal_color = (255, 140, 0)  # orange

            pad = max(1, self._cell_size // 6)
            goal_rect = pygame.Rect(
                sx + pad,
                sy + pad,
                self._cell_size - 2 * pad,
                self._cell_size - 2 * pad,
            )
            pygame.draw.rect(self._screen, goal_color, goal_rect)
            pygame.draw.rect(self._screen, (20, 20, 20), goal_rect, 2)

            agent_num = agent.split("_")[-1] if "_" in agent else "?"
            goal_label = self._get_cached_label(agent_num, (160, 160, 160))
            gl_rect = goal_label.get_rect(center=(goal_rect.centerx, goal_rect.centery))
            self._screen.blit(goal_label, gl_rect)

        # agents: blue circles with heading dot and number
        for agent in self.agents:
            ax, ay = self.agent_location[agent]
            sx, sy = self._grid_to_screen(ax, ay)
            cx = sx + self._cell_size // 2
            cy = sy + self._cell_size // 2

            body_r = max(3, self._cell_size // 3)
            body_color = (40, 120, 255)  # blue
            pygame.draw.circle(self._screen, body_color, (cx, cy), body_r)
            pygame.draw.circle(self._screen, (10, 10, 10), (cx, cy), body_r, 2)

            # Small facing-direction dot.
            heading = self.agent_dir[agent]
            d = max(2, int(body_r * 0.6))
            if heading == 0:  # N
                dot_x, dot_y = cx, cy - d
            elif heading == 1:  # E
                dot_x, dot_y = cx + d, cy
            elif heading == 2:  # S
                dot_x, dot_y = cx, cy + d
            else:  # W
                dot_x, dot_y = cx - d, cy
            pygame.draw.circle(self._screen, (255, 255, 255), (dot_x, dot_y), max(2, body_r // 4))
            pygame.draw.circle(self._screen, (10, 10, 10), (dot_x, dot_y), max(2, body_r // 4), 1)

            # Agent number label.
            agent_num = agent.split("_")[-1] if "_" in agent else "?"
            agent_label = self._get_cached_label(agent_num, (255, 255, 255))
            al_rect = agent_label.get_rect(center=(cx, cy))
            self._screen.blit(agent_label, al_rect)

        if mode == "human":
            pygame.display.flip()
            self._clock.tick(10)
            return None
        else:
            pygame.display.flip()
            return self._get_frame()


    def _get_frame(self):
        data = pygame.surfarray.array3d(self._screen)
        return np.transpose(data, (1, 0, 2))

    def _init_pygame(self):
        if self._pygame_initialized:
            return
        pygame.init()

        width = self.grid_w * self._cell_size + 2 * self._margin
        height = self.grid_h * self._cell_size + 2 * self._margin
        # If still too large for some reason, shrink further (safety loop)
        while (width > self._max_window_w or height > self._max_window_h) and self._cell_size > self._min_cell_size:
            self._cell_size -= 1
            self._margin = int(max(8, min(40, self._cell_size * 0.6)))
            width = self.grid_w * self._cell_size + 2 * self._margin
            height = self.grid_h * self._cell_size + 2 * self._margin
            
        self._screen = pygame.display.set_mode((int(width), int(height)))
        self._clock = pygame.time.Clock()
        
        font_size = int(max(12, min(24, self._cell_size * 0.6)))
        self._font = pygame.font.SysFont("consolas", font_size)
        self._static_surface = pygame.Surface((int(width), int(height)))
        self._build_static_surface()
        self._label_cache = {}
        self._pygame_initialized = True

    def _build_static_surface(self):
        self._static_surface.fill((30, 30, 30))

        # Draw static grid once.
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                sx, sy = self._grid_to_screen(x, y)
                pygame.draw.rect(
                    self._static_surface,
                    (60, 60, 60),
                    pygame.Rect(sx, sy, self._cell_size, self._cell_size),
                    1,
                )

        # Draw static blocked cells once.
        for (bx, by) in self.blocked:
            sx, sy = self._grid_to_screen(bx, by)
            rect = pygame.Rect(sx, sy, self._cell_size, self._cell_size)
            pygame.draw.rect(self._static_surface, (20, 20, 20), rect)
            pygame.draw.rect(self._static_surface, (110, 110, 110), rect, 2)

    def _get_cached_label(self, text: str, color: Tuple[int, int, int]) -> pygame.Surface:
        key = (text, color)
        surface = self._label_cache.get(key)
        if surface is None:
            surface = self._font.render(text, True, color)
            self._label_cache[key] = surface
        return surface

    def _grid_to_screen(self, x, y):
        return (
            self._margin + x * self._cell_size,
            self._margin + (self.grid_h - 1 - y) * self._cell_size,
        )
        
    def _compute_render_scale(self):
        """
        Choose cell size + margin so the grid fits on screen.

        Uses self.grid_w, self.grid_h and clamps cell size into
        [self._min_cell_size, self._max_cell_size].
        """
        # leave room for margin on both sides
        avail_w = max(50, self._max_window_w - 2 * self._margin)
        avail_h = max(50, self._max_window_h - 2 * self._margin)

        # compute cell size that fits (floor)
        cell_w = avail_w // self.grid_w
        cell_h = avail_h // self.grid_h
        cell = int(min(cell_w, cell_h))

        # clamp
        cell = max(self._min_cell_size, min(self._max_cell_size, cell))

        # margin scales a bit with cell size (but keep reasonable)
        margin = int(max(8, min(40, cell * 0.6)))

        return cell, margin

    def close(self):
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False
            self._static_surface = None
            self._label_cache = {}


if __name__ == "__main__":
    env = MAPF(obs_mode="hybrid", map_path="maps/random.domain/random_32_32_20_10.json")
    obs, info = env.reset()
    done = {a: False for a in env.agents}

    while env.agents and not all(done.values()):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, done, trunc, info = env.step(actions)
        env.render()

    env.close()
