from typing import Dict, Optional, List, Tuple, Set
import random
import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import pygame

from map_loader import load_map_configuration


class MAPF(ParallelEnv):
    """
    Unified Multi-Agent Pathfinding Environment.

    Observation mode:
        - "hybrid": local window (3 channels) plus a 2D goal-direction vector

    Task modes:
        - lifelong=True (default): Agents receive new goals upon reaching current ones
        - lifelong=False: Traditional MAPF; agents are done when reaching their goal

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
        blocked_cells: Optional[Set[Tuple[int, int]]] = None,
        lifelong=True,       # True: generate new goal on reaching current one (lifelong)
                             # False: agent done when reaching goal (traditional MAPF)
    ):

        if obs_mode != "hybrid":
            raise ValueError("This environment now only supports obs_mode='hybrid'.")
        self.obs_mode = obs_mode
        self.lifelong = lifelong
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
            loaded = load_map_configuration(
                path=map_path,
                requested_agents=self.n_agents,
                default_grid_shape=(self.grid_h, self.grid_w),
            )
            self.grid_w = loaded.grid_w
            self.grid_h = loaded.grid_h
            self.blocked = set(loaded.blocked)
            self.spawn_points = list(loaded.spawn_points)
            self.goal_points = list(loaded.goal_points)
            self._using_domain_config = loaded.using_domain_config

            self.team_size = loaded.team_size
            self.num_tasks_reveal = loaded.num_tasks_reveal
            self.agent_size = loaded.agent_size
            self.max_counter = loaded.max_counter
            self.delay_config = loaded.delay_config

            if num_agents is None and loaded.team_size is not None:
                self.n_agents = loaded.team_size
        elif blocked_cells is not None:
            self.blocked = set(blocked_cells)
            self._validate_points("blocked_cells", self.blocked, [], [])
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
    # Map validation
    # ============================================================

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

        # If we must sample both spawns and goals from generic free cells,
        # ensure capacity exists up front to avoid opaque "No free cells" errors.
        if not self.spawn_points and not self.goal_points:
            total_free_cells = (self.grid_w * self.grid_h) - len(self.blocked)
            required_unique_cells = 2 * self.n_agents
            if total_free_cells < required_unique_cells:
                raise ValueError(
                    "Not enough free cells to sample unique spawn and goal locations "
                    f"for {self.n_agents} agents. "
                    f"Need at least {required_unique_cells} free cells, "
                    f"but map has {total_free_cells}."
                )

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
        wall_bump_penalty = -0.01
        shaping_scale = 0.05

        # Save originals for collision checks
        orig_pos = {a: self.agent_location[a] for a in self.agents}
        prev_goal_dist = {
            a: abs(self.goal_locations[a][0] - orig_pos[a][0]) + abs(self.goal_locations[a][1] - orig_pos[a][1])
            for a in self.agents
        }

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
                if proposed_pos[agent] == orig_pos[agent]:
                    rewards[agent] += wall_bump_penalty
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
            new_pos = self.agent_location[a]
            new_goal_dist = abs(self.goal_locations[a][0] - new_pos[0]) + abs(self.goal_locations[a][1] - new_pos[1])
            dist_delta = prev_goal_dist[a] - new_goal_dist
            # Only reward if agent moved toward goal (dist decreased)
            if dist_delta > 0:
                rewards[a] += shaping_scale * float(dist_delta)

        # Track which agents reach their goal in this step (for traditional MAPF mode)
        agent_done_early = {a: False for a in self.agents}

        for a in self.agents:
            if self.agent_location[a] == self.goal_locations[a]:
                rewards[a] += 10.0
                if self.lifelong:
                    self._sample_new_goal(a)
                else:
                    # Traditional MAPF: mark agent as done when goal is reached
                    agent_done_early[a] = True

        # Only truncate by time (no "all goals reached" terminal condition now)
        truncated = self.timestep >= self.max_steps
        dones = {a: (truncated or agent_done_early.get(a, False)) for a in self.agents}
        truncs = {a: truncated for a in self.agents}

        if truncated:
            self.agents = []
        else:
            # Remove agents that are done (reached goal in traditional MAPF mode)
            self.agents = [a for a in self.agents if not dones.get(a, False)]

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
            if self._using_domain_config:
                raise ValueError(
                    "No valid goal candidates remain in task-defined goal locations. "
                    "Refusing to spawn goals outside allowed task areas."
                )
            # fall through to random if candidates exhausted (non-benchmark maps)

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
        """
        Goal direction vector relative to the agent's heading.
        
        Returns a normalized 2D vector [forward_component, right_component] where:
        - forward_component: how much the goal is in front of the agent (1 = directly ahead)
        - right_component: how much the goal is to the right of the agent (1 = directly right)
        
        Headings:
            0 = North (+y)
            1 = East  (+x)
            2 = South (-y)
            3 = West  (-x)
        """
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
        
        # Transform global vector (dx, dy) to agent-relative frame based on heading
        heading = self.agent_dir[agent]
        if heading == 0:  # North: forward=(0,1), right=(1,0)
            local_forward = dy
            local_right = dx
        elif heading == 1:  # East: forward=(1,0), right=(0,-1)
            local_forward = dx
            local_right = -dy
        elif heading == 2:  # South: forward=(0,-1), right=(-1,0)
            local_forward = -dy
            local_right = -dx
        else:  # heading == 3, West: forward=(-1,0), right=(0,1)
            local_forward = -dx
            local_right = dy
        
        return np.array([local_forward, local_right], dtype=np.float32)

    # ============================================================
    # WINDOW Observation
    # ============================================================
    def _obs_window(self, agent):
        ax, ay = self.agent_location[agent]
        heading = self.agent_dir[agent]
        R = self.obs_radius
        W = 2 * R + 1

        obs = np.full((W, W, 3), -1.0, dtype=np.float32)

        # Build a heading-aligned window: local +y is always "forward",
        # local +x is always "right" for the observing agent.
        def local_to_global_offset(local_right: int, local_forward: int, h: int) -> Tuple[int, int]:
            if h == 0:  # North
                return local_right, local_forward
            if h == 1:  # East
                return local_forward, -local_right
            if h == 2:  # South
                return -local_right, -local_forward
            # h == 3, West
            return -local_forward, local_right

        def global_to_local_offset(dx: int, dy: int, h: int) -> Tuple[int, int]:
            if h == 0:  # North
                return dx, dy
            if h == 1:  # East
                return -dy, dx
            if h == 2:  # South
                return -dx, -dy
            # h == 3, West
            return dy, -dx

        # Channel 0 is traversability in the rotated local frame.
        for local_right in range(-R, R + 1):
            for local_forward in range(-R, R + 1):
                dx, dy = local_to_global_offset(local_right, local_forward, heading)
                gx, gy = ax + dx, ay + dy
                ix, iy = R + local_right, R + local_forward

                if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                    obs[ix, iy, :] = 0.0
                    if self._blocked_grid[gy, gx]:
                        obs[ix, iy, 0] = -1.0

        # Heading is now encoded by rotating the whole window; use a fixed
        # ego-center marker instead of an absolute heading scalar.
        obs[R, R, 0] = 1.0

        # channel 1 shared encoding:
        # 0.0 = empty
        # 0.5 = other agent goal
        # 1.0 = other agent position
        # 1.5 = both other agent and other goal overlap

        # other agents' goals first (channel 1 -> 0.5)
        for other in self.agents:
            if other == agent:
                continue
            gx_o, gy_o = self.goal_locations[other]
            dx, dy = gx_o - ax, gy_o - ay
            local_right, local_forward = global_to_local_offset(dx, dy, heading)
            if -R <= local_right <= R and -R <= local_forward <= R:
                ix = R + local_right
                iy = R + local_forward
                obs[ix, iy, 1] = max(obs[ix, iy, 1], 0.5)

        # other agents (channel 1 -> 1.0, or 1.5 if overlapping with goal)
        for other in self.agents:
            if other == agent:
                continue
            ox, oy = self.agent_location[other]
            dx, dy = ox - ax, oy - ay
            local_right, local_forward = global_to_local_offset(dx, dy, heading)
            if -R <= local_right <= R and -R <= local_forward <= R:
                ix = R + local_right
                iy = R + local_forward
                if obs[ix, iy, 1] >= 0.5:
                    obs[ix, iy, 1] = 1.5
                else:
                    obs[ix, iy, 1] = 1.0

        # own goal only (channel 2)
        gx, gy = self.goal_locations[agent]
        dx, dy = gx - ax, gy - ay
        local_right, local_forward = global_to_local_offset(dx, dy, heading)
        if -R <= local_right <= R and -R <= local_forward <= R:
            obs[R + local_right, R + local_forward, 2] = 1.0

        return obs

    # ============================================================
    # Rendering
    # ============================================================
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
        # Render goals only for active agents so completed/removed agents do not leave stale targets.
        for agent in self.agents:
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
