from typing import Dict
import random
import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import pygame


class MAPF(ParallelEnv):
    """
    Unified Multi-Agent Pathfinding Environment.

    Observation modes:
        - "vector": flat structured vector (global coordinates)
        - "window": local window (image-like) with egocentric heading encoded in channel 0 at center cell
        - "knn":    K-nearest neighbor encoded vector

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
        grid_size=7,
        num_agents=2,
        obs_mode="vector",    # "vector", "window", or "knn"
        obs_radius=3,         # used only for window mode
        k_agents=2,           # used only for knn mode
    ):

        assert obs_mode in ("vector", "window", "knn")
        self.obs_mode = obs_mode

        self.grid_size = grid_size
        self.n_agents = num_agents
        self.obs_radius = obs_radius
        self.k_agents = k_agents
        self.max_steps = 200
        self.timestep = 0

        # Agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
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
        self._cell_size = 64
        self._margin = 40
        self._screen = None
        self._clock = None
        self._font = None

    # ============================================================
    # Spaces
    # ============================================================
    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def _build_observation_space(self):
        if self.obs_mode == "vector":
            # heading (1) + ego (2) + other agents (2 each) + own goal (2)
            size = 1 + 2 + (self.n_agents - 1) * 2 + 2
            return spaces.Box(low=-1, high=np.inf, shape=(size,), dtype=np.float32)

        elif self.obs_mode == "window":
            w = 2 * self.obs_radius + 1
            return spaces.Box(low=-1, high=1, shape=(w, w, 3), dtype=np.float32)

        elif self.obs_mode == "knn":
            # heading (1) + k_agents * 3 + own goal (3)
            size = 1 + (self.k_agents ) * 3 + 3
            return spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)

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

        # randomly place agents and their goals (distinct positions)
        needed = self.n_agents * 2
        positions = set()
        while len(positions) < needed:
            positions.add(
                (random.randint(0, self.grid_size - 1),
                 random.randint(0, self.grid_size - 1))
            )

        positions = list(positions)
        idx = 0

        # Goals (one per agent)
        for agent in self.agents:
            self.goal_locations[agent] = positions[idx]
            idx += 1

        # Agents
        for agent in self.agents:
            self.agent_location[agent] = positions[idx]
            idx += 1

        return self._get_observations(), {a: {} for a in self.agents}

    # ============================================================
    # Step
    # ============================================================
    def step(self, actions):
        self.timestep += 1

        rewards = {agent: -0.01 for agent in self.agents}
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

        # 3) Edge collisions: swaps => cancel both
        # Only consider agents that actually moved and weren't already vertex-canceled.
        active_movers = [a for a in self.agents if moved[a] and a not in collided]

        for i in range(len(active_movers)):
            a = active_movers[i]
            for j in range(i + 1, len(active_movers)):
                b = active_movers[j]
                if proposed_pos[a] == orig_pos[b] and proposed_pos[b] == orig_pos[a]:
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
                rewards[a] += 1.0

        # Termination: everyone on their own goal
        done = all(self.agent_location[a] == self.goal_locations[a] for a in self.agents)
        truncated = self.timestep >= self.max_steps

        dones = {a: done or truncated for a in self.agents}
        truncs = {a: truncated for a in self.agents}

        if done or truncated:
            self.agents = []

        return (
            self._get_observations(),
            rewards,
            dones,
            truncs,
            {a: {} for a in dones},
        )

    # ============================================================
    # Movement
    # ============================================================
    def _forward(self, loc, heading):
        x, y = loc
        # 0=N,1=E,2=S,3=W
        if heading == 0:
            y = min(self.grid_size - 1, y + 1)
        elif heading == 1:
            x = min(self.grid_size - 1, x + 1)
        elif heading == 2:
            y = max(0, y - 1)
        elif heading == 3:
            x = max(0, x - 1)
        return (x, y)

    # ============================================================
    # Observation dispatcher
    # ============================================================
    def _get_observations(self):
        return {a: self._single_obs(a) for a in self.agents}

    def _single_obs(self, agent):
        if self.obs_mode == "vector":
            return self._obs_vector(agent)
        elif self.obs_mode == "window":
            return self._obs_window(agent)
        elif self.obs_mode == "knn":
            return self._obs_knn(agent)

    # ============================================================
    # Vector Observation
    # ============================================================
    def _obs_vector(self, agent):
        h = float(self.agent_dir[agent])
        ax, ay = self.agent_location[agent]
        obs = [h, ax, ay]

        # others
        for other in self.agents:
            if other == agent:
                continue
            ox, oy = self.agent_location[other]
            obs.extend([ox, oy])

        # own goal only
        gx, gy = self.goal_locations[agent]
        obs.extend([gx, gy])

        return np.array(obs, dtype=np.float32)

    # ============================================================
    # WINDOW Observation
    # ============================================================
    def _obs_window(self, agent):
        ax, ay = self.agent_location[agent]
        R = self.obs_radius
        W = 2 * R + 1

        obs = np.full((W, W, 3), -1.0, dtype=np.float32)

        def in_bounds(x, y):
            return 0 <= x < self.grid_size and 0 <= y < self.grid_size

        # empty cells
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                wx, wy = ax + dx, ay + dy
                if in_bounds(wx, wy):
                    obs[R + dx, R + dy, :] = 0.0

        # ego (channel 0 encodes heading at center cell)
        # heading: 0=N,1=E,2=S,3=W -> 0.25,0.5,0.75,1.0
        obs[R, R, 0] = (self.agent_dir[agent] + 1) / 4.0

        # other agents (channel 1)
        for other in self.agents:
            if other == agent:
                continue
            ox, oy = self.agent_location[other]
            dx, dy = ox - ax, oy - ay
            if -R <= dx <= R and -R <= dy <= R and in_bounds(ox, oy):
                obs[R + dx, R + dy, 1] = 1.0

        # own goal only (channel 2)
        gx, gy = self.goal_locations[agent]
        dx, dy = gx - ax, gy - ay
        if -R <= dx <= R and -R <= dy <= R and in_bounds(gx, gy):
            obs[R + dx, R + dy, 2] = 1.0

        return obs

    # ============================================================
    # KNN Observation
    # ============================================================
    def _obs_knn(self, agent):
        ax, ay = self.agent_location[agent]
        result = [float(self.agent_dir[agent])]

        # agents
        others = []
        for other in self.agents:
            if other == agent:
                continue
            ox, oy = self.agent_location[other]
            dx, dy = ox - ax, oy - ay
            dist = abs(dx) + abs(dy)
            others.append((dist, dx, dy))
        others.sort(key=lambda x: x[0])

        # nearest agents
        for i in range(self.k_agents):
            if i < len(others):
                _, dx, dy = others[i]
                result.extend([dx, dy, 1.0])
            else:
                result.extend([0.0, 0.0, 0.0])

        # "items" block becomes own goal (1)
        gx, gy = self.goal_locations[agent]
        gdx, gdy = gx - ax, gy - ay
        result.extend([gdx, gdy, 1.0])


        return np.array(result, dtype=np.float32)

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

        self._screen.fill((30, 30, 30))

        # grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                sx, sy = self._grid_to_screen(x, y)
                pygame.draw.rect(
                    self._screen,
                    (60, 60, 60),
                    pygame.Rect(sx, sy, self._cell_size, self._cell_size),
                    1,
                )

        # goals (each goal matches its agent's color)
        for agent in self.possible_agents:
            gx, gy = self.goal_locations[agent]
            sx, sy = self._grid_to_screen(gx, gy)
            color = self._agent_color(agent)
            pygame.draw.circle(
                self._screen,
                color,
                (sx + self._cell_size // 2, sy + self._cell_size // 2),
                self._cell_size // 4,
            )
            # optional outline for readability
            pygame.draw.circle(
                self._screen,
                (10, 10, 10),
                (sx + self._cell_size // 2, sy + self._cell_size // 2),
                self._cell_size // 4,
                2,
            )

        # agents (arrows pointing in heading, using same per-agent color)
        for agent in self.agents:
            ax, ay = self.agent_location[agent]
            sx, sy = self._grid_to_screen(ax, ay)
            cx = sx + self._cell_size / 2
            cy = sy + self._cell_size / 2

            color = self._agent_color(agent)
            heading = self.agent_dir[agent]

            tri = self._heading_to_triangle(cx, cy, heading, size=self._cell_size * 0.33)
            pygame.draw.polygon(self._screen, color, tri)
            pygame.draw.polygon(self._screen, (10, 10, 10), tri, 2)  # outline

        text = self._font.render(f"t={self.timestep}", True, (255, 255, 255))
        self._screen.blit(text, (10, 10))

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

        width = self.grid_size * self._cell_size + 2 * self._margin
        height = self.grid_size * self._cell_size + 2 * self._margin
        self._screen = pygame.display.set_mode((width, height))

        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 18)
        self._pygame_initialized = True

    def _grid_to_screen(self, x, y):
        return (
            self._margin + x * self._cell_size,
            self._margin + (self.grid_size - 1 - y) * self._cell_size,
        )

    def close(self):
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False


if __name__ == "__main__":
    env = MAPF(grid_size=10, num_agents=4, obs_mode="vector")
    obs, info = env.reset()
    done = {a: False for a in env.agents}

    while env.agents and not all(done.values()):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, done, trunc, info = env.step(actions)
        env.render()

    env.close()
