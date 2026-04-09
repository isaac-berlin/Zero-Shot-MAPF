import random
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from itertools import combinations

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from MAPF import MAPF


TRAIN_GRID_SHAPE = (16, 16)
TRAIN_NUM_AGENTS = 10
TRAIN_OBS_RADIUS = 5
TRAIN_SCENARIOS = ("warehouse", "random")
TRAIN_RANDOM_ENV_POOL_SIZE = 1000
TRAIN_ENV_SAMPLE_EVERY_EPISODES = 5
TRAIN_TB_LOG_EVERY_EPISODES = 5
WAREHOUSE_MAP_PATH = Path("maps") / "warehouse16.domain" / "maps" / "warehouse_16x16.map"
TRAIN_RANDOM_ENV_DIR = Path("generated_random_envs")


# ============================================================
# Utils
# ============================================================

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


def _sample_obstacle_density(density_range) -> float:
    low, high = density_range
    if low > high:
        raise ValueError("obstacle density range must be ordered as (low, high).")

    mean = (low + high) / 2.0
    std = max((high - low) / 6.0, 1e-3)
    density = np.random.normal(loc=mean, scale=std)
    return float(np.clip(density, low, high))


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
    obstacle_density_range,
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


# ============================================================
# Actor / Critic Networks
# ============================================================
class ActorHybrid(nn.Module):
    def __init__(self, obs_spec, n_actions, hidden=128):
        super().__init__()
        H, W, C = obs_spec["window"]
        vec_dim = obs_spec["vector"][0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            x = torch.zeros(1, C, H, W)
            conv_out = self.cnn(x).view(1, -1).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.LayerNorm(conv_out),
            nn.Linear(conv_out, hidden),
            nn.Tanh(),
        )
        
        self.knn_fc = nn.Sequential(
            nn.LayerNorm(vec_dim),
            nn.Linear(vec_dim, hidden),
            nn.Tanh(),
        )
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        
    def forward(self, obs):
        # obs is a dict with "vector" and "window"
        vec_obs = obs["vector"]
        win_obs = obs["window"]

        # process window through CNN
        win_x = win_obs.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        win_feat = self.cnn(win_x)
        win_feat = win_feat.reshape(win_feat.size(0), -1)
        win_out = self.cnn_fc(win_feat)

        # process vector through MLP
        knn_out = self.knn_fc(vec_obs)

        # fuse and output action logits
        fusion_input = torch.cat([win_out, knn_out], dim=-1)
        return self.fusion_fc(fusion_input)

class ActorMLP(nn.Module):
    """Legacy actor kept only for compatibility with older checkpoints."""
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class ActorCNN(nn.Module):
    """Used for obs_mode = window (CNN)."""
    def __init__(self, obs_shape, n_actions, hidden=128):
        super().__init__()
        H, W, C = obs_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        # compute conv output dimension
        with torch.no_grad():
            x = torch.zeros(1, C, H, W)
            conv_out = self.cnn(x).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.LayerNorm(conv_out),
            nn.Linear(conv_out, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        # obs: (B, H, W, C) → (B, C, H, W)
        obs = obs.permute(0, 3, 1, 2)
        x = self.cnn(obs)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class CentralCritic(nn.Module):
    """Central critic shared across all agents."""
    def __init__(self, state_shape, hidden=64):
        super().__init__()
        if len(state_shape) != 3:
            raise ValueError("CentralCritic expects state_shape=(C, H, W).")

        channels, height, width = state_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.net = nn.Sequential(
            nn.LayerNorm(16),
            nn.Linear(16, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        if state.dim() == 3:
            state = state.unsqueeze(0)

        x = self.cnn(state)
        x = x.reshape(x.size(0), -1)
        return self.net(x).squeeze(-1)


# ============================================================
# Rollout Buffer
# ============================================================

@dataclass
class Transition:
    obs: object
    state: np.ndarray
    action: int
    logp: float
    value: float
    reward: float
    done: bool


class RolloutBuffer:
    def __init__(self, agent_order):
        self.agent_order = agent_order
        self.storage = {a: [] for a in agent_order}

    def add(self, agent, tr):
        self.storage[agent].append(tr)

    def clear(self):
        for a in self.agent_order:
            self.storage[a].clear()

    def compute_gae(self, gamma, lam, last_values):
        self.advantages, self.returns = {}, {}

        for a in self.agent_order:
            traj = self.storage[a]
            T = len(traj)

            adv = np.zeros(T, np.float32)

            next_adv = 0.0
            next_value = last_values[a]

            for t in reversed(range(T)):
                done = traj[t].done
                mask = 0 if done else 1
                delta = traj[t].reward + gamma * next_value * mask - traj[t].value
                next_adv = delta + gamma * lam * mask * next_adv
                adv[t] = next_adv
                next_value = traj[t].value

            ret = adv + np.array([tr.value for tr in traj], np.float32)
            self.advantages[a] = adv
            self.returns[a] = ret

    def get_flat_batches(self):
        first_agent = self.agent_order[0]
        first_traj = self.storage[first_agent]
        if not first_traj:
            raise ValueError("Rollout buffer is empty.")

        obs_vec_list, obs_win_list = [], []

        state_list, act_list = [], []
        logp_list, val_list, adv_list, ret_list = [], [], [], []

        for a in self.agent_order:
            traj = self.storage[a]
            for tr in traj:
                obs_vec_list.append(tr.obs["vector"])
                obs_win_list.append(tr.obs["window"])
                state_list.append(tr.state)
                act_list.append(tr.action)
                logp_list.append(tr.logp)
                val_list.append(tr.value)
            adv_list.append(self.advantages[a])
            ret_list.append(self.returns[a])

        obs = {
            "vector": np.asarray(obs_vec_list, dtype=np.float32),
            "window": np.asarray(obs_win_list, dtype=np.float32),
        }

        state = np.asarray(state_list, dtype=np.float32)
        acts = np.asarray(act_list, dtype=np.int64)
        logps = np.asarray(logp_list, dtype=np.float32)
        vals = np.asarray(val_list, dtype=np.float32)
        advs = np.concatenate(adv_list)
        rets = np.concatenate(ret_list)

        return obs, state, acts, logps, vals, advs, rets


# ============================================================
# MAPPO Algorithm
# ============================================================

class MAPPO:
    def __init__(self, obs_spec, n_actions, state_shape, num_agents, obs_mode, device="cpu"):
        self.device = device
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.num_agents = num_agents
        self.mode = obs_mode

        if obs_mode != "hybrid":
            raise ValueError("This trainer now only supports obs_mode='hybrid'.")
        self.actor = ActorHybrid(obs_spec, n_actions).to(device)

        self.critic = CentralCritic(state_shape).to(device)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.clip_eps = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5

    @torch.no_grad()
    def act(self, obs, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs_t = {
            "vector": torch.tensor(obs["vector"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "window": torch.tensor(obs["window"], dtype=torch.float32, device=self.device).unsqueeze(0),
        }

        logits = self.actor(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        value = self.critic(state_t)
        return int(action.item()), float(logp.item()), float(value.item())

    @torch.no_grad()
    def act_batch(self, obs_dict, state, agent_order):
        """
        Batched action/value inference for all agents at once.
        Returns three dicts keyed by agent: actions, logps, values.
        """
        n = len(agent_order)
        state_batch = torch.tensor(
            np.repeat(state[None, :], n, axis=0),
            dtype=torch.float32,
            device=self.device,
        )

        obs_t = {
            "vector": torch.tensor(
                np.stack([obs_dict[a]["vector"] for a in agent_order]),
                dtype=torch.float32,
                device=self.device,
            ),
            "window": torch.tensor(
                np.stack([obs_dict[a]["window"] for a in agent_order]),
                dtype=torch.float32,
                device=self.device,
            ),
        }

        logits = self.actor(obs_t)
        dist = Categorical(logits=logits)
        actions_t = dist.sample()
        logps_t = dist.log_prob(actions_t)
        values_t = self.critic(state_batch)

        actions = {a: int(actions_t[i].item()) for i, a in enumerate(agent_order)}
        logps = {a: float(logps_t[i].item()) for i, a in enumerate(agent_order)}
        values = {a: float(values_t[i].item()) for i, a in enumerate(agent_order)}
        return actions, logps, values

    def update(self, buffer, epochs, minibatch, writer, global_step):
        obs, state, acts, old_logps, old_vals, advs, rets = buffer.get_flat_batches()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        obs_t = {
            "vector": torch.tensor(obs["vector"], dtype=torch.float32, device=self.device),
            "window": torch.tensor(obs["window"], dtype=torch.float32, device=self.device),
        }
        N = obs_t["vector"].shape[0]
            
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        acts_t = torch.tensor(acts, dtype=torch.int64, device=self.device)
        old_logps_t = torch.tensor(old_logps, dtype=torch.float32, device=self.device)
        advs_t = torch.tensor(advs, dtype=torch.float32, device=self.device)
        rets_t = torch.tensor(rets, dtype=torch.float32, device=self.device)


        idxs = np.arange(N)

        mean_policy_loss = 0.0
        mean_value_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        count = 0

        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, minibatch):
                mb = idxs[start:start + minibatch]

                if self.mode == "hybrid":
                    mb_obs = {
                        "vector": obs_t["vector"][mb],
                        "window": obs_t["window"][mb],
                    }
                else:
                    mb_obs = obs_t[mb]
                mb_state = state_t[mb]
                mb_acts = acts_t[mb]
                mb_old_logps = old_logps_t[mb]
                mb_advs = advs_t[mb]
                mb_rets = rets_t[mb]

                # Actor update
                logits = self.actor(mb_obs)
                dist = Categorical(logits=logits)
                new_logps = dist.log_prob(mb_acts)
                entropy = dist.entropy().mean()

                kl = (mb_old_logps - new_logps).mean()
                mean_kl += kl.item()

                ratio = torch.exp(new_logps - mb_old_logps)
                unclipped = ratio * mb_advs
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advs

                policy_loss = -torch.min(unclipped, clipped).mean()
                actor_loss = policy_loss - self.ent_coef * entropy

                self.opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                # Critic update
                values = self.critic(mb_state)
                vf_loss = (mb_rets - values).pow(2).mean()

                critic_loss = self.vf_coef * vf_loss
                self.opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()

                mean_policy_loss += policy_loss.item()
                mean_value_loss += vf_loss.item()
                mean_entropy += entropy.item()
                count += 1

        return (
            mean_policy_loss / max(count, 1),
            mean_value_loss / max(count, 1),
            mean_entropy / max(count, 1),
            mean_kl / max(count, 1),
        )


# ============================================================
# Unified Training Loop
# ============================================================

def train_mappo(
    total_episodes=3000,
    rollout_len=128,
    update_epochs=4,
    minibatch_size=256,
    gamma=0.99,
    lam=0.95,
    device="cpu",
    enable_timing=True,
    timing_every_episodes=10,
    num_agents=TRAIN_NUM_AGENTS,
    grid_shape=TRAIN_GRID_SHAPE,
    obs_radius=TRAIN_OBS_RADIUS,
    obstacle_density_range=(0.0, 0.5),
    random_env_dir=TRAIN_RANDOM_ENV_DIR,
):
    random_layout_pool = build_random_layout_pool_from_dir(
        Path(random_env_dir),
        TRAIN_RANDOM_ENV_POOL_SIZE,
    )

    prototype_env, _ = build_episode_env(
        scenario="warehouse",
        obs_radius=obs_radius,
        num_agents=num_agents,
        grid_shape=grid_shape,
        obstacle_density_range=obstacle_density_range,
    )

    agent_order = prototype_env.possible_agents[:]

    dummy_obs, _ = prototype_env.reset()
    sample_obs = dummy_obs[agent_order[0]]

    obs_spec = {
        "vector": sample_obs["vector"].shape,
        "window": sample_obs["window"].shape,
    }

    state_shape = stack_global_state(prototype_env).shape
    n_actions = prototype_env.action_space(agent_order[0]).n  # should be 4 now

    prototype_env.close()

    algo = MAPPO(
        obs_spec=obs_spec,
        n_actions=n_actions,
        state_shape=state_shape,
        num_agents=num_agents,
        obs_mode="hybrid",
        device=device,
    )

    buffer = RolloutBuffer(agent_order)
    writer = SummaryWriter(log_dir=f"runs/mapf_hybrid_{grid_shape[0]}x{grid_shape[1]}_{num_agents}agents_mix")

    episode = 0
    step_count = 0
    env = None
    obs = None
    ep_return = 0.0
    ep_len = 0
    items_collected_ep = 0
    current_scenario = None
    current_episode_meta = {}
    visited = set()
    agent_prev_pos = {}
    agent_dist = {}
    action_freq = {0: 0, 1: 0, 2: 0, 3: 0}
    collisions_ep = 0
    episodes_until_resample = 0

    def start_episode_env(force_resample=False):
        scenario = random.choice(TRAIN_SCENARIOS)
        layout = None
        if scenario == "random":
            layout = random.choice(random_layout_pool)
        episode_env, episode_meta = build_episode_env(
            scenario=scenario,
            obs_radius=obs_radius,
            num_agents=num_agents,
            grid_shape=grid_shape,
            obstacle_density_range=obstacle_density_range,
            random_layout=layout,
        )
        episode_obs, _ = episode_env.reset()
        return episode_env, episode_obs, episode_meta, scenario

    env, obs, current_episode_meta, current_scenario = start_episode_env()
    current_scenario = current_episode_meta["scenario"]
    agent_prev_pos = {a: env.agent_location[a] for a in agent_order}
    agent_dist = {a: 0.0 for a in agent_order}
    episodes_until_resample = TRAIN_ENV_SAMPLE_EVERY_EPISODES

    pbar = tqdm.tqdm(total=total_episodes)

    try:
        while episode < total_episodes:
            buffer.clear()
            ep_timing = {
                "state_ms": 0.0,
                "actor_ms": 0.0,
                "env_step_ms": 0.0,
                "ppo_ms": 0.0,
            }
            episode_final_state = None

            # ------------------------------------------------------------
            # rollout
            # ------------------------------------------------------------
            for _ in range(rollout_len):
                t0 = time.perf_counter()
                state = stack_global_state(env)
                ep_timing["state_ms"] += (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                actions, logps, values = algo.act_batch(obs, state, agent_order)
                ep_timing["actor_ms"] += (time.perf_counter() - t1) * 1000.0
                for a in agent_order:
                    action_freq[actions[a]] += 1

                t2 = time.perf_counter()
                next_obs, rewards, dones, truncs, infos = env.step(actions)
                ep_timing["env_step_ms"] += (time.perf_counter() - t2) * 1000.0

                # approximate "collision count" via collision penalty occurrences
                # (env assigns collision_penalty=-0.1 to involved agents)
                collisions_ep += sum(1 for a in agent_order if rewards[a] <= -0.01 - 0.1 + 1e-9)
                items_collected_ep += sum(1 for a in agent_order if rewards[a] >= 5.0)

                # distance travelled + coverage
                for a in agent_order:
                    old = agent_prev_pos[a]
                    new = env.agent_location[a]
                    agent_dist[a] += abs(new[0] - old[0]) + abs(new[1] - old[1])
                    agent_prev_pos[a] = new
                    visited.add(new)

                ep_return += float(sum(rewards.values()))
                ep_len += 1
                step_count += num_agents  # keep your original convention

                # store transitions
                for a in agent_order:
                    buffer.add(
                        a,
                        Transition(
                            obs=obs[a],
                            state=state,
                            action=actions[a],
                            logp=logps[a],
                            value=values[a],
                            reward=rewards[a],
                            done=dones[a] or truncs[a],
                        ),
                    )

                obs = next_obs

                # episode end
                if all(dones.values()) or all(truncs.values()):
                    finished_ep_len = ep_len

                    # coverage
                    unique_cells_visited = len(visited)
                    coverage_fraction = unique_cells_visited / (env.grid_w * env.grid_h)

                    # mean pairwise distance
                    pair_dists = []
                    locs = [env.agent_location[a] for a in agent_order]
                    for (x1, y1), (x2, y2) in combinations(locs, 2):
                        pair_dists.append(abs(x1 - x2) + abs(y1 - y2))
                    mean_pairwise = float(np.mean(pair_dists)) if pair_dists else 0.0

                    episode_final_state = stack_global_state(env)

                    pbar.update(1)
                    episode += 1
                    episodes_until_resample -= 1
                    if enable_timing and episode % max(1, timing_every_episodes) == 0 and finished_ep_len > 0:
                        print(
                            f"[Timing][Episode {episode}] "
                            f"state={ep_timing['state_ms']/finished_ep_len:.3f}ms/step, "
                            f"actor={ep_timing['actor_ms']/finished_ep_len:.3f}ms/step, "
                            f"env={ep_timing['env_step_ms']/finished_ep_len:.3f}ms/step"
                        )
                    # Save checkpoint every 50 episodes
                    if episode % 50 == 0:
                        checkpoint_path = f"mappo_hybrid_{grid_shape[0]}x{grid_shape[1]}_{num_agents}agents_mix_actor_ep{episode}.pth"
                        torch.save(algo.actor.state_dict(), checkpoint_path)

                    log_episode = episode % TRAIN_TB_LOG_EVERY_EPISODES == 0 or episode == total_episodes
                    if log_episode:
                        writer.add_scalar("episode/total_return", ep_return, episode)
                        writer.add_scalar("episode/return", ep_return, episode)
                        writer.add_scalar("episode/length", ep_len, episode)
                        writer.add_scalar("episode/collisions_count", collisions_ep, episode)
                        writer.add_scalar("episode/items_collected", items_collected_ep, episode)
                        writer.add_scalar("episode/scenario_is_random", 1.0 if current_scenario == "random" else 0.0, episode)
                        writer.add_scalar("episode/random_obstacle_density", float(current_episode_meta.get("obstacle_density", 0.0)), episode)
                        writer.add_scalar("episode/coverage_fraction", coverage_fraction, episode)
                        writer.add_scalar("episode/mean_pairwise_dist", mean_pairwise, episode)
                        writer.add_scalar("episode/mean_agent_distance", float(np.mean(list(agent_dist.values()))), episode)
                        total_actions = max(sum(action_freq.values()), 1)
                        writer.add_scalar("episode-actions/action_frac_forward", action_freq[0] / total_actions, episode)
                        writer.add_scalar("episode-actions/action_frac_turn_right", action_freq[1] / total_actions, episode)
                        writer.add_scalar("episode-actions/action_frac_turn_left", action_freq[2] / total_actions, episode)
                        writer.add_scalar("episode-actions/action_frac_wait", action_freq[3] / total_actions, episode)
                        if enable_timing and finished_ep_len > 0:
                            writer.add_scalar("timing/state_ms_per_step", ep_timing["state_ms"] / finished_ep_len, episode)
                            writer.add_scalar("timing/actor_ms_per_step", ep_timing["actor_ms"] / finished_ep_len, episode)
                            writer.add_scalar("timing/env_step_ms_per_step", ep_timing["env_step_ms"] / finished_ep_len, episode)

                    # Reset environment for next rollout if not done
                    if episode < total_episodes:
                        if episodes_until_resample <= 0:
                            env.close()
                            env, obs, current_episode_meta, current_scenario = start_episode_env()
                            episodes_until_resample = TRAIN_ENV_SAMPLE_EVERY_EPISODES
                        else:
                            obs, _ = env.reset()
                        agent_prev_pos = {a: env.agent_location[a] for a in agent_order}
                        agent_dist = {a: 0.0 for a in agent_order}
                        visited.clear()
                        action_freq = {0: 0, 1: 0, 2: 0, 3: 0}
                        collisions_ep = 0
                        items_collected_ep = 0
                        ep_return = 0.0
                        ep_len = 0

                    if episode >= total_episodes:
                        break

            # ============================================================
            # PPO Update — TensorBoard update metrics
            # ============================================================
            final_state = episode_final_state if episode_final_state is not None else stack_global_state(env)
            with torch.no_grad():
                v_last = algo.critic(
                    torch.tensor(final_state, dtype=torch.float32, device=device).unsqueeze(0)
                ).item()
            last_vals = {a: v_last for a in agent_order}

            buffer.compute_gae(gamma, lam, last_vals)
            _, _, _, _, vals, advs, rets = buffer.get_flat_batches()

            # explained variance
            var_y = np.var(rets)
            var_diff = np.var(rets - vals)
            explained_var = 1 - var_diff / var_y if var_y > 1e-8 else 0.0

            # value + advantage stats
            adv_mean, adv_std = float(np.mean(advs)), float(np.std(advs))
            val_mean, val_std = float(np.mean(vals)), float(np.std(vals))

            t_ppo = time.perf_counter()
            policy_loss_avg, value_loss_avg, entropy_avg, approx_kl = algo.update(
                buffer, update_epochs, minibatch_size, writer, step_count
            )
            ep_timing["ppo_ms"] += (time.perf_counter() - t_ppo) * 1000.0
            log_update = episode % TRAIN_TB_LOG_EVERY_EPISODES == 0 or episode == total_episodes
            if enable_timing and log_update:
                writer.add_scalar("timing/ppo_update_ms", ep_timing["ppo_ms"], step_count)

            if log_update:
                writer.add_scalar("update/explained_variance", explained_var, step_count)
                writer.add_scalar("update/adv_mean", adv_mean, step_count)
                writer.add_scalar("update/adv_std", adv_std, step_count)
                writer.add_scalar("update/value_mean", val_mean, step_count)
                writer.add_scalar("update/value_std", val_std, step_count)
                writer.add_scalar("update/policy_loss", policy_loss_avg, step_count)
                writer.add_scalar("update/value_loss", value_loss_avg, step_count)
                writer.add_scalar("update/entropy", entropy_avg, step_count)
                writer.add_scalar("update/approx_kl_avg", approx_kl, step_count)

    finally:
        if env is not None:
            env.close()
    writer.close()
    return algo


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    algo = train_mappo(total_episodes=1000, rollout_len=128, device=device)
    torch.save(
        algo.actor.state_dict(),
        f"mappo_hybrid_{TRAIN_GRID_SHAPE[0]}x{TRAIN_GRID_SHAPE[1]}_{TRAIN_NUM_AGENTS}agents_mix_actor.pth",
    )
