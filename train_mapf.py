import random
import time
from dataclasses import dataclass
from typing import Dict
from itertools import combinations

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from MAPF import MAPF


# ============================================================
# Utils
# ============================================================

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stack_global_state(env) -> np.ndarray:
    """
    Global CTDE state for MAPF:
      concat (agent_x, agent_y, heading, goal_x, goal_y) for each agent in env.possible_agents.
    """
    state = []
    for agent in env.possible_agents:
        ax, ay = env.agent_location[agent]
        h = env.agent_dir[agent]
        gx, gy = env.goal_locations[agent]
        state.extend([ax, ay, h, gx, gy])
    return np.array(state, dtype=np.float32)


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
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


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
    def __init__(self, obs_spec, n_actions, state_dim, num_agents, obs_mode, device="cpu"):
        self.device = device
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.num_agents = num_agents
        self.mode = obs_mode

        if obs_mode != "hybrid":
            raise ValueError("This trainer now only supports obs_mode='hybrid'.")
        self.actor = ActorHybrid(obs_spec, n_actions).to(device)

        self.critic = CentralCritic(state_dim).to(device)

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

                if writer:
                    writer.add_scalar("loss/policy", policy_loss.item(), global_step)
                    writer.add_scalar("loss/value", vf_loss.item(), global_step)
                    writer.add_scalar("loss/entropy", entropy.item(), global_step)
                    writer.add_scalar("diagnostics/approx_kl", kl.item(), global_step)

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
    env,
    total_episodes=3000,
    rollout_len=128,
    update_epochs=4,
    minibatch_size=256,
    gamma=0.99,
    lam=0.95,
    device="cpu",
    enable_timing=True,
    timing_every_episodes=10,
):
    agent_order = env.possible_agents[:]
    num_agents = len(agent_order)

    dummy_obs, _ = env.reset()
    sample_obs = dummy_obs[agent_order[0]]

    obs_spec = {
        "vector": sample_obs["vector"].shape,
        "window": sample_obs["window"].shape,
    }

    state_dim = stack_global_state(env).shape[0]
    n_actions = env.action_space(agent_order[0]).n  # should be 4 now

    algo = MAPPO(
        obs_spec=obs_spec,
        n_actions=n_actions,
        state_dim=state_dim,
        num_agents=num_agents,
        obs_mode=env.obs_mode,
        device=device,
    )

    buffer = RolloutBuffer(agent_order)
    writer = SummaryWriter(log_dir=f"runs/mapf_{env.obs_mode}")

    obs, _ = env.reset()
    ep_return, ep_len, step_count, episode = 0.0, 0, 0, 0

    # per-episode metric tracking
    visited = set()
    agent_prev_pos = {a: env.agent_location[a] for a in agent_order}
    agent_dist = {a: 0.0 for a in agent_order}
    action_freq = {0: 0, 1: 0, 2: 0, 3: 0}
    collisions_ep = 0

    pbar = tqdm.tqdm(total=total_episodes)

    while episode < total_episodes:
        buffer.clear()
        ep_timing = {
            "state_ms": 0.0,
            "actor_ms": 0.0,
            "env_step_ms": 0.0,
            "ppo_ms": 0.0,
        }

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

                # tensorboard episode metrics
                writer.add_scalar("episode/return", ep_return, episode)
                writer.add_scalar("episode/length", ep_len, episode)
                writer.add_scalar("episode/coverage_fraction", coverage_fraction, episode)
                writer.add_scalar("episode/mean_pairwise_dist", mean_pairwise, episode)
                writer.add_scalar("episode/collisions_count", collisions_ep, episode)
                if enable_timing and finished_ep_len > 0:
                    writer.add_scalar("timing/state_ms_per_step", ep_timing["state_ms"] / finished_ep_len, episode)
                    writer.add_scalar("timing/actor_ms_per_step", ep_timing["actor_ms"] / finished_ep_len, episode)
                    writer.add_scalar("timing/env_step_ms_per_step", ep_timing["env_step_ms"] / finished_ep_len, episode)

                # action frequencies (per-episode)
                total_actions = max(sum(action_freq.values()), 1)
                writer.add_scalar("episode-actions/action_frac_forward", action_freq[0] / total_actions, episode)
                writer.add_scalar("episode-actions/action_frac_turn_right", action_freq[1] / total_actions, episode)
                writer.add_scalar("episode-actions/action_frac_turn_left", action_freq[2] / total_actions, episode)
                writer.add_scalar("episode-actions/action_frac_wait", action_freq[3] / total_actions, episode)

                # agent distance travelled (mean)
                writer.add_scalar("episode/mean_agent_distance", float(np.mean(list(agent_dist.values()))), episode)

                # reset episode metrics
                obs, _ = env.reset()
                ep_return, ep_len = 0.0, 0
                visited.clear()
                agent_prev_pos = {a: env.agent_location[a] for a in agent_order}
                agent_dist = {a: 0.0 for a in agent_order}
                action_freq = {0: 0, 1: 0, 2: 0, 3: 0}
                collisions_ep = 0

                pbar.update(1)
                episode += 1
                if enable_timing and episode % max(1, timing_every_episodes) == 0 and finished_ep_len > 0:
                    print(
                        f"[Timing][Episode {episode}] "
                        f"state={ep_timing['state_ms']/finished_ep_len:.3f}ms/step, "
                        f"actor={ep_timing['actor_ms']/finished_ep_len:.3f}ms/step, "
                        f"env={ep_timing['env_step_ms']/finished_ep_len:.3f}ms/step"
                    )
                # Save checkpoint every 50 episodes
                if episode % 50 == 0:
                    checkpoint_path = f"mappo_{env.obs_mode}_{map_path.split('/')[-1].split('.')[0]}_actor_ep{episode}.pth"
                    torch.save(algo.actor.state_dict(), checkpoint_path)
                if episode >= total_episodes:
                    break

        # ============================================================
        # PPO Update — TensorBoard update metrics
        # ============================================================
        final_state = stack_global_state(env)
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
        if enable_timing:
            writer.add_scalar("timing/ppo_update_ms", ep_timing["ppo_ms"], step_count)

        # update-level TB logging
        writer.add_scalar("update/explained_variance", explained_var, step_count)
        writer.add_scalar("update/adv_mean", adv_mean, step_count)
        writer.add_scalar("update/adv_std", adv_std, step_count)
        writer.add_scalar("update/value_mean", val_mean, step_count)
        writer.add_scalar("update/value_std", val_std, step_count)
        writer.add_scalar("update/policy_loss", policy_loss_avg, step_count)
        writer.add_scalar("update/value_loss", value_loss_avg, step_count)
        writer.add_scalar("update/entropy", entropy_avg, step_count)
        writer.add_scalar("update/approx_kl_avg", approx_kl, step_count)

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
    # map_path can be:
    # - a .domain directory (recommended),
    # - a benchmark .json scenario file, or
    # - a legacy text map file.
    map_path = "maps/random.domain/random_32_32_20_10.json"
    obs_mode = "hybrid"
    env = MAPF(
        obs_mode=obs_mode,
        obs_radius=10,
        map_path=map_path,
    )

    algo = train_mappo(env, total_episodes=1000, rollout_len=128, device=device)
    torch.save(algo.actor.state_dict(), f"mappo_{obs_mode}_{map_path.split('/')[-1].split('.')[0]}_actor.pth")
