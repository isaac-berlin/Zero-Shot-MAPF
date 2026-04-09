from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from train_models import ActorHybrid, CentralCritic


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
            t_steps = len(traj)

            adv = np.zeros(t_steps, np.float32)

            next_adv = 0.0
            next_value = last_values[a]

            for t in reversed(range(t_steps)):
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


class MAPPO:
    def __init__(self, obs_spec, n_actions, state_shape, num_agents, obs_mode, device="cpu"):
        self.device = device
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.num_agents = num_agents

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

    def update(self, buffer, epochs, minibatch):
        obs, state, acts, old_logps, old_vals, advs, rets = buffer.get_flat_batches()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        obs_t = {
            "vector": torch.tensor(obs["vector"], dtype=torch.float32, device=self.device),
            "window": torch.tensor(obs["window"], dtype=torch.float32, device=self.device),
        }
        n_samples = obs_t["vector"].shape[0]

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        acts_t = torch.tensor(acts, dtype=torch.int64, device=self.device)
        old_logps_t = torch.tensor(old_logps, dtype=torch.float32, device=self.device)
        advs_t = torch.tensor(advs, dtype=torch.float32, device=self.device)
        rets_t = torch.tensor(rets, dtype=torch.float32, device=self.device)

        idxs = np.arange(n_samples)

        mean_policy_loss = 0.0
        mean_value_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        count = 0

        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, n_samples, minibatch):
                mb = idxs[start:start + minibatch]

                mb_obs = {
                    "vector": obs_t["vector"][mb],
                    "window": obs_t["window"][mb],
                }
                mb_state = state_t[mb]
                mb_acts = acts_t[mb]
                mb_old_logps = old_logps_t[mb]
                mb_advs = advs_t[mb]
                mb_rets = rets_t[mb]

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
