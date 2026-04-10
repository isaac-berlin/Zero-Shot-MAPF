import random
import time
from pathlib import Path
from itertools import combinations

import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from train_core import MAPPO, RolloutBuffer, Transition
from train_helpers import (
    build_episode_env,
    build_random_layout_pool_from_dir,
    set_seed,
    stack_global_state,
)


TRAIN_GRID_SHAPE = (16, 16)
TRAIN_NUM_AGENTS = 10
TRAIN_OBS_RADIUS = 5
TRAIN_SCENARIOS = ("warehouse", "warehouse_onewide", "random")
TRAIN_RANDOM_ENV_POOL_SIZE = 1000
TRAIN_ENV_SAMPLE_EVERY_EPISODES = 5
TRAIN_TB_LOG_EVERY_EPISODES = 5
TRAIN_RANDOM_ENV_DIR = Path("generated_random_envs")


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

    def start_episode_env():
        scenario = random.choice(TRAIN_SCENARIOS)
        layout = None
        if scenario == "random":
            layout = random.choice(random_layout_pool)
        episode_env, episode_meta = build_episode_env(
            scenario=scenario,
            obs_radius=obs_radius,
            num_agents=num_agents,
            grid_shape=grid_shape,
            random_layout=layout,
        )
        episode_obs, _ = episode_env.reset()
        return episode_env, episode_obs, episode_meta

    env, obs, current_episode_meta = start_episode_env()
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
                next_obs, rewards, dones, truncs, _infos = env.step(actions)
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
                            env, obs, current_episode_meta = start_episode_env()
                            current_scenario = current_episode_meta["scenario"]
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
                buffer, update_epochs, minibatch_size
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
