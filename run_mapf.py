import torch
from torch.distributions import Categorical
import numpy as np
import time

from MAPF import MAPF
from train_mapf import ActorMLP, ActorCNN, ActorHybrid


# ============================================================
# Unified Run Function
# ============================================================

def load_actor_for_mode(obs_mode, obs_sample, n_actions, device):
    """
    Automatically load the correct actor architecture depending on obs_mode.
    """
    if obs_mode in ("vector", "knn"):
        # obs_sample = 1D vector
        obs_dim = obs_sample.shape[0]
        actor = ActorMLP(obs_dim, n_actions)
    elif obs_mode == "window":
        # obs_sample = (H, W, C)
        obs_shape = obs_sample.shape
        actor = ActorCNN(obs_shape, n_actions)
    elif obs_mode == "hybrid":
        # obs_sample = Dict with "vector" and "window"
        obs_spec = {
            "vector": obs_sample["vector"].shape,
            "window": obs_sample["window"].shape,
        }
        actor = ActorHybrid(obs_spec, n_actions)
    else:
        raise ValueError(f"Unknown obs_mode: {obs_mode}")

    actor.to(device)
    return actor


def run_policy(
    actor_path: str,
    obs_mode: str = "vector",   # "vector", "window", "knn", "hybrid"
    stochastic=True,            # stochastic (sample) vs argmax
    device="cpu",
    obs_radius=3,              # for knn and hybrid
    k_agents=2,                # for knn and hybrid
    map_path=None,
    enable_timing=True,
    timing_every_episodes=1,
):
    """
    Unified environment runner for all MAPPO actor types.

    NOTE: Updated for new MAPF env:
      - No num_items (goals are per-agent)
      - Actions are Discrete(4): forward, turn right, turn left, wait
      - Global state uses agent pos + heading + goal
    """

    # -----------------------------
    # Create MAPF env
    # -----------------------------
    env = MAPF(
        obs_mode=obs_mode,
        map_path=map_path,
        obs_radius=obs_radius,
        k_agents=k_agents,
    )

    agent_order = env.possible_agents[:]

    # -----------------------------
    # Infer obs and actions
    # -----------------------------
    obs, _ = env.reset()
    sample_obs = obs[agent_order[0]]
    n_actions = env.action_space(agent_order[0]).n  # should be 4

    # -----------------------------
    # Load correct actor architecture
    # -----------------------------
    actor = load_actor_for_mode(obs_mode, sample_obs, n_actions, device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    print(f"\nLoaded {obs_mode} policy from: {actor_path}\n")

    def select_actions_batch(obs_dict):
        if obs_mode == "hybrid":
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
        elif obs_mode == "window":
            obs_t = torch.tensor(
                np.stack([obs_dict[a] for a in agent_order]),
                dtype=torch.float32,
                device=device,
            )
        else:
            obs_t = torch.tensor(
                np.stack([obs_dict[a] for a in agent_order]),
                dtype=torch.float32,
                device=device,
            )

        logits = actor(obs_t)
        dist = Categorical(logits=logits)

        if stochastic:
            actions_t = dist.sample()
        else:
            actions_t = torch.argmax(logits, dim=-1)

        return {a: int(actions_t[i].item()) for i, a in enumerate(agent_order)}

    # -----------------------------
    # Run forever
    # -----------------------------
    episode = 0
    while True:
        obs, _ = env.reset()
        done_flags = {a: False for a in agent_order}

        total_reward = 0.0
        steps = 0
        timing = {
            "render_ms": 0.0,
            "actor_ms": 0.0,
            "env_step_ms": 0.0,
        }

        while env.agents and not all(done_flags.values()):
            t_render = time.perf_counter()
            env.render()
            timing["render_ms"] += (time.perf_counter() - t_render) * 1000.0

            t_actor = time.perf_counter()
            with torch.no_grad():
                actions = select_actions_batch(obs)
            timing["actor_ms"] += (time.perf_counter() - t_actor) * 1000.0

            # Step environment
            t_env = time.perf_counter()
            obs, rewards, dones, truncs, infos = env.step(actions)
            timing["env_step_ms"] += (time.perf_counter() - t_env) * 1000.0

            total_reward += float(sum(rewards.values()))
            steps += 1
            done_flags = {a: (dones.get(a, False) or truncs.get(a, False)) for a in agent_order}

        if enable_timing and steps > 0 and episode % max(1, timing_every_episodes) == 0:
            print(
                f"[Episode {episode}] Return: {total_reward:.2f}, Steps: {steps}, "
                f"render={timing['render_ms']/steps:.3f}ms/step, "
                f"actor={timing['actor_ms']/steps:.3f}ms/step, "
                f"env={timing['env_step_ms']/steps:.3f}ms/step"
            )
        else:
            print(f"[Episode {episode}] Return: {total_reward:.2f}, Steps: {steps}")
        episode += 1


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example usage:
    # map_path can be:
    # - a .domain directory (recommended),
    # - a benchmark .json scenario file, or
    # - a legacy text map file.
    run_policy(
        actor_path="mappo_hybrid_random_32_32_20_10_actor.pth",  # or mappo_window_actor.pth or mappo_hybrid_actor.pth
        obs_mode="hybrid",                    # "vector", "window", "knn", or "hybrid"
        stochastic=True,
        device=device,
        map_path="maps/random.domain/random_32_32_20_10.json",
        obs_radius=10,
        k_agents=5,
    )
