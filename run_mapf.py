import torch
from torch.distributions import Categorical

from MAPF import MAPF
from train_mapf import ActorMLP, ActorCNN, stack_global_state


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
    else:
        raise ValueError(f"Unknown obs_mode: {obs_mode}")

    actor.to(device)
    return actor


def run_policy(
    actor_path: str,
    obs_mode: str = "vector",   # "vector", "window", "knn"
    grid_size=7,
    num_agents=3,
    stochastic=True,            # stochastic (sample) vs argmax
    device="cpu",
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
        grid_size=grid_size,
        num_agents=num_agents,
        obs_mode=obs_mode,
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

    # -----------------------------
    # Run forever
    # -----------------------------
    episode = 0
    while True:
        obs, _ = env.reset()
        done_flags = {a: False for a in agent_order}

        total_reward = 0.0
        steps = 0

        while env.agents and not all(done_flags.values()):
            env.render()

            # Build CTDE global state
            state = stack_global_state(env)

            actions = {}
            for a in agent_order:
                o = obs[a]

                # ActorCNN expects (B,H,W,C), ActorMLP expects (B,D)
                o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)

                logits = actor(o_t)
                dist = Categorical(logits=logits)

                if stochastic:
                    act = dist.sample()
                else:
                    act = torch.argmax(logits, dim=-1)

                actions[a] = int(act.item())

            # Step environment
            obs, rewards, dones, truncs, infos = env.step(actions)

            total_reward += float(sum(rewards.values()))
            steps += 1
            done_flags = {a: (dones.get(a, False) or truncs.get(a, False)) for a in agent_order}

        print(f"[Episode {episode}] Return: {total_reward:.2f}, Steps: {steps}")
        episode += 1


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example usage:
    run_policy(
        actor_path="mappo_window_actor.pth",  # or mappo_window_actor.pth or mappo_knn_actor.pth
        obs_mode="window",                    # "vector", "window", or "knn"
        grid_size=7,
        num_agents=3,
        stochastic=True,
        device=device,
    )
