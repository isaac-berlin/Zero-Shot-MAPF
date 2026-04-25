"""Run a MAPF episode with a trained actor and save a GIF/MP4 recording."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.distributions import Categorical

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one MAPF episode with a model and save a GIF or MP4 recording.",
    )
    parser.add_argument(
        "--actor-path",
        type=str,
        default="mappo_hybrid_agents_mix_v2_actor.pth",
        help="Path to trained actor checkpoint (.pth).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/mapf_episode.gif",
        help="Output path (.gif or .mp4).",
    )
    parser.add_argument(
        "--map-path",
        type=str,
        default=r"C:\Users\isaac\dev\Zero-Shot-MAPF\generated_random_envs\random_env_0999_16x16_a10.map",
        help="Optional map path (.domain, benchmark JSON, or map file).",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="hybrid",
        choices=["hybrid"],
        help="Observation mode. Currently only 'hybrid' is supported.",
    )
    parser.add_argument(
        "--obs-radius",
        type=int,
        default=5,
        help="Observation radius (must match model training).",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=10,
        help="Optional number of agents to force in the environment.",
    )
    parser.add_argument(
        "--lifelong",
        action="store_true",
        help="Use lifelong mode (agents get new goals) instead of traditional MAPF done-on-goal.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max timesteps to run before stopping.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for output media.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Capture every Nth step frame (1 = capture all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment reset seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. cpu or cuda. Default: auto-detect.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions instead of stochastic sampling.",
    )
    return parser.parse_args()


def select_actions_batch(
    actor: torch.nn.Module,
    obs_dict: Dict,
    agent_order: List[str],
    device: str,
    stochastic: bool,
) -> Dict[str, int]:
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

    logits = actor(obs_t)
    dist = Categorical(logits=logits)

    if stochastic:
        actions_t = dist.sample()
    else:
        actions_t = torch.argmax(logits, dim=-1)

    return {a: int(actions_t[i].item()) for i, a in enumerate(agent_order)}


def save_frames(frames: List[np.ndarray], output_path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "imageio is required for video/gif export. Install it with: pip install imageio imageio-ffmpeg"
        ) from exc

    if not frames:
        raise RuntimeError("No frames were captured; nothing to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        imageio.mimsave(output_path.as_posix(), frames, fps=fps)
        return

    if suffix == ".mp4":
        try:
            with imageio.get_writer(output_path.as_posix(), fps=fps, codec="libx264") as writer:
                for frame in frames:
                    writer.append_data(frame)
        except Exception as exc:
            raise RuntimeError(
                "Failed to write MP4. Ensure imageio-ffmpeg is installed: pip install imageio-ffmpeg"
            ) from exc
        return

    raise ValueError("Unsupported output extension. Use .gif or .mp4")


def main() -> None:
    from MAPF import MAPF
    from run_mapf import load_actor_for_mode

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stochastic = not args.deterministic

    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be >= 1")
    if args.fps < 1:
        raise ValueError("--fps must be >= 1")

    env = MAPF(
        obs_mode=args.obs_mode,
        map_path=args.map_path,
        obs_radius=args.obs_radius,
        num_agents=args.num_agents,
        lifelong=args.lifelong,
    )

    try:
        obs, _ = env.reset(seed=args.seed)
        if not env.possible_agents:
            raise RuntimeError("Environment has no agents.")

        first_agent = env.possible_agents[0]
        sample_obs = obs[first_agent]
        n_actions = env.action_space(first_agent).n

        actor = load_actor_for_mode(args.obs_mode, sample_obs, n_actions, device)
        actor.load_state_dict(torch.load(args.actor_path, map_location=device))
        actor.eval()

        frames: List[np.ndarray] = []

        first_frame = env.render("rgb_array")
        if first_frame is not None:
            frames.append(first_frame)

        steps = 0
        while env.agents and steps < args.max_steps:
            active_agents = env.agents[:]
            with torch.no_grad():
                actions = select_actions_batch(
                    actor=actor,
                    obs_dict=obs,
                    agent_order=active_agents,
                    device=device,
                    stochastic=stochastic,
                )

            obs, rewards, dones, truncs, infos = env.step(actions)
            del rewards, dones, truncs, infos
            steps += 1

            if steps % args.frame_skip == 0:
                frame = env.render("rgb_array")
                if frame is not None:
                    frames.append(frame)

        output_path = Path(args.output)
        save_frames(frames, output_path, fps=args.fps)

        print(f"Saved recording: {output_path.resolve()}")
        print(f"Frames: {len(frames)}")
        print(f"Steps run: {steps}")
        print(f"Device: {device}")
        print(f"Policy mode: {'stochastic' if stochastic else 'deterministic'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
