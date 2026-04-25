"""Render a single MAPF environment frame and save it to an image file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from MAPF import MAPF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a MAPF environment, render one frame, and save it as an image."
    )
    parser.add_argument(
        "--map-path",
        type=str,
        #default=r"maps\warehouse16.domain\maps\warehouse_16x16_onewide.map",
        default=r"C:\Users\isaac\dev\Zero-Shot-MAPF\LORR_eval\warehouse.domain\maps\sortation_large.map",
        help="Path to a .domain directory, benchmark JSON, or map file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="env_render.png",
        help="Output image path for the composite figure.",
    )
    parser.add_argument("--window-agent-index", type=int, default=0, help="Agent index to visualize.")
    parser.add_argument(
        "--obs-radius",
        type=int,
        default=5,
        help="Observation radius used when constructing the environment.",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="Number of agents to place in the environment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Reset seed for reproducible placement.",
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=16,
        help="Fallback grid width if the map does not specify one.",
    )
    parser.add_argument(
        "--grid-height",
        type=int,
        default=16,
        help="Fallback grid height if the map does not specify one.",
    )
    parser.add_argument(
        "--lifelong",
        action="store_true",
        help="Use lifelong MAPF goal resampling instead of traditional termination at goal.",
    )
    return parser.parse_args()


def _require_matplotlib():
    raise RuntimeError("This helper should not be called.")


def _window_to_rgb(window: np.ndarray):
    # Color code the channels while keeping the encoded values visible.
    # channel 0: blocked / ego center -> blue
    # channel 1: other agents / other goals / overlap -> orange / red
    # channel 2: own goal -> green
    channel0 = np.clip(window[:, :, 0], -1.0, 1.0)
    channel1 = np.clip(window[:, :, 1], 0.0, 1.5)
    channel2 = np.clip(window[:, :, 2], 0.0, 1.0)

    rgb = np.zeros((*window.shape[:2], 3), dtype=np.float32)

    # Base intensity for the blocked map.
    rgb[:, :, 2] += np.where(channel0 < 0.0, 0.85, 0.08)
    rgb[:, :, 1] += np.where(channel0 < 0.0, 0.15, 0.08)
    rgb[:, :, 0] += np.where(channel0 < 0.0, 0.10, 0.08)

    # Other agents / goals: orange-red scale.
    rgb[:, :, 0] += 0.85 * (channel1 / 1.5)
    rgb[:, :, 1] += 0.42 * (channel1 / 1.5)
    rgb[:, :, 2] += 0.08 * (channel1 / 1.5)

    # Own goal: green.
    rgb[:, :, 1] += 0.85 * channel2
    rgb[:, :, 0] += 0.12 * channel2
    rgb[:, :, 2] += 0.12 * channel2

    # Ego marker in the center: white.
    center_y = window.shape[0] // 2
    center_x = window.shape[1] // 2
    rgb[center_y, center_x, :] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    return np.clip(rgb, 0.0, 1.0)


def _window_channel_panel(window: np.ndarray, channel_index: int) -> np.ndarray:
    channel = window[:, :, channel_index]
    panel = np.zeros((window.shape[0], window.shape[1], 3), dtype=np.uint8)

    if channel_index == 0:
        panel[:] = np.array([42, 42, 48], dtype=np.uint8)
        panel[channel < 0.0] = np.array([40, 100, 255], dtype=np.uint8)
        center_y = window.shape[0] // 2
        center_x = window.shape[1] // 2
        panel[center_y, center_x] = np.array([245, 245, 245], dtype=np.uint8)
    elif channel_index == 1:
        intensity = np.clip(channel / 1.5, 0.0, 1.0)
        panel[:, :, 0] = (55 + 200 * intensity).astype(np.uint8)
        panel[:, :, 1] = (30 + 90 * intensity).astype(np.uint8)
        panel[:, :, 2] = (12 + 20 * intensity).astype(np.uint8)
        panel[channel <= 0.0] = np.array([32, 32, 36], dtype=np.uint8)
    else:
        intensity = np.clip(channel, 0.0, 1.0)
        panel[:, :, 0] = (20 + 35 * intensity).astype(np.uint8)
        panel[:, :, 1] = (35 + 210 * intensity).astype(np.uint8)
        panel[:, :, 2] = (18 + 35 * intensity).astype(np.uint8)
        panel[channel <= 0.0] = np.array([32, 32, 36], dtype=np.uint8)

    # Convert from [local_right, local_forward] tensor indexing into image axes,
    # so the saved panel is agent-facing with forward at the top.
    panel = np.flip(np.transpose(panel, (1, 0, 2)), axis=0)
    return panel


def _save_image(array: np.ndarray, output_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to save image outputs. Install it or add an image-saving backend."
        ) from exc

    image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_agent_view_images(
    frame: np.ndarray,
    window: np.ndarray,
    obs_vector: np.ndarray,
    agent_name: str,
    agent_px: tuple[float, float],
    output_path: Path,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to save image outputs. Install it or add an image-saving backend."
        ) from exc

    if window.ndim != 3 or window.shape[2] != 3:
        raise ValueError(f"Expected window shape (H, W, 3), got {window.shape!r}.")
    if obs_vector.shape != (2,):
        raise ValueError(f"Expected observation vector shape (2,), got {obs_vector.shape!r}.")

    def as_image(array: np.ndarray) -> Image.Image:
        return Image.fromarray(array.astype(np.uint8), mode="RGB")

    try:
        resample = Image.Resampling.NEAREST
    except AttributeError:  # Pillow < 9.1
        resample = Image.NEAREST

    def draw_grid(draw: ImageDraw.ImageDraw, panel_size: int, cells_x: int, cells_y: int) -> None:
        if cells_x <= 0 or cells_y <= 0:
            return
        cell_w = panel_size / cells_x
        cell_h = panel_size / cells_y
        grid_color = (255, 255, 255)
        for idx in range(cells_x + 1):
            x = int(round(idx * cell_w))
            draw.line((x, 0, x, panel_size), fill=grid_color, width=1)
        for idx in range(cells_y + 1):
            y = int(round(idx * cell_h))
            draw.line((0, y, panel_size, y), fill=grid_color, width=1)

    def make_panel_image(panel_array: np.ndarray, panel_size: int) -> Image.Image:
        panel_img = as_image(panel_array)
        panel_img = panel_img.resize((panel_size, panel_size), resample=resample)
        return panel_img

    world_img = as_image(frame)
    world_original_w, world_original_h = world_img.size
    panel_size = max(280, min(420, max(world_img.width, world_img.height)))
    world_img = world_img.resize((panel_size, panel_size), resample=resample)

    channel_panels = [
        make_panel_image(_window_channel_panel(window, 0), panel_size),
        make_panel_image(_window_channel_panel(window, 1), panel_size),
        make_panel_image(_window_channel_panel(window, 2), panel_size),
    ]

    for channel_img in channel_panels:
        draw = ImageDraw.Draw(channel_img)
        draw_grid(draw, panel_size, window.shape[1], window.shape[0])

    # Keep environment exports untouched by overlays/markers.

    def save_single_panel(image: Image.Image, output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_file)

    env_path = output_path
    env_arrow_path = output_path.with_name(f"{output_path.stem}_arrow{output_path.suffix}")
    channel_paths = [
        output_path.with_name(f"{output_path.stem}_channel0{output_path.suffix}"),
        output_path.with_name(f"{output_path.stem}_channel1{output_path.suffix}"),
        output_path.with_name(f"{output_path.stem}_channel2{output_path.suffix}"),
    ]

    save_single_panel(world_img, env_path)
    save_single_panel(channel_panels[0], channel_paths[0])
    save_single_panel(channel_panels[1], channel_paths[1])
    save_single_panel(channel_panels[2], channel_paths[2])
    save_single_panel(world_img, env_arrow_path)


def main() -> None:
    args = parse_args()
    env = MAPF(
        obs_mode="hybrid",
        map_path=args.map_path,
        obs_radius=args.obs_radius,
        num_agents=args.num_agents,
        grid_shape=(args.grid_height, args.grid_width),
        lifelong=args.lifelong,
    )

    try:
        obs, _ = env.reset(seed=args.seed)
        frame = env.render("rgb_array")
        if frame is None:
            raise RuntimeError("Environment render returned no frame.")

        output_path = Path(args.output)
        if not env.possible_agents:
            raise RuntimeError("Environment has no agents available for observation export.")

        agent_index = max(0, min(args.window_agent_index, len(env.possible_agents) - 1))
        agent_name = env.possible_agents[agent_index]
        window = env._obs_window(agent_name)
        obs_vector = obs[agent_name]["vector"]
        agent_x, agent_y = env.agent_location[agent_name]
        agent_px = env._grid_to_screen(agent_x, agent_y)
        agent_px = (agent_px[0] + env._cell_size / 2.0, agent_px[1] + env._cell_size / 2.0)

        save_agent_view_images(frame, window, obs_vector, agent_name, agent_px, output_path)
        print(f"Saved environment image to: {output_path.resolve()}")
        print(f"Saved environment-with-arrow image to: {output_path.with_name(f'{output_path.stem}_arrow{output_path.suffix}').resolve()}")
        print(f"Saved channel images next to: {output_path.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
