import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


def sample_obstacle_density(density_range: Tuple[float, float], rng: random.Random) -> float:
    low, high = density_range
    if low > high:
        raise ValueError("density_range must be ordered as (low, high)")

    mean = (low + high) / 2.0
    std = max((high - low) / 6.0, 1e-3)
    density = rng.normalvariate(mean, std)
    return max(low, min(high, density))


def generate_layout(
    width: int,
    height: int,
    num_agents: int,
    density_range: Tuple[float, float],
    rng: random.Random,
) -> dict:
    total_cells = width * height
    density = sample_obstacle_density(density_range, rng)

    # Keep enough free cells for agents plus a small buffer.
    max_blocked = max(0, total_cells - (2 * num_agents + 4))
    blocked_count = min(int(round(density * total_cells)), max_blocked)

    all_cells = [(x, y) for y in range(height) for x in range(width)]
    blocked_cells = rng.sample(all_cells, blocked_count) if blocked_count > 0 else []

    return {
        "grid_shape": [height, width],
        "num_agents": num_agents,
        "obstacle_density": density,
        "blocked_cells": blocked_cells,
    }


def write_layout_files(layout: dict, output_dir: Path, index: int) -> dict:
    height, width = layout["grid_shape"]
    base_name = f"random_env_{index:04d}_{height}x{width}_a{layout['num_agents']}"

    json_path = output_dir / f"{base_name}.json"
    map_path = output_dir / f"{base_name}.map"

    json_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")

    lines = [
        f"GRID {height} {width}",
    ]
    for x, y in layout["blocked_cells"]:
        lines.append(f"BLOCK {x} {y}")
    map_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "name": base_name,
        "json": json_path.name,
        "map": map_path.name,
        "grid_shape": layout["grid_shape"],
        "num_agents": layout["num_agents"],
        "obstacle_density": layout["obstacle_density"],
        "blocked_cells": len(layout["blocked_cells"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random MAPF environments and save them to disk.")
    parser.add_argument("--output-dir", type=Path, default=Path("generated_random_envs"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--density-min", type=float, default=0.0)
    parser.add_argument("--density-max", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    density_range = (args.density_min, args.density_max)

    manifest: List[dict] = []
    for i in range(args.count):
        layout = generate_layout(
            width=args.width,
            height=args.height,
            num_agents=args.num_agents,
            density_range=density_range,
            rng=rng,
        )
        manifest.append(write_layout_files(layout, output_dir, i))

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} layouts to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
