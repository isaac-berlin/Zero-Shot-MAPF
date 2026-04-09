from dataclasses import dataclass
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class MapLoadResult:
    grid_w: int
    grid_h: int
    blocked: Set[Tuple[int, int]]
    spawn_points: List[Tuple[int, int]]
    goal_points: List[Tuple[int, int]]
    using_domain_config: bool
    team_size: Optional[int]
    num_tasks_reveal: Optional[float]
    agent_size: Optional[float]
    max_counter: Optional[int]
    delay_config: Dict


def load_map_configuration(
    path: str,
    requested_agents: Optional[int],
    default_grid_shape: Tuple[int, int],
) -> MapLoadResult:
    """
    Load either:
    - .domain directory containing one or more benchmark JSON configs,
    - benchmark JSON config (.json) that points to .map/.agents/.tasks files, or
    - octile map file (.map) with no spawn/goal points, or
    - legacy text directive format.
    """
    grid_h, grid_w = default_grid_shape

    resolved_path = path
    norm = os.path.normpath(path)
    if os.path.isdir(norm) or norm.lower().endswith(".domain"):
        resolved_path = _resolve_domain_config_json(norm, requested_agents)

    if resolved_path.lower().endswith(".json"):
        return _load_benchmark_json(resolved_path)
    
    # Check if it's an octile map file (Moving AI format)
    if resolved_path.lower().endswith(".map"):
        width, height, blocked = _parse_octile_map_file(resolved_path)
        return MapLoadResult(
            grid_w=width,
            grid_h=height,
            blocked=set(blocked),
            spawn_points=[],
            goal_points=[],
            using_domain_config=False,
            team_size=None,
            num_tasks_reveal=None,
            agent_size=None,
            max_counter=None,
            delay_config={},
        )

    return _load_legacy_map_file(resolved_path, grid_w=grid_w, grid_h=grid_h)


def _resolve_domain_config_json(domain_dir: str, requested_agents: Optional[int]) -> str:
    """
    Resolve a .domain directory into a concrete benchmark JSON config file.

    Selection strategy:
    - if only one JSON exists: use it
    - if multiple JSON files exist and requested_agents is provided: prefer exact teamSize,
      otherwise the smallest teamSize >= requested_agents, otherwise the largest teamSize
    - if no teamSize metadata can be used: pick lexicographically first file
    """
    if not os.path.isdir(domain_dir):
        raise ValueError(f"{domain_dir}: expected an existing .domain directory.")

    json_candidates = [
        os.path.join(domain_dir, name)
        for name in os.listdir(domain_dir)
        if name.lower().endswith(".json")
    ]
    json_candidates.sort()

    if not json_candidates:
        raise ValueError(f"{domain_dir}: no JSON scenario files found in .domain directory.")
    if len(json_candidates) == 1:
        return json_candidates[0]

    annotated: List[Tuple[str, Optional[int]]] = []
    for cfg_path in json_candidates:
        team_size = None
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "teamSize" in cfg:
                team_size = int(cfg["teamSize"])
        except Exception:
            team_size = None
        annotated.append((cfg_path, team_size))

    if requested_agents is not None:
        exact = [p for (p, ts) in annotated if ts == requested_agents]
        if exact:
            return sorted(exact)[0]

        larger_or_equal = sorted(
            [(p, ts) for (p, ts) in annotated if ts is not None and ts >= requested_agents],
            key=lambda item: (item[1], item[0]),
        )
        if larger_or_equal:
            return larger_or_equal[0][0]

        known_sizes = sorted(
            [(p, ts) for (p, ts) in annotated if ts is not None],
            key=lambda item: (item[1], item[0]),
        )
        if known_sizes:
            return known_sizes[-1][0]

    return annotated[0][0]


def _load_benchmark_json(path: str) -> MapLoadResult:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_dir = os.path.dirname(path)

    def resolve_rel(rel_path: str) -> str:
        return os.path.normpath(os.path.join(base_dir, rel_path))

    try:
        map_file = resolve_rel(cfg["mapFile"])
        agent_file = resolve_rel(cfg["agentFile"])
        task_file = resolve_rel(cfg["taskFile"])
    except KeyError as e:
        raise ValueError(f"{path}: missing required key {e.args[0]!r}") from e

    width, height, blocked = _parse_octile_map_file(map_file)
    spawn_points = _parse_agents_file(agent_file, width=width, height=height)
    goal_points = _parse_tasks_file(task_file, width=width, height=height)

    team_size = int(cfg["teamSize"]) if "teamSize" in cfg else None
    num_tasks_reveal = float(cfg["numTasksReveal"]) if "numTasksReveal" in cfg else None
    agent_size = float(cfg["agentSize"]) if "agentSize" in cfg else None

    max_counter = None
    if "maxCounter" in cfg:
        max_counter = int(cfg["maxCounter"])
    elif "agentCounter" in cfg:
        max_counter = int(cfg["agentCounter"])

    delay_cfg = cfg.get("delayConfig", {})
    if isinstance(delay_cfg, str):
        delay_path = resolve_rel(delay_cfg)
        with open(delay_path, "r", encoding="utf-8") as f:
            delay_config = json.load(f)
    elif isinstance(delay_cfg, dict):
        delay_config = delay_cfg
    else:
        raise ValueError(f"{path}: delayConfig must be an object or relative json path.")

    _validate_points(path, width, height, blocked, spawn_points, goal_points)

    return MapLoadResult(
        grid_w=width,
        grid_h=height,
        blocked=set(blocked),
        spawn_points=_dedup_points(spawn_points),
        goal_points=_dedup_points(goal_points),
        using_domain_config=True,
        team_size=team_size,
        num_tasks_reveal=num_tasks_reveal,
        agent_size=agent_size,
        max_counter=max_counter,
        delay_config=delay_config,
    )


def _load_legacy_map_file(path: str, grid_w: int, grid_h: int) -> MapLoadResult:
    """
    Load a text map specification.

    Supported directives (case-insensitive), one per line:

        GRID <H> <W>
            - sets grid shape to (H, W) (optional; overrides constructor arg)

        BLOCK <x> <y>
            - marks a single blocked cell

        BLOCK_RECT <x1> <y1> <x2> <y2>
            - marks a rectangle of blocked cells (inclusive)

        SPAWN <x> <y>
            - adds a candidate spawn location

        GOAL <x> <y>
            - adds a candidate goal location
    """
    blocked: Set[Tuple[int, int]] = set()
    spawns: List[Tuple[int, int]] = []
    goals: List[Tuple[int, int]] = []

    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            key = parts[0].upper()

            def parse_ints(n: int) -> List[int]:
                if len(parts) != n + 1:
                    raise ValueError(f"{path}:{lineno}: expected {n} ints after {parts[0]!r}, got: {parts[1:]}")
                try:
                    return [int(v) for v in parts[1:]]
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno}: non-integer value in: {parts}") from e

            if key == "GRID":
                h, w = parse_ints(2)
                if h <= 1 or w <= 1:
                    raise ValueError(f"{path}:{lineno}: GRID dimensions must be > 1")
                grid_h = h
                grid_w = w
            elif key == "BLOCK":
                x, y = parse_ints(2)
                blocked.add((x, y))
            elif key == "BLOCK_RECT":
                x1, y1, x2, y2 = parse_ints(4)
                xa, xb = sorted((x1, x2))
                ya, yb = sorted((y1, y2))
                for x in range(xa, xb + 1):
                    for y in range(ya, yb + 1):
                        blocked.add((x, y))
            elif key == "SPAWN":
                x, y = parse_ints(2)
                spawns.append((x, y))
            elif key == "GOAL":
                x, y = parse_ints(2)
                goals.append((x, y))
            else:
                raise ValueError(f"{path}:{lineno}: unknown directive {parts[0]!r}")

    _validate_points(path, grid_w, grid_h, blocked, spawns, goals)

    return MapLoadResult(
        grid_w=grid_w,
        grid_h=grid_h,
        blocked=set(blocked),
        spawn_points=_dedup_points(spawns),
        goal_points=_dedup_points(goals),
        using_domain_config=False,
        team_size=None,
        num_tasks_reveal=None,
        agent_size=None,
        max_counter=None,
        delay_config={},
    )


def _parse_octile_map_file(path: str) -> Tuple[int, int, Set[Tuple[int, int]]]:
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    if len(raw_lines) < 4:
        raise ValueError(f"{path}: invalid map file, expected at least 4 header lines.")

    height = None
    width = None
    map_start = None

    for i, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("height"):
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{i + 1}: malformed height line.")
            height = int(parts[1])
        elif lower.startswith("width"):
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{i + 1}: malformed width line.")
            width = int(parts[1])
        elif lower == "map":
            map_start = i + 1
            break

    if height is None or width is None or map_start is None:
        raise ValueError(f"{path}: expected header fields: height, width, and map.")
    if height <= 1 or width <= 1:
        raise ValueError(f"{path}: height and width must be > 1.")

    if len(raw_lines) < map_start + height:
        raise ValueError(f"{path}: expected {height} map rows, got fewer.")

    blocked: Set[Tuple[int, int]] = set()
    obstacle_symbols = {"@", "O", "T"}

    for row in range(height):
        map_row = raw_lines[map_start + row]
        if len(map_row) < width:
            raise ValueError(
                f"{path}:{map_start + row + 1}: map row too short (expected {width}, got {len(map_row)})."
            )
        y = height - 1 - row
        for x, symbol in enumerate(map_row[:width]):
            if symbol in obstacle_symbols:
                blocked.add((x, y))

    return width, height, blocked


def _parse_agents_file(path: str, width: int, height: int) -> List[Tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if not lines:
        raise ValueError(f"{path}: empty agents file.")

    try:
        n_agents = int(lines[0])
    except ValueError as e:
        raise ValueError(f"{path}: first non-comment line must be integer agent count.") from e

    if len(lines) < 1 + n_agents:
        raise ValueError(f"{path}: expected {n_agents} agent locations, found {len(lines) - 1}.")

    starts: List[Tuple[int, int]] = []
    for i in range(n_agents):
        line = lines[1 + i]
        nums = re.findall(r"-?\d+", line)
        if not nums:
            raise ValueError(f"{path}:{i + 2}: missing location index.")
        loc = int(nums[0])
        starts.append(_linear_to_xy(loc, width, height, f"{path}:{i + 2}"))

    return starts


def _parse_tasks_file(path: str, width: int, height: int) -> List[Tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if not lines:
        raise ValueError(f"{path}: empty tasks file.")

    try:
        n_tasks = int(lines[0])
    except ValueError as e:
        raise ValueError(f"{path}: first non-comment line must be integer task count.") from e

    if len(lines) < 1 + n_tasks:
        raise ValueError(f"{path}: expected {n_tasks} task lines, found {len(lines) - 1}.")

    points: List[Tuple[int, int]] = []
    for i in range(n_tasks):
        line = lines[1 + i]
        nums = re.findall(r"-?\d+", line)
        if not nums:
            raise ValueError(f"{path}:{i + 2}: task line has no locations.")
        for token in nums:
            loc = int(token)
            points.append(_linear_to_xy(loc, width, height, f"{path}:{i + 2}"))

    return points


def _linear_to_xy(loc: int, width: int, height: int, source: str) -> Tuple[int, int]:
    if loc < 0:
        raise ValueError(f"{source}: negative location index {loc}.")
    row = loc // width
    col = loc % width
    if row >= height:
        raise ValueError(f"{source}: location index {loc} is out of bounds for {height}x{width}.")
    return col, (height - 1 - row)


def _dedup_points(seq: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _validate_points(
    source: str,
    grid_w: int,
    grid_h: int,
    blocked: Set[Tuple[int, int]],
    spawns: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
) -> None:
    def in_bounds(p: Tuple[int, int]) -> bool:
        x, y = p
        return 0 <= x < grid_w and 0 <= y < grid_h

    for p in blocked:
        if not in_bounds(p):
            raise ValueError(f"{source}: blocked cell out of bounds: {p} for GRID {grid_h}x{grid_w}")
    for p in spawns:
        if not in_bounds(p):
            raise ValueError(f"{source}: spawn out of bounds: {p} for GRID {grid_h}x{grid_w}")
        if p in blocked:
            raise ValueError(f"{source}: spawn on blocked cell: {p}")
    for p in goals:
        if not in_bounds(p):
            raise ValueError(f"{source}: goal out of bounds: {p} for GRID {grid_h}x{grid_w}")
        if p in blocked:
            raise ValueError(f"{source}: goal on blocked cell: {p}")


@dataclass
class MovingAIScenarioCase:
    """Represents a single test case from a Moving AI scenario file."""
    bucket: int
    map_file: str
    map_width: int
    map_height: int
    start_x: int
    start_y: int
    goal_x: int
    goal_y: int
    optimal_length: float


def parse_moving_ai_scenario(scenario_path: str, base_dir: str = None) -> List[MovingAIScenarioCase]:
    """
    Parse a Moving AI .scen scenario file.
    
    Args:
        scenario_path: Path to the .scen file
        base_dir: Base directory for resolving relative map paths (defaults to scenario parent dir)
    
    Returns:
        List of MovingAIScenarioCase objects
    """
    if base_dir is None:
        base_dir = os.path.dirname(scenario_path)
    
    cases: List[MovingAIScenarioCase] = []
    
    with open(scenario_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        raise ValueError(f"{scenario_path}: empty scenario file.")
    
    version_line = lines[0].lower()
    if not version_line.startswith("version"):
        raise ValueError(f"{scenario_path}: first line must be 'version x.x', got: {lines[0]}")
    
    for lineno, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) < 9:
            raise ValueError(f"{scenario_path}:{lineno}: expected 9 fields, got {len(parts)}")
        
        try:
            bucket = int(parts[0])
            map_file = parts[1]
            map_width = int(parts[2])
            map_height = int(parts[3])
            start_x = int(parts[4])
            start_y = int(parts[5])
            goal_x = int(parts[6])
            goal_y = int(parts[7])
            optimal_length = float(parts[8])
        except (ValueError, IndexError) as e:
            raise ValueError(f"{scenario_path}:{lineno}: failed to parse fields: {e}")
        
        # Resolve map file path relative to scenario directory
        try:
            map_path = os.path.normpath(os.path.join(base_dir, map_file))
        except Exception as e:
            raise ValueError(f"{scenario_path}:{lineno}: failed to resolve map path '{map_file}': {e}")
        
        cases.append(
            MovingAIScenarioCase(
                bucket=bucket,
                map_file=map_path,
                map_width=map_width,
                map_height=map_height,
                start_x=start_x,
                start_y=start_y,
                goal_x=goal_x,
                goal_y=goal_y,
                optimal_length=optimal_length,
            )
        )
    
    return cases

