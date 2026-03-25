import argparse
import json
import math
from pathlib import Path

import torch


def _to_list(tensor):
    return tensor.detach().cpu().tolist()


def _is_mlp(sd):
    return "net.0.weight" in sd and "net.5.weight" in sd


def _is_cnn(sd):
    return "cnn.0.weight" in sd and "fc.5.weight" in sd and "fusion_fc.0.weight" not in sd


def _is_hybrid(sd):
    return "cnn.0.weight" in sd and "fusion_fc.2.weight" in sd and "knn_fc.1.weight" in sd


def _infer_window_size_from_conv_out(conv_out: int) -> int:
    side_sq = conv_out / 64.0
    side = int(round(math.sqrt(side_sq)))
    if side <= 0 or (64 * side * side) != conv_out:
        raise ValueError(f"Cannot infer window side from conv_out={conv_out}")
    return side


def _export_mlp(sd):
    obs_dim = int(sd["net.0.weight"].numel())
    hidden_1 = int(sd["net.1.weight"].shape[0])
    hidden_2 = int(sd["net.3.weight"].shape[0])
    n_actions = int(sd["net.5.weight"].shape[0])
    k_agents = int((obs_dim - 4) // 3) if obs_dim >= 4 and (obs_dim - 4) % 3 == 0 else None

    return {
        "style": "knn",
        "obs_dim": obs_dim,
        "k_agents": k_agents,
        "hidden_1": hidden_1,
        "hidden_2": hidden_2,
        "n_actions": n_actions,
        "layer_norm": {
            "eps": 1e-5,
            "weight": _to_list(sd["net.0.weight"]),
            "bias": _to_list(sd["net.0.bias"]),
        },
        "layers": [
            {
                "name": "fc1",
                "weight": _to_list(sd["net.1.weight"]),
                "bias": _to_list(sd["net.1.bias"]),
            },
            {
                "name": "fc2",
                "weight": _to_list(sd["net.3.weight"]),
                "bias": _to_list(sd["net.3.bias"]),
            },
            {
                "name": "fc_out",
                "weight": _to_list(sd["net.5.weight"]),
                "bias": _to_list(sd["net.5.bias"]),
            },
        ],
    }


def _export_cnn(sd):
    conv_out = int(sd["fc.0.weight"].numel())
    window_size = _infer_window_size_from_conv_out(conv_out)
    n_actions = int(sd["fc.5.weight"].shape[0])

    return {
        "style": "window",
        "window_size": window_size,
        "obs_channels": 3,
        "n_actions": n_actions,
        "cnn": [
            {"weight": _to_list(sd["cnn.0.weight"]), "bias": _to_list(sd["cnn.0.bias"]), "padding": 1},
            {"weight": _to_list(sd["cnn.2.weight"]), "bias": _to_list(sd["cnn.2.bias"]), "padding": 1},
            {"weight": _to_list(sd["cnn.4.weight"]), "bias": _to_list(sd["cnn.4.bias"]), "padding": 1},
        ],
        "fc": {
            "layer_norm": {
                "eps": 1e-5,
                "weight": _to_list(sd["fc.0.weight"]),
                "bias": _to_list(sd["fc.0.bias"]),
            },
            "layers": [
                {"weight": _to_list(sd["fc.1.weight"]), "bias": _to_list(sd["fc.1.bias"])},
                {"weight": _to_list(sd["fc.3.weight"]), "bias": _to_list(sd["fc.3.bias"])},
                {"weight": _to_list(sd["fc.5.weight"]), "bias": _to_list(sd["fc.5.bias"])},
            ],
        },
    }


def _export_hybrid(sd):
    vec_dim = int(sd["knn_fc.0.weight"].numel())
    conv_out = int(sd["cnn_fc.0.weight"].numel())
    window_size = _infer_window_size_from_conv_out(conv_out)
    n_actions = int(sd["fusion_fc.2.weight"].shape[0])
    k_agents = int((vec_dim - 4) // 3) if vec_dim >= 4 and (vec_dim - 4) % 3 == 0 else None

    return {
        "style": "hybrid",
        "vector_dim": vec_dim,
        "k_agents": k_agents,
        "window_size": window_size,
        "obs_channels": 3,
        "n_actions": n_actions,
        "cnn": [
            {"weight": _to_list(sd["cnn.0.weight"]), "bias": _to_list(sd["cnn.0.bias"]), "padding": 1},
            {"weight": _to_list(sd["cnn.2.weight"]), "bias": _to_list(sd["cnn.2.bias"]), "padding": 1},
            {"weight": _to_list(sd["cnn.4.weight"]), "bias": _to_list(sd["cnn.4.bias"]), "padding": 1},
        ],
        "cnn_fc": {
            "layer_norm": {
                "eps": 1e-5,
                "weight": _to_list(sd["cnn_fc.0.weight"]),
                "bias": _to_list(sd["cnn_fc.0.bias"]),
            },
            "linear": {"weight": _to_list(sd["cnn_fc.1.weight"]), "bias": _to_list(sd["cnn_fc.1.bias"])},
        },
        "knn_fc": {
            "layer_norm": {
                "eps": 1e-5,
                "weight": _to_list(sd["knn_fc.0.weight"]),
                "bias": _to_list(sd["knn_fc.0.bias"]),
            },
            "linear": {"weight": _to_list(sd["knn_fc.1.weight"]), "bias": _to_list(sd["knn_fc.1.bias"])},
        },
        "fusion_fc": {
            "layers": [
                {"weight": _to_list(sd["fusion_fc.0.weight"]), "bias": _to_list(sd["fusion_fc.0.bias"])},
                {"weight": _to_list(sd["fusion_fc.2.weight"]), "bias": _to_list(sd["fusion_fc.2.bias"])},
            ]
        },
    }


def export_policy(checkpoint_path: Path, output_path: Path) -> None:
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")

    if _is_hybrid(state_dict):
        model = _export_hybrid(state_dict)
    elif _is_cnn(state_dict):
        model = _export_cnn(state_dict)
    elif _is_mlp(state_dict):
        model = _export_mlp(state_dict)
    else:
        keys = sorted(list(state_dict.keys()))[:20]
        raise ValueError(
            "Unsupported checkpoint architecture. First keys: " + ", ".join(keys)
        )

    payload = {
        "format": "zsmf_policy_v2",
        "source": "train_mapf.py",
        "checkpoint": str(checkpoint_path),
        "model": model,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Exported policy to {output_path}")
    print(f"Detected style: {model['style']}")
    print(f"n_actions: {model['n_actions']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained MAPF actor checkpoint (knn/window/hybrid) to Start-Kit policy JSON format."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to mappo_*_actor.pth produced by train_mapf.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Start-Kit/policy_export.json"),
        help="Output JSON path consumed by Start-Kit PolicyAdapter.",
    )
    args = parser.parse_args()

    export_policy(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
