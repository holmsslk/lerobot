#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run XVLA inference directly from local inputs without implementing a LeRobot Robot."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a trained XVLA checkpoint directory containing config.json and model.safetensors.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        dest="images",
        default=[],
        help="Path to an input image. Repeat this flag for multi-view XVLA models.",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        action="append",
        dest="image_keys",
        default=[],
        help="Optional explicit observation image key. Repeat in the same order as --image.",
    )
    parser.add_argument(
        "--state-json",
        type=str,
        default=None,
        help='JSON list for observation.state, e.g. "[0.1, 0.2, 0.3]".',
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Language instruction for XVLA.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Override device from config.json, e.g. "cuda", "mps", or "cpu".',
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the loaded feature contract before inference.",
    )
    return parser.parse_args()


def load_image(image_path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow is required for --image loading. Install it in your environment.") from e

    image = Image.open(image_path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_state_vector(state_json: str | None, expected_dim: int | None) -> np.ndarray | None:
    if state_json is None:
        return None

    state = np.asarray(json.loads(state_json), dtype=np.float32)
    if state.ndim != 1:
        raise ValueError(f"--state-json must decode to a 1D list, got shape {state.shape}.")
    if expected_dim is not None and state.shape[0] != expected_dim:
        raise ValueError(
            f"State dimension mismatch: expected {expected_dim} from config.json, got {state.shape[0]}."
        )
    return state


def build_observation(
    cfg: PreTrainedConfig,
    image_paths: list[Path],
    image_keys: list[str],
    state: np.ndarray | None,
    task: str,
) -> dict[str, Any]:
    observation: dict[str, Any] = {}

    expected_image_keys = list(cfg.image_features.keys())
    if image_keys and len(image_keys) != len(image_paths):
        raise ValueError("--image-key count must match --image count.")

    if not image_keys:
        if len(image_paths) > len(expected_image_keys):
            raise ValueError(
                f"Model expects at most {len(expected_image_keys)} image views, got {len(image_paths)} image paths."
            )
        image_keys = expected_image_keys[: len(image_paths)]

    for key, path in zip(image_keys, image_paths, strict=True):
        observation[key] = load_image(path)

    expected_state_feature = cfg.robot_state_feature
    if expected_state_feature is not None:
        if state is None:
            raise ValueError(
                f"Model expects {OBS_STATE} with shape {expected_state_feature.shape}, but --state-json was not provided."
            )
        observation[OBS_STATE] = state
    elif state is not None:
        observation[OBS_STATE] = state

    observation["task"] = task
    return observation


def main() -> None:
    args = parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.model_path)
    if cfg.type != "xvla":
        raise ValueError(f"This script only supports XVLA checkpoints, got policy type '{cfg.type}'.")

    if args.device is not None:
        cfg.device = args.device

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.model_path, config=cfg)

    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(args.model_path),
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    expected_state_dim = cfg.robot_state_feature.shape[0] if cfg.robot_state_feature is not None else None
    state = load_state_vector(args.state_json, expected_state_dim)
    observation = build_observation(cfg, args.images, args.image_keys, state, args.task)

    if args.print_config:
        print("policy_type:", cfg.type)
        print("device:", cfg.device)
        print("image_features:", list(cfg.image_features.keys()))
        print("state_feature:", None if cfg.robot_state_feature is None else cfg.robot_state_feature.shape)
        print("action_feature:", None if cfg.action_feature is None else cfg.action_feature.shape)
        print()

    action = predict_action(
        observation=observation,
        policy=policy,
        device=torch.device(cfg.device),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=cfg.use_amp,
        task=args.task,
    )

    action_np = action.squeeze(0).detach().cpu().numpy()
    print("action_shape:", tuple(action_np.shape))
    print("action:", json.dumps(action_np.tolist(), ensure_ascii=False))


if __name__ == "__main__":
    main()
