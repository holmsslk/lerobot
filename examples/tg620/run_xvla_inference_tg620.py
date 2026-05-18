#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.policies import get_policy_class, make_pre_post_processors, make_robot_action
from lerobot.policies.utils import validate_visual_features_consistency
from lerobot.processor import (
    RobotObservation,
    RobotProcessorPipeline,
    TG620Joint7ObservationProcessorStep,
    make_default_processors,
    observation_to_transition,
    transition_to_observation,
)
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import combine_feature_dicts, dataset_to_policy_features
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an XVLA checkpoint online on TG620 through LeRobot.")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a trained XVLA checkpoint directory containing config.json and model.safetensors.",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        required=True,
        help="Path to a JSON RobotConfig for tg_arm620_follower.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Language instruction passed to the XVLA policy.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Control loop frequency for online inference.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum number of inference steps. Use 0 to run forever.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Override checkpoint device, e.g. "cuda", "cpu", or "mps".',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the loop and print predicted actions without sending commands to the robot.",
    )
    parser.add_argument(
        "--print-contract",
        action="store_true",
        help="Print the normalized TG620 observation/action contract inferred for inference.",
    )
    return parser.parse_args()


def load_robot_config(config_path: Path) -> RobotConfig:
    import draccus

    with draccus.config_type("json"):
        return draccus.parse(RobotConfig, config_path, args=[])


def build_dataset_features(robot) -> dict[str, dict]:
    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    if robot.name == "tg_arm620_follower":
        robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
            steps=[TG620Joint7ObservationProcessorStep()],
            to_transition=observation_to_transition,
            to_output=transition_to_observation,
        )

    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=False,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=False,
        ),
    )


def validate_contract(cfg: PreTrainedConfig, dataset_features: dict[str, dict]) -> None:
    policy_features = dataset_to_policy_features(dataset_features)
    provided_feature_names = set(policy_features)
    expected_feature_names = set(cfg.input_features or {})

    missing_inputs = expected_feature_names - provided_feature_names
    if missing_inputs:
        raise ValueError(
            "Robot observation contract does not satisfy the XVLA checkpoint.\n"
            f"Missing input features: {sorted(missing_inputs)}\n"
            f"Available robot features: {sorted(provided_feature_names)}"
        )

    validate_visual_features_consistency(cfg, policy_features)

    state_feature = cfg.robot_state_feature
    dataset_state = dataset_features.get(OBS_STATE)
    if state_feature is not None and dataset_state is not None and dataset_state["shape"] != state_feature.shape:
        raise ValueError(
            f"State shape mismatch: checkpoint expects {state_feature.shape}, "
            f"but TG620 inference builds {dataset_state['shape']}."
        )

    action_feature = cfg.action_feature
    dataset_action = dataset_features.get(ACTION)
    if action_feature is None or dataset_action is None:
        raise ValueError("Both checkpoint action feature and TG620 action feature must be present.")
    if dataset_action["shape"] != action_feature.shape:
        raise ValueError(
            f"Action shape mismatch: checkpoint expects {action_feature.shape}, "
            f"but TG620 robot action contract is {dataset_action['shape']}."
        )


def print_contract(cfg: PreTrainedConfig, dataset_features: dict[str, dict]) -> None:
    print("checkpoint_inputs:", sorted((cfg.input_features or {}).keys()))
    print("checkpoint_outputs:", sorted((cfg.output_features or {}).keys()))
    print("dataset_features:", sorted(dataset_features.keys()))
    print("action_names:", dataset_features[ACTION]["names"])
    if OBS_STATE in dataset_features:
        print("state_names:", dataset_features[OBS_STATE]["names"])
    print()


def main() -> None:
    args = parse_args()
    init_logging()
    register_third_party_plugins()

    cfg = PreTrainedConfig.from_pretrained(args.model_path)
    if cfg.type != "xvla":
        raise ValueError(f"This script only supports XVLA checkpoints, got policy type '{cfg.type}'.")
    if args.device is not None:
        cfg.device = args.device

    robot_cfg = load_robot_config(args.robot_config)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.model_path, config=cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(args.model_path),
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    robot = make_robot_from_config(robot_cfg)
    robot.connect()

    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[TG620Joint7ObservationProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    dataset_features = build_dataset_features(robot)
    validate_contract(cfg, dataset_features)
    if args.print_contract:
        print_contract(cfg, dataset_features)

    device = torch.device(cfg.device)

    step = 0
    try:
        while args.max_steps <= 0 or step < args.max_steps:
            start_t = time.perf_counter()

            raw_obs = robot.get_observation()
            obs_processed = robot_observation_processor(raw_obs)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            action_tensor = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=cfg.use_amp,
                task=args.task,
                robot_type=robot.robot_type,
            )

            robot_action = make_robot_action(action_tensor, dataset_features)
            if args.dry_run:
                LOGGER.info("Predicted TG620 action: %s", robot_action)
            else:
                robot.send_action(robot_action)

            step += 1
            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - start_t), 0.0))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
