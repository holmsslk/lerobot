#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import draccus
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.robots.tg_arm620.config_tg_arm620 import JOINT_ORDER
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
except ImportError:  # pragma: no cover - runtime environment specific
    rclpy = None
    SingleThreadedExecutor = None
    Node = None
    JointState = None


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    cli_parser = argparse.ArgumentParser(
        description="Bridge Arm_Project ROS2 /joint_target teleop commands into LeRobot TGArm620Follower."
    )
    cli_parser.add_argument(
        "--robot-config",
        type=Path,
        required=True,
        help="Path to a robot config JSON file for a LeRobot tg_arm620_follower.",
    )
    cli_parser.add_argument(
        "--joint-topic",
        type=str,
        default="/joint_target",
        help="ROS2 JointState topic carrying teleop commands.",
    )
    cli_parser.add_argument(
        "--ros-node-name",
        type=str,
        default="lerobot_tg620_joint_bridge",
        help="ROS2 node name for this bridge.",
    )
    cli_parser.add_argument(
        "--control-hz",
        type=float,
        default=30.0,
        help="Maximum command forwarding frequency to the LeRobot robot.",
    )
    cli_parser.add_argument(
        "--stale-command-timeout-s",
        type=float,
        default=0.5,
        help="If no new ROS2 command arrives within this timeout, stop forwarding commands.",
    )
    cli_parser.add_argument(
        "--invert-joints",
        type=str,
        default="joint3,joint4,joint6",
        help="Comma-separated joint names to invert, matching Arm_Project's historical ROS2 mapping.",
    )
    cli_parser.add_argument(
        "--gripper-scale",
        type=float,
        default=100.0,
        help="Scale factor when incoming gripper command is in [0, 1]. Values above 1 are passed through.",
    )
    cli_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log translated commands without sending them to the robot.",
    )
    return cli_parser.parse_args()


def load_robot_config(config_path: Path) -> RobotConfig:
    del json
    with draccus.config_type("json"):
        return draccus.parse(RobotConfig, config_path, args=[])


def make_joint_target_bridge_class():
    if Node is None or JointState is None:
        raise ImportError("ROS2 Python packages are required to construct the bridge node.")

    class JointTargetBridge(Node):
        def __init__(
            self,
            robot: Robot,
            topic: str,
            node_name: str,
            control_hz: float,
            stale_command_timeout_s: float,
            invert_joints: set[str],
            gripper_scale: float,
            dry_run: bool,
        ) -> None:
            super().__init__(node_name)
            self.robot = robot
            self.control_hz = control_hz
            self.stale_command_timeout_s = stale_command_timeout_s
            self.invert_joints = invert_joints
            self.gripper_scale = gripper_scale
            self.dry_run = dry_run

            self._latest_action: dict[str, float] | None = None
            self._latest_timestamp = 0.0
            self._last_sent_timestamp = 0.0

            self.create_subscription(JointState, topic, self._joint_callback, 100)
            self.get_logger().info(
                f"Listening on {topic} and forwarding commands to {self.robot.__class__.__name__}."
            )

        def _joint_callback(self, msg: JointState) -> None:
            if len(msg.position) < 6:
                self.get_logger().warning(
                    f"Received JointState with {len(msg.position)} positions; expected at least 6."
                )
                return

            action: dict[str, float] = {}
            for idx, joint_name in enumerate(JOINT_ORDER):
                value = float(msg.position[idx])
                if joint_name in self.invert_joints:
                    value = -value
                action[f"{joint_name}.pos"] = value

            if len(msg.position) >= 7:
                gripper_value = float(msg.position[6])
                if abs(gripper_value) <= 1.0:
                    gripper_value *= self.gripper_scale
                action["gripper.pos"] = float(np.clip(gripper_value, 0.0, 100.0))
            else:
                action["gripper.pos"] = 0.0

            self._latest_action = action
            self._latest_timestamp = time.monotonic()

        def spin_step(self) -> None:
            if self._latest_action is None:
                return

            now = time.monotonic()
            if now - self._latest_timestamp > self.stale_command_timeout_s:
                return

            min_period = 1.0 / self.control_hz
            if now - self._last_sent_timestamp < min_period:
                return

            if self.dry_run:
                LOGGER.info("Dry-run action: %s", self._latest_action)
            else:
                self.robot.send_action(self._latest_action)
            self._last_sent_timestamp = now

    return JointTargetBridge


def main() -> None:
    args = parse_args()

    if rclpy is None or SingleThreadedExecutor is None or JointState is None:
        raise ImportError(
            "ROS2 Python packages are required. Source your ROS2 environment before running this script."
        )
    init_logging()
    register_third_party_plugins()

    robot_cfg = load_robot_config(args.robot_config)
    robot = make_robot_from_config(robot_cfg)
    robot.connect()

    invert_joints = {name.strip() for name in args.invert_joints.split(",") if name.strip()}
    JointTargetBridge = make_joint_target_bridge_class()

    rclpy.init()
    bridge = JointTargetBridge(
        robot=robot,
        topic=args.joint_topic,
        node_name=args.ros_node_name,
        control_hz=args.control_hz,
        stale_command_timeout_s=args.stale_command_timeout_s,
        invert_joints=invert_joints,
        gripper_scale=args.gripper_scale,
        dry_run=args.dry_run,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.05)
            bridge.spin_step()
    finally:
        executor.remove_node(bridge)
        bridge.destroy_node()
        robot.disconnect()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
