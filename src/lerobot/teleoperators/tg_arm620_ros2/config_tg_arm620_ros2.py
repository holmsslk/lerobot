#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from lerobot.robots.tg_arm620.config_tg_arm620 import JOINT_ORDER

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("tg_arm620_ros2")
@dataclass
class TGArm620ROS2TeleopConfig(TeleoperatorConfig):
    """
    ROS2 teleoperator config for feeding Arm_Project `/joint_target` commands into LeRobot.

    Action output is always:
    - joint1.pos ... joint6.pos (radians)
    - gripper.pos (0..100)
    """

    topic: str = "/joint_target"
    node_name: str = "lerobot_tg620_ros2_teleop"
    stale_data_timeout_s: float = 0.5
    invert_joints: tuple[str, ...] = ("joint3", "joint4", "joint6")
    gripper_scale: float = 100.0
    default_gripper: float = 0.0

    def __post_init__(self) -> None:
        if not self.topic:
            raise ValueError("`topic` cannot be empty.")
        if not self.node_name:
            raise ValueError("`node_name` cannot be empty.")
        if self.stale_data_timeout_s <= 0:
            raise ValueError(
                f"`stale_data_timeout_s` must be positive, got {self.stale_data_timeout_s}."
            )
        unknown = set(self.invert_joints) - set(JOINT_ORDER)
        if unknown:
            raise ValueError(f"`invert_joints` contains unknown joints: {sorted(unknown)}.")
        if self.gripper_scale <= 0:
            raise ValueError(f"`gripper_scale` must be positive, got {self.gripper_scale}.")
        if not 0.0 <= self.default_gripper <= 100.0:
            raise ValueError(
                f"`default_gripper` must be within [0, 100], got {self.default_gripper}."
            )
