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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

JOINT_ORDER = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
DEFAULT_JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "joint1": (-2.967, 2.967),
    "joint2": (-1.5708, 1.5708),
    "joint3": (-1.5708, 1.5708),
    "joint4": (-2.967, 2.967),
    "joint5": (-1.5708, 1.5708),
    "joint6": (-2.967, 2.967),
}


@RobotConfig.register_subclass("tg_arm620_follower")
@dataclass
class TGArm620Config(RobotConfig):
    remote_ip: str = "127.0.0.1"
    cmd_port: int = 6001
    state_port: int = 6002

    # Runtime control/communication cadence.
    control_hz: float = 30.0
    polling_timeout_ms: int = 300
    connect_timeout_s: float = 5.0
    max_latency_ms: int = 300

    # Limits (rad units).
    joint_limits_rad: dict[str, tuple[float, float]] = field(
        default_factory=lambda: DEFAULT_JOINT_LIMITS_RAD.copy()
    )

    # Clamp values aligned with user requirement 150deg/s and 150deg/s^2.
    max_vel_rad_s: float = 2.61799
    max_acc_rad_s2: float = 2.61799

    # Gripper command mapping.
    gripper_type: int = 1
    gripper_speed: float = 40.0
    gripper_effort: float = 100.0

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.remote_ip:
            raise ValueError("`remote_ip` cannot be empty.")

        for port_name, port in (("cmd_port", self.cmd_port), ("state_port", self.state_port)):
            if port <= 0 or port > 65535:
                raise ValueError(f"`{port_name}` must be in [1, 65535], got {port}.")

        if self.control_hz <= 0:
            raise ValueError(f"`control_hz` must be positive, got {self.control_hz}.")
        if self.polling_timeout_ms <= 0:
            raise ValueError(
                f"`polling_timeout_ms` must be positive, got {self.polling_timeout_ms}."
            )
        if self.connect_timeout_s <= 0:
            raise ValueError(f"`connect_timeout_s` must be positive, got {self.connect_timeout_s}.")

        if self.max_vel_rad_s <= 0:
            raise ValueError(f"`max_vel_rad_s` must be positive, got {self.max_vel_rad_s}.")
        if self.max_acc_rad_s2 <= 0:
            raise ValueError(f"`max_acc_rad_s2` must be positive, got {self.max_acc_rad_s2}.")

        for joint in JOINT_ORDER:
            if joint not in self.joint_limits_rad:
                raise ValueError(f"Missing joint limit for `{joint}` in `joint_limits_rad`.")
            low, high = self.joint_limits_rad[joint]
            if low >= high:
                raise ValueError(f"Invalid limit range for `{joint}`: ({low}, {high}).")
