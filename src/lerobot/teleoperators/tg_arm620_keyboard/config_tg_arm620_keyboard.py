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

from lerobot.robots.tg_arm620.config_tg_arm620 import DEFAULT_JOINT_LIMITS_RAD, JOINT_ORDER

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("tg_arm620_keyboard")
@dataclass
class TGArm620KeyboardConfig(TeleoperatorConfig):
    """
    Keyboard teleoperator config for TG arm620.

    Action output is always:
    - joint1.pos ... joint6.pos (radians)
    - gripper.pos (0..100)
    """

    joint_step_rad: float = 0.02
    gripper_step: float = 2.0

    positive_joint_keys: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")
    negative_joint_keys: tuple[str, ...] = ("q", "w", "e", "r", "t", "y")
    gripper_close_key: str = "o"
    gripper_open_key: str = "p"
    home_key: str = "h"
    # `auto`: prefer pynput global listener, fallback to stdin terminal reader when unavailable.
    # `pynput`: force global listener.
    # `stdin`: force terminal local reader (recommended in headless/Wayland/VM sessions).
    input_backend: str = "auto"

    joint_limits_rad: dict[str, tuple[float, float]] = field(
        default_factory=lambda: DEFAULT_JOINT_LIMITS_RAD.copy()
    )
    gripper_min: float = 0.0
    gripper_max: float = 100.0

    def __post_init__(self) -> None:
        if self.joint_step_rad <= 0:
            raise ValueError(f"`joint_step_rad` must be positive, got {self.joint_step_rad}.")
        if self.gripper_step <= 0:
            raise ValueError(f"`gripper_step` must be positive, got {self.gripper_step}.")

        if len(self.positive_joint_keys) != len(JOINT_ORDER):
            raise ValueError(
                f"`positive_joint_keys` must have {len(JOINT_ORDER)} keys, got "
                f"{len(self.positive_joint_keys)}."
            )
        if len(self.negative_joint_keys) != len(JOINT_ORDER):
            raise ValueError(
                f"`negative_joint_keys` must have {len(JOINT_ORDER)} keys, got "
                f"{len(self.negative_joint_keys)}."
            )

        all_keys = (
            list(self.positive_joint_keys)
            + list(self.negative_joint_keys)
            + [self.gripper_close_key, self.gripper_open_key, self.home_key]
        )
        if len(set(all_keys)) != len(all_keys):
            raise ValueError("Key bindings must be unique.")

        for joint in JOINT_ORDER:
            if joint not in self.joint_limits_rad:
                raise ValueError(f"Missing joint limit for `{joint}` in `joint_limits_rad`.")
            low, high = self.joint_limits_rad[joint]
            if low >= high:
                raise ValueError(f"Invalid limit range for `{joint}`: ({low}, {high}).")

        if self.gripper_min >= self.gripper_max:
            raise ValueError(
                f"`gripper_min` must be smaller than `gripper_max`, got ({self.gripper_min}, "
                f"{self.gripper_max})."
            )

        if self.input_backend not in {"auto", "pynput", "stdin"}:
            raise ValueError(
                f"`input_backend` must be one of ('auto', 'pynput', 'stdin'), got {self.input_backend!r}."
            )
