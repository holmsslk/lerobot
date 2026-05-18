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

from __future__ import annotations

import logging
import threading
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.robots.tg_arm620.config_tg_arm620 import JOINT_ORDER
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_tg_arm620_ros2 import TGArm620ROS2TeleopConfig

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    ROS2_AVAILABLE = True
except ImportError:
    rclpy = None
    SingleThreadedExecutor = None
    Node = None
    JointState = None
    ROS2_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


def _zero_action(default_gripper: float) -> dict[str, float]:
    action = {f"{joint}.pos": 0.0 for joint in JOINT_ORDER}
    action["gripper.pos"] = default_gripper
    return action


class TGArm620ROS2Teleoperator(Teleoperator):
    config_class = TGArm620ROS2TeleopConfig
    name = "tg_arm620_ros2"

    def __init__(self, config: TGArm620ROS2TeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False

        self._lock = threading.Lock()
        self._latest_action: dict[str, float] = _zero_action(config.default_gripper)
        self._last_update_time = 0.0

        self._context = None
        self._node = None
        self._executor = None
        self._spin_thread = None
        self._running = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = {f"{joint}.pos": float for joint in JOINT_ORDER}
        features["gripper.pos"] = float
        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if not ROS2_AVAILABLE or rclpy is None or Node is None or JointState is None:
            raise ImportError(
                "ROS2 Python packages are required for tg_arm620_ros2 teleoperation. "
                "Please source your ROS2 environment first."
            )

        self._context = rclpy.Context()
        rclpy.init(context=self._context)

        self._node = Node(self.config.node_name, context=self._context)
        self._node.create_subscription(JointState, self.config.topic, self._joint_callback, 100)

        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)

        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()
        self._is_connected = True
        LOGGER.info(
            "TGArm620ROS2Teleoperator connected. Listening on topic %s.", self.config.topic
        )

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def _joint_callback(self, msg: JointState) -> None:
        if len(msg.position) < 6:
            return

        action = _zero_action(self.config.default_gripper)
        for idx, joint_name in enumerate(JOINT_ORDER):
            value = float(msg.position[idx])
            if joint_name in self.config.invert_joints:
                value = -value
            action[f"{joint_name}.pos"] = value

        if len(msg.position) >= 7:
            gripper_value = float(msg.position[6])
            if abs(gripper_value) <= 1.0:
                gripper_value *= self.config.gripper_scale
            action["gripper.pos"] = float(np.clip(gripper_value, 0.0, 100.0))

        with self._lock:
            self._latest_action = action
            self._last_update_time = time.monotonic()

    def _spin_loop(self) -> None:
        assert self._executor is not None
        while self._running:
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception as exc:  # nosec B110
                if self._running:
                    LOGGER.error("ROS2 spin loop failed: %s", exc)
                break

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        with self._lock:
            age = time.monotonic() - self._last_update_time if self._last_update_time > 0 else None
            action = self._latest_action.copy()

        if age is None or age > self.config.stale_data_timeout_s:
            return _zero_action(self.config.default_gripper)
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback
        return

    def _shutdown_ros(self) -> None:
        self._running = False

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
            self._spin_thread = None

        if self._executor is not None and self._node is not None:
            try:
                self._executor.remove_node(self._node)
            except Exception:  # nosec B110
                pass
            try:
                self._executor.shutdown()
            except Exception:  # nosec B110
                pass
            self._executor = None

        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:  # nosec B110
                pass
            self._node = None

        if self._context is not None and rclpy is not None:
            try:
                rclpy.shutdown(context=self._context)
            except Exception:  # nosec B110
                pass
            self._context = None

    @check_if_not_connected
    def disconnect(self) -> None:
        self._shutdown_ros()
        self._is_connected = False
