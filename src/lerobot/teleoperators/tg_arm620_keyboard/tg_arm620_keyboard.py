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

import logging
import os
import sys
from functools import cached_property
from queue import Queue
from typing import Any

from lerobot.robots.tg_arm620.config_tg_arm620 import JOINT_ORDER
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available, require_package

from ..teleoperator import Teleoperator
from .config_tg_arm620_keyboard import TGArm620KeyboardConfig

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
            logging.info("No DISPLAY set. Skipping pynput import.")
            PYNPUT_AVAILABLE = False
        else:
            from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info(f"Could not import pynput: {e}")

logger = logging.getLogger(__name__)


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


class TGArm620Keyboard(Teleoperator):
    """
    Keyboard teleoperator for TG arm620.

    Key mapping defaults:
    - Increase joints: 1 2 3 4 5 6
    - Decrease joints: q w e r t y
    - Gripper close/open: o / p
    - Reset to home target: h
    - Exit listener: Esc
    """

    config_class = TGArm620KeyboardConfig
    name = "tg_arm620_keyboard"

    def __init__(self, config: TGArm620KeyboardConfig):
        require_package("pynput", extra="pynput-dep")
        super().__init__(config)
        self.config = config

        self._event_queue: Queue[tuple[str, bool]] = Queue()
        self._pressed: dict[str, bool] = {}
        self._listener = None
        self._is_connected = False

        self._target_action: dict[str, float] = self._zero_action()
        self._home_action: dict[str, float] = self._zero_action()

        self._positive_key_by_joint = {
            joint: key for joint, key in zip(JOINT_ORDER, config.positive_joint_keys, strict=True)
        }
        self._negative_key_by_joint = {
            joint: key for joint, key in zip(JOINT_ORDER, config.negative_joint_keys, strict=True)
        }

    def _zero_action(self) -> dict[str, float]:
        action = {f"{joint}.pos": 0.0 for joint in JOINT_ORDER}
        action["gripper.pos"] = self.config.gripper_min
        return action

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
        if not PYNPUT_AVAILABLE or keyboard is None:
            raise RuntimeError(
                "pynput is not available for keyboard teleoperation. "
                "Install it with `pip install 'lerobot[pynput-dep]'` and run in an environment with DISPLAY."
            )

        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        self._is_connected = True
        logger.info("TGArm620Keyboard connected.")

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def _key_to_str(self, key: Any) -> str | None:
        char = getattr(key, "char", None)
        if char is None:
            return None
        return str(char).lower()

    def _on_press(self, key: Any) -> None:
        key_str = self._key_to_str(key)
        if key_str is not None:
            self._event_queue.put((key_str, True))

    def _on_release(self, key: Any) -> None:
        key_str = self._key_to_str(key)
        if key_str is not None:
            self._event_queue.put((key_str, False))
            return

        if keyboard is not None and key == keyboard.Key.esc:
            self.disconnect()

    def _drain_key_events(self) -> None:
        while not self._event_queue.empty():
            key_str, pressed = self._event_queue.get_nowait()
            if pressed:
                self._pressed[key_str] = True
            else:
                self._pressed.pop(key_str, None)

    def _is_key_pressed(self, key: str) -> bool:
        return self._pressed.get(key, False)

    def _apply_joint_steps(self) -> None:
        for joint in JOINT_ORDER:
            pos_key = self._positive_key_by_joint[joint]
            neg_key = self._negative_key_by_joint[joint]
            pos_pressed = self._is_key_pressed(pos_key)
            neg_pressed = self._is_key_pressed(neg_key)

            delta = 0.0
            if pos_pressed and not neg_pressed:
                delta = self.config.joint_step_rad
            elif neg_pressed and not pos_pressed:
                delta = -self.config.joint_step_rad

            key = f"{joint}.pos"
            low, high = self.config.joint_limits_rad[joint]
            self._target_action[key] = _clip(self._target_action[key] + delta, low, high)

    def _apply_gripper_step(self) -> None:
        open_pressed = self._is_key_pressed(self.config.gripper_open_key)
        close_pressed = self._is_key_pressed(self.config.gripper_close_key)

        delta = 0.0
        if open_pressed and not close_pressed:
            delta = self.config.gripper_step
        elif close_pressed and not open_pressed:
            delta = -self.config.gripper_step

        self._target_action["gripper.pos"] = _clip(
            self._target_action["gripper.pos"] + delta, self.config.gripper_min, self.config.gripper_max
        )

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_key_events()

        if self._is_key_pressed(self.config.home_key):
            self._target_action = self._home_action.copy()

        self._apply_joint_steps()
        self._apply_gripper_step()

        return self._target_action.copy()

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        updated = self._target_action.copy()
        for key in self.action_features:
            if key not in feedback:
                continue

            try:
                value = float(feedback[key])
            except (TypeError, ValueError):
                continue

            if key == "gripper.pos":
                updated[key] = _clip(value, self.config.gripper_min, self.config.gripper_max)
            else:
                joint = key.split(".", maxsplit=1)[0]
                low, high = self.config.joint_limits_rad[joint]
                updated[key] = _clip(value, low, high)

        self._target_action = updated
        self._home_action = updated.copy()

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

        self._pressed.clear()
        while not self._event_queue.empty():
            self._event_queue.get_nowait()
        self._is_connected = False
        logger.info("TGArm620Keyboard disconnected.")
