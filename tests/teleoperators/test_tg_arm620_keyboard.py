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

from unittest.mock import patch

import pytest

from lerobot.teleoperators.tg_arm620_keyboard import TGArm620Keyboard, TGArm620KeyboardConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.utils.errors import DeviceNotConnectedError


class _FakeKey:
    esc = object()


class _FakeListener:
    def __init__(self, on_press, on_release):
        self.on_press = on_press
        self.on_release = on_release
        self._alive = False

    def start(self):
        self._alive = True

    def stop(self):
        self._alive = False

    def is_alive(self):
        return self._alive


class _FakeKeyboard:
    Key = _FakeKey
    Listener = _FakeListener


@pytest.fixture
def teleop():
    config = TGArm620KeyboardConfig(joint_step_rad=0.05, gripper_step=10.0)
    with (
        patch(
            "lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.require_package",
            return_value=None,
        ),
        patch("lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.PYNPUT_AVAILABLE", True),
        patch("lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.keyboard", _FakeKeyboard),
    ):
        device = TGArm620Keyboard(config)
        device.connect()
        try:
            yield device
        finally:
            if device.is_connected:
                device.disconnect()


def test_action_and_feedback_features_have_expected_keys(teleop):
    expected_keys = {f"joint{i}.pos" for i in range(1, 7)} | {"gripper.pos"}
    assert set(teleop.action_features) == expected_keys
    assert set(teleop.feedback_features) == expected_keys


def test_make_teleoperator_from_config_returns_tg_arm620_keyboard():
    config = TGArm620KeyboardConfig()
    with patch(
        "lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.require_package",
        return_value=None,
    ):
        device = make_teleoperator_from_config(config)
    assert isinstance(device, TGArm620Keyboard)


def test_get_action_requires_connection():
    with (
        patch(
            "lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.require_package",
            return_value=None,
        ),
        patch("lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.PYNPUT_AVAILABLE", True),
        patch("lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.keyboard", _FakeKeyboard),
    ):
        device = TGArm620Keyboard(TGArm620KeyboardConfig())
        with pytest.raises(DeviceNotConnectedError):
            _ = device.get_action()


def test_joint_increment_hold_release_and_clamp(teleop):
    teleop._event_queue.put(("1", True))
    action_1 = teleop.get_action()
    assert action_1["joint1.pos"] == pytest.approx(0.05)

    action_2 = teleop.get_action()
    assert action_2["joint1.pos"] == pytest.approx(0.10)

    teleop._event_queue.put(("1", False))
    action_3 = teleop.get_action()
    assert action_3["joint1.pos"] == pytest.approx(0.10)

    teleop._target_action["joint1.pos"] = 2.96
    teleop._event_queue.put(("1", True))
    action_4 = teleop.get_action()
    assert action_4["joint1.pos"] == pytest.approx(2.967)


def test_gripper_control_clamp_and_home_key(teleop):
    teleop.send_feedback(
        {
            "joint1.pos": 0.2,
            "joint2.pos": 0.1,
            "joint3.pos": 0.0,
            "joint4.pos": -0.1,
            "joint5.pos": 0.3,
            "joint6.pos": -0.2,
            "gripper.pos": 30.0,
        }
    )

    teleop._event_queue.put(("1", True))
    moved = teleop.get_action()
    assert moved["joint1.pos"] == pytest.approx(0.25)

    teleop._event_queue.put(("1", False))
    teleop._event_queue.put(("h", True))
    home = teleop.get_action()
    assert home["joint1.pos"] == pytest.approx(0.2)
    assert home["gripper.pos"] == pytest.approx(30.0)

    teleop._event_queue.put(("h", False))
    teleop._event_queue.put(("p", True))
    grip_open = teleop.get_action()
    assert grip_open["gripper.pos"] == pytest.approx(40.0)

    teleop._target_action["gripper.pos"] = 95.0
    grip_max = teleop.get_action()
    assert grip_max["gripper.pos"] == pytest.approx(100.0)


def test_connect_raises_if_pynput_is_unavailable():
    with (
        patch(
            "lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.require_package",
            return_value=None,
        ),
        patch("lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.PYNPUT_AVAILABLE", False),
        patch("lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.keyboard", None),
    ):
        device = TGArm620Keyboard(TGArm620KeyboardConfig(input_backend="pynput"))
        with pytest.raises(RuntimeError, match="pynput backend is unavailable"):
            device.connect()


def test_stdin_backend_applies_single_step_from_terminal_char():
    config = TGArm620KeyboardConfig(input_backend="stdin", joint_step_rad=0.05)
    with (
        patch(
            "lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.require_package",
            return_value=None,
        ),
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdin.fileno", return_value=0),
        patch("lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.termios.tcgetattr", return_value=[0]),
        patch("lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.termios.tcsetattr", return_value=None),
        patch("lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.fcntl.fcntl", side_effect=[0, 0, 0]),
        patch("lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.tty.setcbreak", return_value=None),
        patch(
            "lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.select.select",
            side_effect=[([0], [], []), ([], [], [])],
        ),
        patch("lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.os.read", return_value=b"1"),
    ):
        device = TGArm620Keyboard(config)
        device.connect()
        try:
            action = device.get_action()
            assert action["joint1.pos"] == pytest.approx(0.05)
        finally:
            if device.is_connected:
                device.disconnect()


def test_auto_backend_prefers_stdin_on_wayland():
    with (
        patch(
            "lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.require_package",
            return_value=None,
        ),
        patch("lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.PYNPUT_AVAILABLE", True),
        patch("lerobot.teleoperators.tg_arm620_keyboard.tg_arm620_keyboard.keyboard", _FakeKeyboard),
        patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}, clear=False),
        patch.object(TGArm620Keyboard, "_setup_stdin_backend", return_value=None),
        patch.object(TGArm620Keyboard, "_teardown_stdin_backend", return_value=None),
    ):
        device = TGArm620Keyboard(TGArm620KeyboardConfig(input_backend="auto"))
        device.connect()
        try:
            assert device._input_backend == "stdin"
        finally:
            if device.is_connected:
                device.disconnect()
