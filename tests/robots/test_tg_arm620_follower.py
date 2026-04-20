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

import json
from unittest.mock import patch

import pytest

from lerobot.robots.tg_arm620 import TGArm620Config, TGArm620Follower


class FakeAgain(Exception):
    pass


class FakeSocket:
    def __init__(self):
        self.recv_queue: list[str] = []
        self.sent_queue: list[str] = []
        self.closed = False

    def setsockopt(self, *_args, **_kwargs):
        return

    def setsockopt_string(self, *_args, **_kwargs):
        return

    def connect(self, *_args, **_kwargs):
        return

    def send_string(self, msg: str, flags: int = 0):
        del flags
        self.sent_queue.append(msg)

    def recv_string(self, flags: int = 0) -> str:
        del flags
        if not self.recv_queue:
            raise FakeAgain
        return self.recv_queue.pop(0)

    def close(self):
        self.closed = True


class FakePoller:
    def __init__(self):
        self._registered: list[FakeSocket] = []

    def register(self, sock: FakeSocket, _pollin: int):
        self._registered.append(sock)

    def poll(self, _timeout_ms: int):
        for sock in self._registered:
            if sock.recv_queue:
                return [(sock, 1)]
        return []


class FakeContext:
    def __init__(self, initial_state: str):
        self._initial_state = initial_state
        self.cmd_socket: FakeSocket | None = None
        self.state_socket: FakeSocket | None = None
        self._state_seeded = False
        self.terminated = False

    def socket(self, socket_type: int) -> FakeSocket:
        sock = FakeSocket()
        if socket_type == FakeZMQ.PUSH:
            self.cmd_socket = sock
        elif socket_type == FakeZMQ.SUB:
            self.state_socket = sock
            if not self._state_seeded:
                sock.recv_queue.append(self._initial_state)
                self._state_seeded = True
        return sock

    def term(self):
        self.terminated = True


class FakeZMQ:
    PUSH = 1
    SUB = 2
    POLLIN = 3
    NOBLOCK = 4
    CONFLATE = 5
    LINGER = 6
    SUBSCRIBE = 7
    Again = FakeAgain

    def __init__(self, initial_state: str):
        self._context = FakeContext(initial_state)

    def Context(self) -> FakeContext:
        return self._context

    def Poller(self) -> FakePoller:
        return FakePoller()


@pytest.fixture
def follower_and_fake_zmq():
    initial_state = json.dumps(
        {
            "joint1.pos": 0.1,
            "joint2.pos": 0.2,
            "joint3.pos": 0.3,
            "joint4.pos": 0.4,
            "joint5.pos": 0.5,
            "joint6.pos": 0.6,
            "gripper.pos": 7.0,
        }
    )
    fake_zmq = FakeZMQ(initial_state)

    with (
        patch("lerobot.robots.tg_arm620.tg_arm620.require_package", return_value=None),
        patch("lerobot.robots.tg_arm620.tg_arm620.make_cameras_from_configs", return_value={}),
        patch("lerobot.robots.tg_arm620.tg_arm620.zmq", fake_zmq),
    ):
        cfg = TGArm620Config(connect_timeout_s=0.5)
        follower = TGArm620Follower(cfg)
        follower.connect()
        try:
            yield follower, fake_zmq
        finally:
            if follower.is_connected:
                follower.disconnect()


def test_features_have_expected_keys(follower_and_fake_zmq):
    follower, _ = follower_and_fake_zmq

    expected_joint_keys = {f"joint{i}.pos" for i in range(1, 7)}
    expected_keys = expected_joint_keys | {"gripper.pos"}
    assert set(follower.action_features) == expected_keys
    assert expected_keys.issubset(set(follower.observation_features))


def test_get_observation_reads_and_caches_state(follower_and_fake_zmq):
    follower, fake_zmq = follower_and_fake_zmq

    # First read comes from connection-seeded packet and cache.
    obs_1 = follower.get_observation()
    assert obs_1["joint1.pos"] == pytest.approx(0.1)
    assert obs_1["joint6.pos"] == pytest.approx(0.6)
    assert obs_1["gripper.pos"] == pytest.approx(7.0)

    # Push a new state packet and ensure values update.
    assert fake_zmq._context.state_socket is not None
    fake_zmq._context.state_socket.recv_queue.append(
        json.dumps(
            {
                "joint1.pos": -0.1,
                "joint2.pos": -0.2,
                "joint3.pos": -0.3,
                "joint4.pos": -0.4,
                "joint5.pos": -0.5,
                "joint6.pos": -0.6,
                "gripper.pos": 13.0,
            }
        )
    )

    obs_2 = follower.get_observation()
    assert obs_2["joint1.pos"] == pytest.approx(-0.1)
    assert obs_2["joint6.pos"] == pytest.approx(-0.6)
    assert obs_2["gripper.pos"] == pytest.approx(13.0)


def test_send_action_clamps_values_before_sending(follower_and_fake_zmq):
    follower, fake_zmq = follower_and_fake_zmq

    sent_action = follower.send_action(
        {
            "joint1.pos": 9.0,
            "joint2.pos": -9.0,
            "joint3.pos": 0.1,
            "joint4.pos": -0.1,
            "joint5.pos": 0.0,
            "joint6.pos": 0.2,
            "gripper.pos": 150.0,
        }
    )

    assert sent_action["joint1.pos"] == pytest.approx(2.967)
    assert sent_action["joint2.pos"] == pytest.approx(-1.5708)
    assert sent_action["gripper.pos"] == pytest.approx(100.0)

    assert fake_zmq._context.cmd_socket is not None
    payload = json.loads(fake_zmq._context.cmd_socket.sent_queue[-1])
    assert payload["joint1.pos"] == pytest.approx(2.967)
    assert payload["joint2.pos"] == pytest.approx(-1.5708)
    assert payload["gripper.pos"] == pytest.approx(100.0)
    assert "timestamp" in payload


def test_disconnect_releases_zmq_resources(follower_and_fake_zmq):
    follower, fake_zmq = follower_and_fake_zmq

    assert follower.is_connected
    follower.disconnect()

    assert not follower.is_connected
    assert fake_zmq._context.terminated
    assert fake_zmq._context.cmd_socket is not None and fake_zmq._context.cmd_socket.closed
    assert fake_zmq._context.state_socket is not None and fake_zmq._context.state_socket.closed
