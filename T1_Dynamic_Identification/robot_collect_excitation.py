#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


ROOT = Path(__file__).resolve().parent
CONTROL_DT = 0.002
B1_JOINT_CNT = 29
UPPER_BODY_INDICES = list(range(17))
LOWER_BODY_INDICES = list(range(17, 29))
ARM_INDICES = {
    "left": list(range(2, 9)),
    "right": list(range(9, 16)),
}

# Identical to t1_robot_controller2_walking_upperbody.py.
KP = np.asarray([
    5.0, 5.0,
    100.0, 100.0, 100.0, 100.0, 80.0, 100.0, 80.0,
    100.0, 100.0, 100.0, 100.0, 80.0, 100.0, 80.0,
    100.0,
    350.0, 350.0, 180.0, 350.0, 400.0, 400.0,
    350.0, 350.0, 180.0, 350.0, 400.0, 400.0,
])
KD = np.asarray([
    0.1, 0.1,
    1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
    1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
    5.0,
    2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
    2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
])
FIXED_HOME_POSITION = np.asarray([
    0.00, 0.80,
    0.1207, -1.3649, -0.0025, -1.5215, -0.2081, -0.1417, 0.0086,
    0.1207, 1.3649, -0.0025, 1.5215, -0.2081, 0.1417, 0.0086,
    0.0,
    -0.1, 0.0, 0.0, 0.2, 0.104, 0.098,
    -0.1, 0.0, 0.0, 0.2, 0.104, 0.098,
])
OTHER_ARM_HOLD = {
    "left": FIXED_HOME_POSITION[ARM_INDICES["left"]].copy(),
    "right": FIXED_HOME_POSITION[ARM_INDICES["right"]].copy(),
}

MAX_JOINT_VELOCITY = 0.4
HANDOFF_MAX_ERROR_RAD = 0.15
HANDOFF_MAX_VELOCITY_RAD_S = 0.5
START_MAX_ERROR_RAD = 0.10
START_MAX_VELOCITY_RAD_S = 0.10
LOW_STATE_TIMEOUT_S = 0.2
WEIGHT_RAMP_RATE = 0.2  # Same as teleop: 0 -> 1 in about 5 s.
DEFAULT_HANDOFF_SEC = 5.0


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass
class RobotState:
    wall_time: float
    monotonic_time: float
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau_est: np.ndarray


class StateBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Optional[RobotState] = None
        self._capture = False
        self._history: List[RobotState] = []

    def set(self, state: RobotState) -> None:
        with self._lock:
            self._latest = state
            if self._capture:
                self._history.append(state)

    def get(self) -> Optional[RobotState]:
        with self._lock:
            return self._latest

    def begin_capture(self) -> None:
        with self._lock:
            self._history = [] if self._latest is None else [self._latest]
            self._capture = True

    def end_capture(self) -> List[RobotState]:
        with self._lock:
            self._capture = False
            history = self._history
            self._history = []
            return history


class QuinticTrajectory:
    """Piecewise quintic interpolation preserving q, qd and qdd at knots."""

    def __init__(self, t: np.ndarray, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray):
        t = np.asarray(t, dtype=float).reshape(-1)
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)
        if len(t) < 2:
            raise ValueError("trajectory must contain at least two samples")
        if q.shape != (len(t), 7) or qd.shape != q.shape or qdd.shape != q.shape:
            raise ValueError("t must be N and q/qd/qdd must all be N x 7")
        if not all(np.all(np.isfinite(x)) for x in (t, q, qd, qdd)):
            raise ValueError("trajectory contains NaN or infinity")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("trajectory time must be strictly increasing")

        self.t = t - t[0]
        self.duration = float(self.t[-1])
        h = np.diff(self.t)[:, None]
        c0 = q[:-1]
        c1 = qd[:-1] * h
        c2 = 0.5 * qdd[:-1] * h * h
        delta_q = q[1:] - (c0 + c1 + c2)
        delta_v = qd[1:] * h - (c1 + 2.0 * c2)
        delta_a = qdd[1:] * h * h - 2.0 * c2
        c3 = 10.0 * delta_q - 4.0 * delta_v + 0.5 * delta_a
        c4 = -15.0 * delta_q + 7.0 * delta_v - delta_a
        c5 = 6.0 * delta_q - 3.0 * delta_v + 0.5 * delta_a
        self._h = h[:, 0]
        self._coeff = (c0, c1, c2, c3, c4, c5)

    def evaluate(self, query_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = float(np.clip(query_time, 0.0, self.duration))
        index = min(
            int(np.searchsorted(self.t, query, side="right") - 1),
            len(self._h) - 1,
        )
        index = max(index, 0)
        u = (query - self.t[index]) / self._h[index]
        c0, c1, c2, c3, c4, c5 = (c[index] for c in self._coeff)
        q = c0 + u * (c1 + u * (c2 + u * (c3 + u * (c4 + u * c5))))
        du = c1 + u * (
            2.0 * c2 + u * (3.0 * c3 + u * (4.0 * c4 + u * 5.0 * c5))
        )
        ddu = 2.0 * c2 + u * (6.0 * c3 + u * (12.0 * c4 + u * 20.0 * c5))
        return q, du / self._h[index], ddu / (self._h[index] ** 2)


class T1ExcitationCollector:
    def __init__(self, network_interface: str = "", simulation: bool = False):
        self.network_interface = network_interface
        self.simulation = simulation
        self.state_buffer = StateBuffer()
        self.current_jpos_des = FIXED_HOME_POSITION.copy()
        # q_target mirrors the working teleop controller.  This collector uses
        # synchronous publication instead of a background publishing thread, but
        # keeping q_target makes the handoff sequence identical in intent:
        # enable UpperBodyCustomControl, initialize current_jpos_des/q_target
        # from measured q, enable publication, then ramp weight from 0 to 1.
        self.q_target = FIXED_HOME_POSITION.copy()
        self.handoff_q: Optional[np.ndarray] = None
        self._publish_enabled = False
        self.weight = 0.0
        self.weight_rate = WEIGHT_RAMP_RATE
        self.weight_margin = self.weight_rate * CONTROL_DT
        self.custom_control_enabled = False
        self._publisher = None
        self._subscriber = None
        self._loco_client = None
        self._motor_cmds = None
        self._LowCmd = None
        self._LowCmdType = None
        self._B1LocoApiId = None
        self._consecutive_write_failures = 0

        if simulation:
            zeros = np.zeros(B1_JOINT_CNT)
            self.state_buffer.set(RobotState(
                time.time(), time.perf_counter(), FIXED_HOME_POSITION.copy(),
                zeros.copy(), zeros.copy(), zeros.copy(),
            ))
        else:
            self._init_sdk()

    def _init_sdk(self) -> None:
        from booster_robotics_sdk_python import (
            B1LocoClient,
            B1LowCmdPublisher,
            B1LowStateSubscriber,
            ChannelFactory,
            LowCmd,
            LowCmdType,
            MotorCmd,
        )
        try:
            from booster_robotics_sdk_python import B1LocoApiId
        except Exception:
            B1LocoApiId = None

        factory = ChannelFactory.Instance()
        print(
            f"[T1] ChannelFactory.Init(domain=0, "
            f"network='{self.network_interface or '<default>'}')"
        )
        try:
            if self.network_interface:
                factory.Init(0, self.network_interface)
            else:
                factory.Init(0)
        except TypeError:
            factory.Init(0)

        self._LowCmd = LowCmd
        self._LowCmdType = LowCmdType
        self._B1LocoApiId = B1LocoApiId
        self._motor_cmds = [MotorCmd() for _ in range(B1_JOINT_CNT)]
        self._publisher = B1LowCmdPublisher()
        self._publisher.InitChannel()
        self._subscriber = B1LowStateSubscriber(handler=self._on_state)
        self._subscriber.InitChannel()
        self._loco_client = B1LocoClient()
        self._loco_client.Init()

        print("[T1] waiting for rt/low_state ...")
        deadline = time.perf_counter() + 5.0
        while self.state_buffer.get() is None:
            if time.perf_counter() >= deadline:
                raise RuntimeError("Timed out waiting for rt/low_state")
            time.sleep(0.01)
        print("[T1] rt/low_state received.")

    @staticmethod
    def _motor_state_array(state: Any):
        if hasattr(state, "motor_state_parallel"):
            return state.motor_state_parallel
        if hasattr(state, "motor_state_serial"):
            return state.motor_state_serial
        if hasattr(state, "motor_state"):
            return state.motor_state
        raise AttributeError("LowState has no motor state array")

    def _on_state(self, state: Any) -> None:
        received_wall = time.time()
        received_monotonic = time.perf_counter()
        motors = self._motor_state_array(state)
        if len(motors) < B1_JOINT_CNT:
            return
        q = np.empty(B1_JOINT_CNT)
        dq = np.empty(B1_JOINT_CNT)
        ddq = np.empty(B1_JOINT_CNT)
        tau = np.empty(B1_JOINT_CNT)
        for i in range(B1_JOINT_CNT):
            motor = motors[i]
            q[i] = float(getattr(motor, "q", 0.0))
            dq[i] = float(getattr(motor, "dq", 0.0))
            ddq[i] = float(getattr(motor, "ddq", 0.0))
            tau[i] = float(getattr(motor, "tau_est", 0.0))
        self.state_buffer.set(RobotState(
            received_wall, received_monotonic, q, dq, ddq, tau,
        ))

    def get_state(self, require_fresh: bool = True) -> RobotState:
        state = self.state_buffer.get()
        if state is None:
            raise RuntimeError("No low_state has been received")
        age = time.perf_counter() - state.monotonic_time
        if require_fresh and not self.simulation and age > LOW_STATE_TIMEOUT_S:
            raise RuntimeError(f"low_state is stale ({age:.3f} s)")
        return state

    def _call_upper_body_custom_control(self, start: bool) -> bool:
        if self.simulation:
            self.custom_control_enabled = bool(start)
            return True
        try:
            if hasattr(self._loco_client, "UpperBodyCustomControl"):
                result = self._loco_client.UpperBodyCustomControl(bool(start))
            else:
                api_id = (
                    getattr(self._B1LocoApiId, "kUpperBodyCustomControl", None)
                    if self._B1LocoApiId is not None else None
                )
                if api_id is None or not hasattr(self._loco_client, "SendApiRequest"):
                    print("[ERROR] UpperBodyCustomControl is unavailable in this SDK")
                    return False
                result = self._loco_client.SendApiRequest(
                    api_id, json.dumps({"start": bool(start)})
                )
        except Exception as exc:
            print(f"[ERROR] UpperBodyCustomControl({start}) raised: {exc}")
            return False

        if result not in (None, 0):
            print(f"[ERROR] UpperBodyCustomControl({start}) failed: {result}")
            return False
        self.custom_control_enabled = bool(start)
        print(f"[T1] UpperBodyCustomControl({start})")
        return True

    def _publish(self, q_command: np.ndarray) -> None:
        if not self._publish_enabled:
            return
        q_command = np.asarray(q_command, dtype=float).reshape(B1_JOINT_CNT).copy()
        state = self.get_state()
        q_command[LOWER_BODY_INDICES] = state.q[LOWER_BODY_INDICES]
        self.current_jpos_des = q_command

        if self.simulation:
            zeros = np.zeros(B1_JOINT_CNT)
            self.state_buffer.set(RobotState(
                time.time(), time.perf_counter(), q_command.copy(),
                zeros.copy(), zeros.copy(), zeros.copy(),
            ))
            return

        for index in range(B1_JOINT_CNT):
            motor = self._motor_cmds[index]
            motor.q = float(q_command[index])
            motor.dq = 0.0
            motor.tau = 0.0
            if index in LOWER_BODY_INDICES:
                motor.kp = 0.0
                motor.kd = 0.0
                motor.weight = 0.0
            else:
                motor.kp = float(KP[index])
                motor.kd = float(KD[index])
                motor.weight = float(self.weight)

        # Recreate LowCmd on every cycle, exactly like the working teleop loop.
        low_cmd = self._LowCmd()
        low_cmd.cmd_type = self._LowCmdType.PARALLEL
        low_cmd.motor_cmd = self._motor_cmds
        result = self._publisher.Write(low_cmd)
        if result is False:
            self._consecutive_write_failures += 1
            if self._consecutive_write_failures >= 5:
                raise RuntimeError("LowCmd Write() failed five consecutive times")
        else:
            self._consecutive_write_failures = 0

    @staticmethod
    def _sleep_cycle(cycle_start: float) -> None:
        remaining = CONTROL_DT - (time.perf_counter() - cycle_start)
        if remaining > 0.0:
            time.sleep(remaining)

    def start_control(self, handoff_sec: float = DEFAULT_HANDOFF_SEC) -> None:
        """Start upper-body control using the same handoff order as teleop.

        The caller must put the robot in walking mode (mw) before running this
        function.  This function only opens UpperBodyCustomControl and takes
        over upper-body joints while locomotion continues to own the legs.
        """
        # 1) First open walking-mode upper-body custom control.
        if not self._call_upper_body_custom_control(True):
            self.weight = 0.0
            self._publish_enabled = False
            raise RuntimeError("Failed to enable UpperBodyCustomControl")

        try:
            # 2) Then initialize current_jpos_des/q_target from measured joints
            # to avoid any position jump at handoff.
            handoff_q = self.get_state().q.copy()
            self.handoff_q = handoff_q.copy()
            self.current_jpos_des = handoff_q.copy()
            self.q_target = handoff_q.copy()

            # 3) Enable publication, starting at zero weight.
            self.weight = 0.0
            self._publish_enabled = True

            # 4) Publish the measured handoff pose while ramping weight 0 -> 1.
            # With WEIGHT_RAMP_RATE = 0.2 and CONTROL_DT = 0.002, this takes
            # about 5 seconds, matching the working teleop controller.
            min_steps = max(1, int(handoff_sec / CONTROL_DT))
            steps = 0
            while self.weight < 1.0 or steps < min_steps:
                cycle_start = time.perf_counter()
                if self.weight < 1.0:
                    self.weight = min(1.0, self.weight + self.weight_margin)
                self._publish(self.q_target)
                steps += 1
                self._sleep_cycle(cycle_start)

            state = self.get_state()
            upper = np.asarray(UPPER_BODY_INDICES)
            max_error = float(np.max(np.abs(state.q[upper] - handoff_q[upper])))
            max_velocity = float(np.max(np.abs(state.dq[upper])))
            print(
                f"[T1] handoff check: max_error={max_error:.4f} rad, "
                f"max_velocity={max_velocity:.4f} rad/s"
            )
            if (
                max_error > HANDOFF_MAX_ERROR_RAD
                or max_velocity > HANDOFF_MAX_VELOCITY_RAD_S
            ):
                raise RuntimeError(
                    "Upper-body handoff is unstable; trajectory will not start"
                )
        except Exception:
            self._publish_enabled = False
            self._call_upper_body_custom_control(False)
            self.weight = 0.0
            raise

    def _return_to_handoff_pose(self, timeout_sec: float = 10.0) -> None:
        """Move upper body back near the recorded handoff pose before release.

        This reduces the jump that can happen when UpperBodyCustomControl(False)
        returns authority to the walking controller, whose internal upper-body
        reference is usually close to the posture held when mw was entered.
        """
        if self.handoff_q is None:
            print("[WARN] no handoff_q recorded; skip return-to-handoff motion.")
            return

        target = self.handoff_q.copy()
        initial_state = self.get_state().q.copy()
        upper = np.asarray(UPPER_BODY_INDICES)
        required_motion = float(np.max(np.abs(target[upper] - initial_state[upper])))
        effective_timeout = max(timeout_sec, required_motion / MAX_JOINT_VELOCITY + 3.0)
        deadline = time.perf_counter() + effective_timeout
        stable_cycles = 0

        print(
            "[T1] returning upper body to handoff pose before disabling "
            "UpperBodyCustomControl ..."
        )
        while True:
            cycle_start = time.perf_counter()
            # Legs are still owned by locomotion.  Keep their command slots
            # synced to the measured state; _publish will also enforce zero
            # lower-body gains/weight.
            target[LOWER_BODY_INDICES] = self.get_state().q[LOWER_BODY_INDICES]
            command = self._clip_upper_target(target)
            self._publish(command)

            state = self.get_state()
            command_remaining = float(
                np.max(np.abs(target[upper] - self.current_jpos_des[upper]))
            )
            measured_error = float(np.max(np.abs(state.q[upper] - target[upper])))
            measured_velocity = float(np.max(np.abs(state.dq[upper])))

            if (
                command_remaining < 1e-4
                and measured_error <= HANDOFF_MAX_ERROR_RAD
                and measured_velocity <= HANDOFF_MAX_VELOCITY_RAD_S
            ):
                stable_cycles += 1
            else:
                stable_cycles = 0

            if stable_cycles >= 100:
                print(
                    f"[T1] handoff pose reached before release: "
                    f"max_error={measured_error:.4f} rad, "
                    f"max_velocity={measured_velocity:.4f} rad/s"
                )
                return

            if time.perf_counter() >= deadline:
                print(
                    "[WARN] return-to-handoff timeout: "
                    f"max_error={measured_error:.4f} rad, "
                    f"max_velocity={measured_velocity:.4f} rad/s; "
                    "UpperBodyCustomControl will still be disabled for safety."
                )
                return

            self._sleep_cycle(cycle_start)

    def stop_control(self) -> None:
        if not self.custom_control_enabled:
            self.weight = 0.0
            self._publish_enabled = False
            return

        # Best effort: first move back to the posture recorded at the mw/custom
        # control handoff, then ramp weight down and close UpperBodyCustomControl.
        self._publish_enabled = True
        if self.weight <= 0.0:
            self.weight = 1.0

        try:
            self._return_to_handoff_pose()
        except Exception as exc:
            print(f"[WARN] return-to-handoff failed: {exc}")

        while self.weight > 0.0:
            cycle_start = time.perf_counter()
            self.weight = max(0.0, self.weight - self.weight_margin)
            self._publish(self.current_jpos_des)
            self._sleep_cycle(cycle_start)

        self._publish_enabled = False
        self._call_upper_body_custom_control(False)
        self.weight = 0.0

    def make_target(
        self,
        side: str,
        active_arm_q: np.ndarray,
        other_arm_q: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        target = self.get_state().q.copy()
        target[0] = FIXED_HOME_POSITION[0]
        target[1] = FIXED_HOME_POSITION[1]
        # Keep waist at the current measured angle, like the teleop controller.
        # Do not force target[16] to FIXED_HOME_POSITION[16] while walking.
        target[ARM_INDICES[side]] = np.asarray(active_arm_q, dtype=float).reshape(7)
        other = "right" if side == "left" else "left"
        if other_arm_q is None:
            other_arm_q = OTHER_ARM_HOLD[other]
        target[ARM_INDICES[other]] = np.asarray(other_arm_q, dtype=float).reshape(7)
        return target

    def _clip_upper_target(self, target: np.ndarray) -> np.ndarray:
        command = self.current_jpos_des.copy()
        upper = np.asarray(UPPER_BODY_INDICES)
        delta = np.asarray(target, dtype=float)[upper] - command[upper]
        scale = float(np.max(np.abs(delta))) / (MAX_JOINT_VELOCITY * CONTROL_DT)
        command[upper] += delta / max(scale, 1.0)
        command[LOWER_BODY_INDICES] = self.get_state().q[LOWER_BODY_INDICES]
        return command

    def move_to_trajectory_start(
        self,
        side: str,
        start_q: np.ndarray,
        other_arm_q: np.ndarray,
        timeout_sec: float = 10.0,
    ) -> None:
        print("[T1] moving to trajectory start with 0.4 rad/s rate limit ...")
        print("[T1] this transition is before data capture; recorded excitation is not rate-limited.")
        print(f"[T1] start-pose readiness check uses the active {side} arm only.")
        watchdog_deadline = time.perf_counter() + 1.0
        initial_state = self.get_state().q.copy()
        initial_target = self.make_target(side, start_q, other_arm_q)
        check = np.asarray(ARM_INDICES[side])
        required_motion = float(np.max(np.abs(
            initial_target[check] - initial_state[check]
        )))
        # Ensure there is enough time for the rate-limited active-arm transition
        # from the current measured posture to the excitation initial posture.
        transition_timeout = required_motion / MAX_JOINT_VELOCITY + 3.0
        effective_timeout = max(timeout_sec, transition_timeout)
        deadline = time.perf_counter() + effective_timeout
        stable_cycles = 0
        while True:
            cycle_start = time.perf_counter()
            target = self.make_target(side, start_q, other_arm_q)
            command = self._clip_upper_target(target)
            self._publish(command)

            state = self.get_state()
            command_remaining = float(
                np.max(np.abs(target[check] - self.current_jpos_des[check]))
            )
            measured_error = float(
                np.max(np.abs(state.q[check] - target[check]))
            )
            measured_velocity = float(np.max(np.abs(state.dq[check])))
            if (
                command_remaining < 1e-4
                and measured_error <= START_MAX_ERROR_RAD
                and measured_velocity <= START_MAX_VELOCITY_RAD_S
            ):
                stable_cycles += 1
            else:
                stable_cycles = 0

            if stable_cycles >= 100:
                print(
                    f"[T1] active-arm start pose reached: "
                    f"max_error={measured_error:.4f} rad, "
                    f"max_velocity={measured_velocity:.4f} rad/s"
                )
                return
            if (
                time.perf_counter() >= watchdog_deadline
                and required_motion > 0.10
            ):
                measured_motion = float(np.max(np.abs(
                    state.q[check] - initial_state[check]
                )))
                if measured_motion < 0.02:
                    raise RuntimeError(
                        "Active-arm LowCmd is not taking effect: target changed by "
                        f"{required_motion:.3f} rad but measured motion after 1 s "
                        f"is only {measured_motion:.3f} rad"
                    )
                watchdog_deadline = float("inf")
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    "Active arm did not reach the trajectory start pose within "
                    f"{effective_timeout:.1f} s; refusing to run excitation"
                )
            self._sleep_cycle(cycle_start)

    @staticmethod
    def _control_grid(duration: float) -> np.ndarray:
        count = int(np.floor(duration / CONTROL_DT + 1e-12))
        grid = np.arange(count + 1, dtype=float) * CONTROL_DT
        if duration - grid[-1] > 1e-9:
            grid = np.append(grid, duration)
        else:
            grid[-1] = duration
        return grid

    @staticmethod
    def _aligned_state_arrays(
        history: List[RobotState],
        start_monotonic: float,
        control_times: np.ndarray,
    ):
        state_times = np.asarray([state.monotonic_time for state in history])
        valid = np.isfinite(state_times) & (state_times > 0.0)
        history = [state for state, keep in zip(history, valid) if keep]
        state_times = state_times[valid]
        if len(state_times) < 2:
            raise RuntimeError("Fewer than two low_state samples were captured")

        order = np.argsort(state_times)
        state_times = state_times[order]
        history = [history[i] for i in order]
        keep = np.r_[True, np.diff(state_times) > 1e-9]
        state_times = state_times[keep]
        history = [state for state, use in zip(history, keep) if use]
        if len(state_times) < 2:
            raise RuntimeError("low_state timestamps did not advance")

        relative_times = state_times - start_monotonic
        if relative_times[0] > 0.0 or relative_times[-1] < control_times[-1]:
            print(
                "[WARN] low_state does not fully bracket the trajectory: "
                f"[{relative_times[0]:.6f}, {relative_times[-1]:.6f}] s"
            )

        def resample(field: str) -> np.ndarray:
            values = np.stack([
                np.asarray(getattr(state, field), dtype=float) for state in history
            ])
            return np.column_stack([
                np.interp(control_times, relative_times, values[:, joint])
                for joint in range(B1_JOINT_CNT)
            ])

        wall_time = np.interp(
            control_times,
            relative_times,
            np.asarray([state.wall_time for state in history]),
        )
        return (
            wall_time,
            resample("q"),
            resample("dq"),
            resample("ddq"),
            resample("tau_est"),
        )

    def execute_trajectory(
        self,
        side: str,
        trajectory: QuinticTrajectory,
        csv_path: Path,
    ) -> Path:
        other = "right" if side == "left" else "left"
        other_arm_q = OTHER_ARM_HOLD[other]
        start_q, _, _ = trajectory.evaluate(0.0)
        self.move_to_trajectory_start(side, start_q, other_arm_q)

        control_times = self._control_grid(trajectory.duration)
        count = len(control_times)
        q_ref = np.empty((count, 7))
        qd_ref = np.empty_like(q_ref)
        qdd_ref = np.empty_like(q_ref)
        for k, query in enumerate(control_times):
            q_ref[k], qd_ref[k], qdd_ref[k] = trajectory.evaluate(float(query))

        publish_wall_time = np.full(count, np.nan)
        publish_lateness = np.full(count, np.nan)
        # Kept for CSV compatibility with earlier logs.  During the recorded
        # excitation segment we no longer rate-limit the command, so this value
        # should stay exactly zero for every published sample.
        clip_error = np.zeros(count, dtype=float)
        missed_deadlines = 0

        self.state_buffer.begin_capture()
        start_monotonic = time.perf_counter()
        try:
            k = 0
            while k < count:
                target_time = start_monotonic + float(control_times[k])
                remaining = target_time - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)

                published_monotonic = time.perf_counter()
                if k < count - 1:
                    due_index = int(np.searchsorted(
                        control_times,
                        min(
                            published_monotonic - start_monotonic,
                            trajectory.duration,
                        ),
                        side="right",
                    ) - 1)
                    if due_index > k:
                        missed_deadlines += due_index - k
                        k = due_index
                        target_time = start_monotonic + float(control_times[k])

                # Identification input: publish the generated reference
                # directly.  Do not call _clip_upper_target() here; otherwise
                # the actual commanded arm trajectory may differ from q_ref.
                target = self.make_target(side, q_ref[k], other_arm_q)
                self._publish(target)
                publish_wall_time[k] = time.time()
                publish_lateness[k] = published_monotonic - target_time
                k += 1

            final_target = self.make_target(side, q_ref[-1], other_arm_q)
            for _ in range(2):
                cycle_start = time.perf_counter()
                self._publish(final_target)
                self._sleep_cycle(cycle_start)
        finally:
            history = self.state_buffer.end_capture()

        wall_time, q_meas, dq_meas, ddq_meas, tau_est = (
            self._aligned_state_arrays(history, start_monotonic, control_times)
        )
        arm = ARM_INDICES[side]
        q_meas = q_meas[:, arm]
        dq_meas = dq_meas[:, arm]
        ddq_meas = ddq_meas[:, arm]
        tau_est = tau_est[:, arm]

        max_lateness = float(np.nanmax(publish_lateness))
        max_clip_error = float(np.nanmax(clip_error))
        tracking_rmse = np.sqrt(np.mean((q_meas - q_ref) ** 2, axis=0))
        print(
            f"[T1] published {count - missed_deadlines}/{count} points on the "
            f"{1.0 / CONTROL_DT:.1f} Hz grid; "
            f"max_lateness={max_lateness * 1e3:.3f} ms"
        )
        print(
            "[T1] tracking RMSE rad: "
            + np.array2string(tracking_rmse, precision=4, separator=", ")
        )
        if missed_deadlines:
            print(
                f"[WARN] missed {missed_deadlines} control deadlines; "
                "do not use this run for identification"
            )
        if max_clip_error > 1e-12:
            print(
                f"[WARN] command/reference mismatch max={max_clip_error:.6e} rad; "
                "do not use this run for identification"
            )
        else:
            print(
                "[T1] recorded excitation command path has no rate limiter: "
                "active-arm q_command == q_ref on the 500 Hz grid."
            )

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "wall_time", "traj_time", "publish_wall_time",
            "publish_lateness_s", "command_published",
            "command_clip_error_rad",
        ]
        for prefix in [
            "q_ref", "qd_ref", "qdd_ref",
            "q_meas", "qd_meas", "ddq_meas", "tau_est",
        ]:
            header += [f"{prefix}_{i}" for i in range(7)]

        with csv_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(header)
            for k, query in enumerate(control_times):
                row = [
                    wall_time[k],
                    float(query),
                    publish_wall_time[k],
                    publish_lateness[k],
                    float(np.isfinite(publish_wall_time[k])),
                    clip_error[k],
                ]
                row += q_ref[k].tolist()
                row += qd_ref[k].tolist()
                row += qdd_ref[k].tolist()
                row += q_meas[k].tolist()
                row += dq_meas[k].tolist()
                row += ddq_meas[k].tolist()
                row += tau_est[k].tolist()
                writer.writerow(row)
        return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect an excitation trajectory in walking mode. Before running "
            "this script, put the robot into mw walking mode with change_mode.py."
        )
    )
    parser.add_argument(
        "network_interface",
        nargs="?",
        default=None,
        help="SDK network interface, e.g. enp6s0. Required unless --simulation.",
    )
    parser.add_argument(
        "--traj", default="trajectories/left_arm_payload_excitation.npz"
    )
    parser.add_argument("--side", choices=["left", "right"], default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tag", default="empty_id_A")
    parser.add_argument("--simulation", action="store_true")
    args = parser.parse_args()

    if not args.simulation and not args.network_interface:
        parser.error(
            "network_interface is required, e.g. "
            "python3 robot_collect_excitation.py enp6s0"
        )

    trajectory_path = resolve_path(args.traj)
    raw = np.load(trajectory_path, allow_pickle=True)
    side = args.side
    if side is None and "side" in raw.files:
        side = str(np.asarray(raw["side"]).item())
    if side is None:
        side = "left"
    trajectory = QuinticTrajectory(raw["t"], raw["q"], raw["qd"], raw["qdd"])
    output_path = resolve_path(args.out or f"logs/{args.tag}_{side}.csv")

    robot = T1ExcitationCollector(
        network_interface=args.network_interface or "",
        simulation=args.simulation,
    )

    print(
        f"[INFO] executing {side} arm excitation from {trajectory_path}; "
        f"output={output_path}"
    )
    print(
        "[INFO] assuming robot is already in mw walking mode; "
        "UpperBodyCustomControl will be enabled now."
    )
    started = False
    try:
        robot.start_control(handoff_sec=DEFAULT_HANDOFF_SEC)
        started = True
        result = robot.execute_trajectory(side, trajectory, output_path)
        print(f"[OK] saved {result}")
    finally:
        if started:
            robot.stop_control()


if __name__ == "__main__":
    main()
