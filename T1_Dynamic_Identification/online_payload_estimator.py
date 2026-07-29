#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


ROOT = Path(__file__).resolve().parent
CONTROL_DT = 0.002
B1_JOINT_CNT = 29
DEFAULT_NETWORK = "127.0.0.1"
DEFAULT_CONFIG = "configs/t1_payload_config.json"
DEFAULT_MODEL = "models/left_empty_payload_prior_traj1.npz"
DEFAULT_TRAJECTORY = "trajectories/left_arm_payload_excitation1.npz"
DEFAULT_SIDE = "left"
UPPER_BODY_INDICES = list(range(17))
LOWER_BODY_INDICES = list(range(17, 29))
ARM_INDICES = {
    "left": list(range(2, 9)),
    "right": list(range(9, 16)),
}

# These values are intentionally identical to robot_collect_excitation.py and
# t1_robot_controller2_walking_upperbody.py.
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
WEIGHT_RAMP_RATE = 0.2
DEFAULT_HANDOFF_SEC = 5.0


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


MOMENTUM_METHODS = {"momentum_observer", "momentum_direct", "momentum"}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def asvec(value: Any, size: Optional[int] = None, name: str = "array") -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if size is not None and array.size != size:
        raise ValueError(f"{name} must have length {size}, got {array.size}")
    return array


def _sym3(values: np.ndarray) -> np.ndarray:
    x = asvec(values, 6)
    return np.array(
        [[x[0], x[1], x[2]], [x[1], x[3], x[4]], [x[2], x[4], x[5]]],
        dtype=float,
    )


def _vech3(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    matrix = 0.5 * (matrix + matrix.T)
    return np.array([
        matrix[0, 0], matrix[0, 1], matrix[0, 2],
        matrix[1, 1], matrix[1, 2], matrix[2, 2],
    ])


def payload_parameter_map(mode: str) -> np.ndarray:
    """Map the selected payload coordinates to Pinocchio's dyn10 ordering."""
    if mode == "xu10":
        transform = np.zeros((10, 10))
        # xu10=[Ixx,Ixy,Ixz,Iyy,Iyz,Izz,hx,hy,hz,m]
        transform[0, 9] = 1.0
        transform[1, 6] = 1.0
        transform[2, 7] = 1.0
        transform[3, 8] = 1.0
        transform[4, 0] = 1.0
        transform[5, 1] = 1.0
        transform[6, 2] = 1.0
        transform[7, 3] = 1.0
        transform[8, 4] = 1.0
        transform[9, 5] = 1.0
        return transform
    if mode == "pcpdi16":
        transform = np.zeros((10, 16))
        # theta16=[m,h,Q,Ic], Io=Ic+tr(Q)I-Q
        transform[0, 0] = 1.0
        transform[1, 1] = 1.0
        transform[2, 2] = 1.0
        transform[3, 3] = 1.0
        transform[4, 7] = 1.0
        transform[4, 9] = 1.0
        transform[4, 10] = 1.0
        transform[5, 5] = -1.0
        transform[5, 11] = 1.0
        transform[6, 6] = -1.0
        transform[6, 12] = 1.0
        transform[7, 4] = 1.0
        transform[7, 9] = 1.0
        transform[7, 13] = 1.0
        transform[8, 8] = -1.0
        transform[8, 14] = 1.0
        transform[9, 4] = 1.0
        transform[9, 7] = 1.0
        transform[9, 15] = 1.0
        return transform
    raise ValueError(f"unsupported payload parameter mode: {mode}")


def theta16_to_xu10(theta: np.ndarray) -> np.ndarray:
    dyn10 = payload_parameter_map("pcpdi16") @ asvec(theta, 16, "theta16")
    return np.array([
        dyn10[4], dyn10[5], dyn10[6], dyn10[7], dyn10[8],
        dyn10[9], dyn10[1], dyn10[2], dyn10[3], dyn10[0],
    ])


def xu10_to_theta16(xu10: np.ndarray) -> np.ndarray:
    x = asvec(xu10, 10, "xu10")
    mass = float(x[9])
    first_moment = x[6:9].copy()
    inertia_origin = np.array(
        [[x[0], x[1], x[2]], [x[1], x[3], x[4]], [x[2], x[4], x[5]]],
        dtype=float,
    )
    if not np.isfinite(mass) or mass <= 1e-12:
        first_moment[:] = 0.0
        pseudo_moment = np.zeros((3, 3))
    else:
        pseudo_moment = np.outer(first_moment, first_moment) / mass
    inertia_com = inertia_origin - (
        np.trace(pseudo_moment) * np.eye(3) - pseudo_moment
    )
    return np.r_[
        mass, first_moment, _vech3(pseudo_moment), _vech3(inertia_com)
    ]


def _project_psd(matrix: np.ndarray, epsilon: float) -> np.ndarray:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.maximum(values, float(epsilon))) @ vectors.T


def project_theta16(
    theta: np.ndarray,
    mass_min: float,
    mass_max: Optional[float],
) -> np.ndarray:
    """Project only the reported payload; the unconstrained RLS state is retained."""
    theta = asvec(theta, 16, "theta16")
    mass = float(np.nan_to_num(theta[0]))
    mass = max(float(mass_min), mass)
    if mass_max is not None:
        mass = min(float(mass_max), mass)
    first_moment = np.nan_to_num(theta[1:4])
    inertia_com = _project_psd(_sym3(np.nan_to_num(theta[10:16])), 1e-8)
    if mass <= 1e-12:
        first_moment[:] = 0.0
        pseudo_moment = np.zeros((3, 3))
    else:
        pseudo_moment = np.outer(first_moment, first_moment) / mass
    return np.r_[mass, first_moment, _vech3(pseudo_moment), _vech3(inertia_com)]


def payload_report(theta: np.ndarray) -> dict[str, Any]:
    theta = asvec(theta, 16, "theta16")
    mass = float(theta[0])
    first_moment = theta[1:4]
    com = first_moment / mass if mass > 1e-12 else np.zeros(3)
    return {
        "theta16": theta.tolist(),
        "xu10": theta16_to_xu10(theta).tolist(),
        "mass": mass,
        "first_moment": first_moment.tolist(),
        "com": com.tolist(),
        "Q": _sym3(theta[4:10]).tolist(),
        "Ic": _sym3(theta[10:16]).tolist(),
    }


def nonlinear_friction_regressor(
    qd: np.ndarray,
    alpha: np.ndarray,
    sign_smoothing_velocity: float,
) -> np.ndarray:
    qd = asvec(qd, 7, "qd")
    alpha = asvec(alpha, 7, "alpha")
    regressor = np.zeros((7, 21))
    for joint in range(7):
        sign = (
            float(np.tanh(qd[joint] / sign_smoothing_velocity))
            if sign_smoothing_velocity > 0.0 else float(np.sign(qd[joint]))
        )
        regressor[joint, 3 * joint] = sign
        regressor[joint, 3 * joint + 1] = (
            (abs(float(qd[joint])) + 1e-12) ** max(float(alpha[joint]), 1e-6)
            * sign
        )
        regressor[joint, 3 * joint + 2] = 1.0
    return regressor


class FirstOrderIIR:
    def __init__(self, gain: float | list[float] | np.ndarray, size: int):
        self.gain = np.asarray(
            gain if np.ndim(gain) else [gain] * size, dtype=float
        ).reshape(size)
        self.value: Optional[np.ndarray] = None

    def update(self, sample: np.ndarray, dt: float) -> np.ndarray:
        sample = asvec(sample, self.gain.size, "filter sample")
        if self.value is None:
            self.value = sample.copy()
        else:
            factor = 1.0 - np.exp(-np.maximum(self.gain, 0.0) * max(float(dt), 1e-6))
            factor = np.where(self.gain <= 0.0, 1.0, factor)
            self.value += factor * (sample - self.value)
        return self.value.copy()


class BlockRLS:
    def __init__(
        self,
        x0: np.ndarray,
        p0: float,
        forgetting: float,
        process_noise: float,
    ):
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = float(p0) * np.eye(len(self.x))
        self.forgetting = float(forgetting)
        self.process_noise = float(process_noise)
        self.k = 0
        self.last_step_norm = 0.0
        self.last_residual_norm = 0.0

    def update(self, phi: np.ndarray, observation: np.ndarray, noise: np.ndarray) -> None:
        phi = np.asarray(phi, dtype=float)
        observation = asvec(observation, phi.shape[0], "RLS observation")
        if not all(np.all(np.isfinite(x)) for x in (phi, observation, self.x, self.P)):
            raise FloatingPointError("RLS received a non-finite input or state")
        predicted_covariance = (
            self.P / max(self.forgetting, 1e-9)
            + self.process_noise * np.eye(len(self.x))
        )
        innovation_covariance = phi @ predicted_covariance @ phi.T + noise
        gain = predicted_covariance @ phi.T @ np.linalg.pinv(innovation_covariance)
        residual = observation - phi @ self.x
        step = gain @ residual
        identity = np.eye(len(self.x))
        covariance_factor = identity - gain @ phi
        new_covariance = (
            covariance_factor @ predicted_covariance @ covariance_factor.T
            + gain @ noise @ gain.T
        )
        new_state = self.x + step
        if not (np.all(np.isfinite(new_state)) and np.all(np.isfinite(new_covariance))):
            raise FloatingPointError("RLS update produced a non-finite state")
        self.x = new_state
        self.P = 0.5 * (new_covariance + new_covariance.T)
        self.k += 1
        self.last_step_norm = float(np.linalg.norm(step))
        self.last_residual_norm = float(np.linalg.norm(residual))


class PayloadStateTracker:
    def __init__(self, config: dict[str, Any]):
        self.pickup_mass = float(config.get("pickup_mass_kg", 0.08))
        self.drop_mass = float(config.get("drop_mass_kg", 0.05))
        self.placed_fraction = float(config.get("placed_mass_fraction", 0.35))
        self.dwell = max(1, int(config.get("dwell_samples", 25)))
        self.slow_velocity = float(config.get("slow_velocity_rad_s", 0.08))
        self.slow_acceleration = float(config.get("slow_acceleration_rad_s2", 0.6))
        self.state = "empty"
        self.carried_mass = 0.0
        self.pickup_count = 0
        self.drop_count = 0

    def update(self, mass: float, qd_norm: float, qdd_norm: float) -> dict[str, Any]:
        mass = max(0.0, float(mass))
        slow = qd_norm <= self.slow_velocity and qdd_norm <= self.slow_acceleration
        if mass >= self.pickup_mass:
            self.pickup_count += 1
            self.drop_count = 0
            if self.pickup_count >= self.dwell:
                self.state = "carried"
                self.carried_mass = max(self.carried_mass, mass)
        else:
            self.pickup_count = 0
        drop_level = max(self.drop_mass, self.placed_fraction * self.carried_mass)
        if self.carried_mass >= self.pickup_mass and mass <= drop_level:
            self.drop_count += 1
            if self.drop_count >= self.dwell:
                self.state = "placed_on_table" if slow else "released_or_supported"
        elif self.carried_mass < self.pickup_mass and mass <= self.drop_mass:
            self.state = "empty"
            self.drop_count = 0
        support_ratio = (
            float(np.clip(1.0 - mass / self.carried_mass, 0.0, 1.0))
            if self.carried_mass > 1e-6 else 0.0
        )
        return {
            "payload_state": self.state,
            "placed_on_table": self.state == "placed_on_table",
            "carried_mass_ref": float(self.carried_mass),
            "support_ratio": support_ratio,
            "drop_threshold_kg": float(drop_level),
            "arm_slow": bool(slow),
        }


class T1RegressorBuilder:
    """Minimal robot-side Pinocchio regressor; no local project import."""

    def __init__(self, config: dict[str, Any], side: str):
        try:
            import pinocchio as pin
        except Exception as exc:
            raise ImportError(
                "Pinocchio is required on the robot PC. Install the official "
                "Python package before running online identification."
            ) from exc
        self.pin = pin
        robot_config = config["robot"]
        identification_config = config.get("identification", {})
        urdf_path = resolve_path(robot_config["urdf_path"])
        mesh_dir = resolve_path(robot_config.get("mesh_dir", urdf_path.parent))
        try:
            robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), str(mesh_dir))
            self.model = robot.model
        except Exception:
            self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.include_motor = bool(identification_config.get("include_motor_inertia", True))
        self.sign_epsilon = float(
            identification_config.get("friction_sign_smoothing_vel", 0.0)
        )
        joint_names = list(robot_config["urdf_joint_names"][side])
        self.q_indices = []
        self.v_indices = []
        for name in joint_names:
            joint_id = int(self.model.getJointId(name))
            if joint_id == 0:
                raise ValueError(f"joint not found in URDF: {name}")
            self.q_indices.append(int(self.model.joints[joint_id].idx_q))
            self.v_indices.append(int(self.model.joints[joint_id].idx_v))
        self.q_indices = np.asarray(self.q_indices, dtype=int)
        self.v_indices = np.asarray(self.v_indices, dtype=int)
        payload_joint_name = robot_config.get("payload_target_joint", {}).get(
            side, joint_names[-1]
        )
        self.payload_joint_id = int(self.model.getJointId(payload_joint_name))
        if self.payload_joint_id == 0:
            raise ValueError(f"payload target joint not found: {payload_joint_name}")

    def rigid_regressor(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        q_full = self.pin.neutral(self.model)
        qd_full = np.zeros(self.model.nv)
        qdd_full = np.zeros(self.model.nv)
        q_full[self.q_indices] = asvec(q, 7, "q")
        qd_full[self.v_indices] = asvec(qd, 7, "qd")
        qdd_full[self.v_indices] = asvec(qdd, 7, "qdd")
        regressor = self.pin.computeJointTorqueRegressor(
            self.model, self.data, q_full, qd_full, qdd_full
        )
        if regressor is None:
            regressor = self.data.jointTorqueRegressor
        return np.asarray(regressor)[self.v_indices, :]

    def full_regressor(
        self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        blocks = [self.rigid_regressor(q, qd, qdd)]
        if self.include_motor:
            motor = np.zeros((7, 7))
            np.fill_diagonal(motor, asvec(qdd, 7))
            blocks.append(motor)
        blocks.append(nonlinear_friction_regressor(qd, alpha, self.sign_epsilon))
        return np.hstack(blocks)

    def payload_regressor(
        self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray
    ) -> np.ndarray:
        rigid = self.rigid_regressor(q, qd, qdd)
        injection = np.zeros((10 * (self.model.njoints - 1), 10))
        start = 10 * (self.payload_joint_id - 1)
        injection[start:start + 10, :] = np.eye(10)
        return rigid @ injection @ payload_parameter_map("xu10")


class T1OnlinePayloadEstimator:

    def __init__(
        self,
        model_npz: str | Path,
        method: str = "momentum_observer",
        forgetting: float = 0.99,
        p0: float = 1.0,
        process_noise: float = 0.0,
        observation_var: float = 1.0,
        mass_min: float = 0.0,
        mass_max: Optional[float] = 8.0,
        mass_ema_alpha: float = 0.1,
        exact_project_period: int = 100,
        convergence_window: int = 40,
        convergence_mass_std_kg: float = 0.03,
    ):
        if method not in MOMENTUM_METHODS:
            raise ValueError("standalone robot entry supports only momentum methods")
        model = np.load(resolve_path(model_npz), allow_pickle=True)
        try:
            self.config = json.loads(str(model["config_json"]))
            self.side = str(np.asarray(model["side"]).item())
            self.alpha = np.asarray(model["alpha"], dtype=float).reshape(7)
            self.beta0 = np.asarray(model["beta0"], dtype=float).reshape(-1)
            self.keep_mask = np.asarray(model["keep_mask"], dtype=bool)
            self.base_indices = np.asarray(model["base_idx_in_keep"], dtype=int)
            theta_virtual = (
                np.asarray(model["theta_virtual0"], dtype=float).reshape(16)
                if "theta_virtual0" in model.files else np.zeros(16)
            )
            virtual_bias = (
                np.asarray(model["virtual_bias0"], dtype=float).reshape(7)
                if "virtual_bias0" in model.files else np.zeros(7)
            )
            self.virtual_xu10 = (
                np.asarray(model["online_virtual_xu10"], dtype=float).reshape(10)
                if "online_virtual_xu10" in model.files
                else theta16_to_xu10(theta_virtual)
            )
            self.virtual_bias = (
                np.asarray(model["online_virtual_bias0"], dtype=float).reshape(7)
                if "online_virtual_bias0" in model.files else virtual_bias
            )
            self.has_causal_calibration = "online_virtual_xu10" in model.files
        finally:
            model.close()

        self.method = "momentum_observer" if method == "momentum" else method
        self.builder = T1RegressorBuilder(self.config, self.side)
        online = self.config.get("online", {})
        observer_gain = online.get(
            "observer_gain", online.get("momentum_observer_gain", 20.0)
        )
        self.default_dt = float(self.config.get("robot", {}).get("control_dt", CONTROL_DT))
        self.q_filter = FirstOrderIIR(observer_gain, 7)
        self.qd_filter = FirstOrderIIR(observer_gain, 7)
        self.qdd_filter = FirstOrderIIR(observer_gain, 7)
        self.tau_filter = FirstOrderIIR(observer_gain, 7)
        self.qdd_limit = float(online.get("online_qdd_abs_limit_rad_s2", 15.0))
        self.use_commanded_acceleration = bool(
            online.get("use_commanded_acceleration_in_momentum_mode", False)
        )
        self.use_virtual_calibration = bool(online.get("use_virtual_calibration", True))
        self.estimate_bias = bool(online.get("estimate_external_bias", True))
        initial = self.virtual_xu10.copy()
        if self.estimate_bias:
            initial = np.r_[initial, self.virtual_bias]
        self.rls = BlockRLS(initial, p0, forgetting, process_noise)
        self.observation_noise = float(observation_var) * np.eye(7)
        self.mass_min = float(mass_min)
        self.mass_max = None if mass_max is None else float(mass_max)
        self.mass_ema_alpha = float(mass_ema_alpha)
        self.exact_project_period = max(1, int(exact_project_period))
        self.mass_ema: Optional[float] = None
        self.mass_history = deque(maxlen=max(2, int(convergence_window)))
        self.convergence_std = float(convergence_mass_std_kg)
        self.state_tracker = PayloadStateTracker(online.get("placed_detection", {}))
        self.last_timestamp: Optional[float] = None
        self.reference_table: Optional[dict[str, np.ndarray]] = None
        self.last_report: dict[str, Any] = {}

    def _selected_regressor(
        self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray
    ) -> np.ndarray:
        full = self.builder.full_regressor(q, qd, qdd, self.alpha)
        return full[:, self.keep_mask][:, self.base_indices]

    def prepare_reference_table(
        self,
        q_ref: np.ndarray,
        qd_ref: np.ndarray,
        qdd_ref: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, np.ndarray]:
        q_ref = np.asarray(q_ref, dtype=float)
        qd_ref = np.asarray(qd_ref, dtype=float)
        qdd_ref = np.asarray(qdd_ref, dtype=float)
        timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
        if q_ref.shape != (len(timestamps), 7) or qd_ref.shape != q_ref.shape or qdd_ref.shape != q_ref.shape:
            raise ValueError("reference table requires t=N and q/qd/qdd=Nx7")
        if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("reference table timestamps must strictly increase")

        q_filter = FirstOrderIIR(self.q_filter.gain, 7)
        qd_filter = FirstOrderIIR(self.qd_filter.gain, 7)
        qdd_filter = FirstOrderIIR(self.qdd_filter.gain, 7)
        filtered_q = np.empty_like(q_ref)
        filtered_qd = np.empty_like(qd_ref)
        filtered_qdd = np.empty_like(qdd_ref)
        empty_torque = np.empty_like(q_ref)
        payload_regressor = np.empty((len(timestamps), 7, 10))
        last_qd: Optional[np.ndarray] = None
        for index in range(len(timestamps)):
            dt = self.default_dt if index == 0 else max(
                float(timestamps[index] - timestamps[index - 1]), 1e-5
            )
            filtered_q[index] = q_filter.update(q_ref[index], dt)
            filtered_qd[index] = qd_filter.update(qd_ref[index], dt)
            if self.use_commanded_acceleration:
                acceleration_source = qdd_ref[index]
            elif last_qd is None:
                acceleration_source = np.zeros(7)
            else:
                acceleration_source = (filtered_qd[index] - last_qd) / dt
            last_qd = filtered_qd[index].copy()
            acceleration_source = np.clip(
                acceleration_source, -self.qdd_limit, self.qdd_limit
            )
            filtered_qdd[index] = qdd_filter.update(acceleration_source, dt)
            selected = self._selected_regressor(
                filtered_q[index], filtered_qd[index], filtered_qdd[index]
            )
            empty_torque[index] = selected @ self.beta0
            payload_regressor[index] = self.builder.payload_regressor(
                filtered_q[index], filtered_qd[index], filtered_qdd[index]
            )
        self.reference_table = {
            "t": timestamps.copy(), "q_ref": q_ref.copy(),
            "qd_ref": qd_ref.copy(), "qdd_ref": qdd_ref.copy(),
            "qd_f": filtered_qd, "qdd_f": filtered_qdd,
            "tau0": empty_torque, "W10": payload_regressor,
        }
        return self.reference_table

    def _dt(self, timestamp: Optional[float]) -> float:
        if timestamp is None:
            return self.default_dt
        timestamp = float(timestamp)
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return self.default_dt
        dt = max(timestamp - self.last_timestamp, 1e-5)
        self.last_timestamp = timestamp
        return dt

    def update_precomputed(
        self,
        table_index: int,
        tau_loaded: np.ndarray,
        q_meas: Optional[np.ndarray] = None,
        measured_qd: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> dict[str, Any]:
        if self.reference_table is None:
            raise RuntimeError("prepare_reference_table must be called first")
        index = int(table_index)
        if index < 0 or index >= len(self.reference_table["t"]):
            raise IndexError("reference table index out of range")
        dt = self._dt(timestamp)
        torque = asvec(tau_loaded, 7, "tau_loaded")
        filtered_torque = self.tau_filter.update(torque, dt)
        empty_torque = self.reference_table["tau0"][index]
        payload_regressor = self.reference_table["W10"][index]
        external_torque = filtered_torque - empty_torque
        phi = (
            np.hstack([payload_regressor, np.eye(7)])
            if self.estimate_bias else payload_regressor
        )
        self.rls.update(phi, external_torque, self.observation_noise)

        raw_xu10 = self.rls.x[:10]
        payload_xu10 = raw_xu10 - (
            self.virtual_xu10 if self.use_virtual_calibration else 0.0
        )
        theta = project_theta16(
            xu10_to_theta16(payload_xu10), self.mass_min, self.mass_max
        )
        report = payload_report(theta)
        mass = float(report["mass"])
        self.mass_ema = (
            mass if self.mass_ema is None
            else (1.0 - self.mass_ema_alpha) * self.mass_ema
            + self.mass_ema_alpha * mass
        )
        self.mass_history.append(mass)
        qd = self.reference_table["qd_f"][index]
        qdd = self.reference_table["qdd_f"][index]
        state = self.state_tracker.update(
            self.mass_ema, float(np.linalg.norm(qd)), float(np.linalg.norm(qdd))
        )
        bias_raw = self.rls.x[10:17] if self.estimate_bias else np.zeros(7)
        bias_hat = bias_raw - (
            self.virtual_bias
            if self.use_virtual_calibration and self.estimate_bias else 0.0
        )
        standard_deviation = (
            float(np.std(self.mass_history)) if len(self.mass_history) > 2 else None
        )
        report.update({
            "sample": int(self.rls.k),
            "side": self.side,
            "method": self.method,
            "mass_ema": float(self.mass_ema),
            "mass_window_std": standard_deviation,
            "converged": (
                len(self.mass_history) == self.mass_history.maxlen
                and standard_deviation is not None
                and standard_deviation < self.convergence_std
            ),
            "rls_step_norm": self.rls.last_step_norm,
            "rls_residual_norm": self.rls.last_residual_norm,
            "qd_norm": float(np.linalg.norm(qd)),
            "qdd_norm": float(np.linalg.norm(qdd)),
            "observer_dt": float(dt),
            "reference_table_index": index,
            "reference_table_time": float(self.reference_table["t"][index]),
            "tau_loaded": torque.tolist(),
            "tau_loaded_filtered": filtered_torque.tolist(),
            "tau_empty_hat": empty_torque.tolist(),
            "tau_external_filtered": external_torque.tolist(),
            "bias_hat": bias_hat.tolist(),
            "virtual_calibration_enabled": self.use_virtual_calibration,
            "causal_online_calibration_available": self.has_causal_calibration,
        })
        if q_meas is not None:
            report["q_tracking_error"] = (
                asvec(q_meas, 7) - self.reference_table["q_ref"][index]
            ).tolist()
        report.update(state)
        self.last_report = report
        return report


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

    def set(self, state: RobotState) -> None:
        with self._lock:
            self._latest = state

    def get(self) -> Optional[RobotState]:
        with self._lock:
            return self._latest


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


class T1OnlineController:
    """Known-good walking-mode upper-body controller used by online ID."""

    def __init__(self, network_interface: str = "", simulation: bool = False):
        self.network_interface = network_interface
        self.simulation = bool(simulation)
        self.state_buffer = StateBuffer()
        self.current_jpos_des = FIXED_HOME_POSITION.copy()
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

        if self.simulation:
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

        # Recreate LowCmd every cycle, matching the fixed collector/teleop path.
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
        wait_until(cycle_start + CONTROL_DT)

    def start_control(self, handoff_sec: float = DEFAULT_HANDOFF_SEC) -> None:
        if not self._call_upper_body_custom_control(True):
            self.weight = 0.0
            self._publish_enabled = False
            raise RuntimeError("Failed to enable UpperBodyCustomControl")
        try:
            handoff_q = self.get_state().q.copy()
            self.handoff_q = handoff_q.copy()
            self.current_jpos_des = handoff_q.copy()
            self.q_target = handoff_q.copy()
            self.weight = 0.0
            self._publish_enabled = True

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

    def make_target(
        self,
        side: str,
        active_arm_q: np.ndarray,
        other_arm_q: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        target = self.get_state().q.copy()
        target[0] = FIXED_HOME_POSITION[0]
        target[1] = FIXED_HOME_POSITION[1]
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
        print("[T1] the recorded/identified excitation itself is not rate-limited.")
        initial_state = self.get_state().q.copy()
        initial_target = self.make_target(side, start_q, other_arm_q)
        check = np.asarray(ARM_INDICES[side])
        required_motion = float(np.max(np.abs(
            initial_target[check] - initial_state[check]
        )))
        effective_timeout = max(
            timeout_sec, required_motion / MAX_JOINT_VELOCITY + 3.0
        )
        deadline = time.perf_counter() + effective_timeout
        watchdog_deadline = time.perf_counter() + 1.0
        stable_cycles = 0
        while True:
            cycle_start = time.perf_counter()
            target = self.make_target(side, start_q, other_arm_q)
            self._publish(self._clip_upper_target(target))
            state = self.get_state()
            command_remaining = float(np.max(np.abs(
                target[check] - self.current_jpos_des[check]
            )))
            measured_error = float(np.max(np.abs(state.q[check] - target[check])))
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
            if time.perf_counter() >= watchdog_deadline and required_motion > 0.10:
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

    def _return_to_handoff_pose(self, timeout_sec: float = 10.0) -> None:
        if self.handoff_q is None:
            print("[WARN] no handoff_q recorded; skip return-to-handoff motion.")
            return
        target = self.handoff_q.copy()
        initial_state = self.get_state().q.copy()
        upper = np.asarray(UPPER_BODY_INDICES)
        required_motion = float(np.max(np.abs(target[upper] - initial_state[upper])))
        deadline = time.perf_counter() + max(
            timeout_sec, required_motion / MAX_JOINT_VELOCITY + 3.0
        )
        stable_cycles = 0
        print("[T1] returning upper body to handoff pose before release ...")
        while True:
            cycle_start = time.perf_counter()
            target[LOWER_BODY_INDICES] = self.get_state().q[LOWER_BODY_INDICES]
            self._publish(self._clip_upper_target(target))
            state = self.get_state()
            command_remaining = float(np.max(np.abs(
                target[upper] - self.current_jpos_des[upper]
            )))
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
                    f"max_velocity={measured_velocity:.4f} rad/s"
                )
                return
            self._sleep_cycle(cycle_start)

    def stop_control(self) -> None:
        if not self.custom_control_enabled:
            self.weight = 0.0
            self._publish_enabled = False
            return
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


def control_grid(duration: float) -> np.ndarray:
    return time_grid(duration, CONTROL_DT)


def time_grid(duration: float, dt: float) -> np.ndarray:
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("time-grid dt must be positive")
    count = int(np.floor(duration / dt + 1e-12))
    grid = np.arange(count + 1, dtype=float) * dt
    if duration - grid[-1] > 1e-9:
        grid = np.append(grid, duration)
    else:
        grid[-1] = duration
    return grid


def wait_until(deadline: float) -> None:
    """Wait for an absolute monotonic deadline.

    Python 3.10 on Windows has a coarse sleep timer that cannot represent the
    2 ms validation grid.  The robot runs Linux and keeps the collector's sleep
    path; Windows simulation uses a short spin wait solely for timing fidelity.
    """
    remaining = float(deadline) - time.perf_counter()
    if remaining <= 0.0:
        return
    if os.name == "nt":
        while time.perf_counter() < deadline:
            pass
    else:
        time.sleep(remaining)


def csv_header() -> list[str]:
    header = [
        "wall_time", "traj_time", "command_traj_time", "publish_wall_time",
        "publish_lateness_s", "low_state_age_s", "command_index",
        "mass", "mass_ema", "known_mass_kg", "mass_error_kg",
        "com_x", "com_y", "com_z", "residual_norm", "step_norm",
        "mass_window_std", "converged", "payload_state", "placed_on_table",
        "support_ratio", "carried_mass_ref",
    ]
    for prefix in [
        "q_ref", "qd_ref", "qdd_ref", "q_meas", "qd_meas", "tau_est",
    ]:
        header += [f"{prefix}_{i}" for i in range(7)]
    return header


class OnlineEstimatorTableWorker:
    """Lightweight runtime update using dynamics precomputed before SDK start."""

    def __init__(
        self,
        estimator: T1OnlinePayloadEstimator,
        arm: np.ndarray,
        known_mass: Optional[float],
    ):
        self.estimator = estimator
        self.arm = np.asarray(arm, dtype=int)
        self.known_mass = known_mass
        self.rows: list[list[Any]] = []
        self.last_report: dict[str, Any] = {}
        self.dropped_updates = 0
        self._closed = False

    def submit(self, item: dict[str, Any]) -> None:
        state: RobotState = item["state"]
        report = self.estimator.update_precomputed(
            item["table_index"], state.tau_est[self.arm],
            q_meas=state.q[self.arm], measured_qd=state.dq[self.arm],
            timestamp=state.monotonic_time,
        )
        self.last_report = report
        mass_error = (
            "" if self.known_mass is None
            else float(report["mass_ema"]) - float(self.known_mass)
        )
        row = [
            state.wall_time, item["state_time"], item["command_time"],
            item["publish_wall"], item["lateness"], item["state_age"],
            item["command_index"], report["mass"], report["mass_ema"],
            "" if self.known_mass is None else self.known_mass, mass_error,
            report["com"][0], report["com"][1], report["com"][2],
            report["rls_residual_norm"], report["rls_step_norm"],
            "" if report["mass_window_std"] is None else report["mass_window_std"],
            report["converged"], report["payload_state"],
            report["placed_on_table"], report["support_ratio"],
            "" if report["carried_mass_ref"] is None
            else report["carried_mass_ref"],
        ]
        row += item["q_ref"].tolist() + item["qd_ref"].tolist()
        row += item["qdd_ref"].tolist() + state.q[self.arm].tolist()
        row += state.dq[self.arm].tolist() + state.tau_est[self.arm].tolist()
        self.rows.append(row)

    def is_idle(self) -> bool:
        return True

    def close(self, timeout: float = 0.0) -> None:
        self._closed = True


def execute_online(
    robot: T1OnlineController,
    worker: OnlineEstimatorTableWorker,
    side: str,
    trajectory: QuinticTrajectory,
    output_path: Path,
    print_period: float,
    known_mass: Optional[float],
    estimator_update_period: float,
    estimator_table_times: np.ndarray,
) -> dict[str, Any]:
    other = "right" if side == "left" else "left"
    other_arm_q = OTHER_ARM_HOLD[other]
    start_q, _, _ = trajectory.evaluate(0.0)
    robot.move_to_trajectory_start(side, start_q, other_arm_q)
    times = control_grid(trajectory.duration)
    missed_deadlines = 0
    max_lateness = 0.0
    max_state_age = 0.0
    last_state_monotonic = float("-inf")
    next_estimator_index = 0
    next_print = time.perf_counter()
    try:
        start_monotonic = time.perf_counter()
        k = 0
        while k < len(times):
            target_time = start_monotonic + float(times[k])
            remaining = target_time - time.perf_counter()
            if remaining > 0.0:
                wait_until(target_time)

            published_monotonic = time.perf_counter()
            if k < len(times) - 1:
                due_index = int(np.searchsorted(
                    times,
                    min(published_monotonic - start_monotonic, trajectory.duration),
                    side="right",
                ) - 1)
                if due_index > k:
                    missed_deadlines += due_index - k
                    k = due_index
                    target_time = start_monotonic + float(times[k])

            q_cmd, _, _ = trajectory.evaluate(float(times[k]))
            robot._publish(robot.make_target(side, q_cmd, other_arm_q))
            publish_wall = time.time()
            lateness = max(0.0, published_monotonic - target_time)
            max_lateness = max(max_lateness, lateness)

            state = robot.get_state()
            if state.monotonic_time > last_state_monotonic:
                last_state_monotonic = state.monotonic_time
                state_time = float(np.clip(
                    state.monotonic_time - start_monotonic,
                    0.0,
                    trajectory.duration,
                ))
                q_ref, qd_ref, qdd_ref = trajectory.evaluate(state_time)
                state_age = max(0.0, published_monotonic - state.monotonic_time)
                max_state_age = max(max_state_age, state_age)
                due_estimator_index = int(np.searchsorted(
                    estimator_table_times,
                    state_time + 0.5 * CONTROL_DT,
                    side="right",
                ) - 1)
                if due_estimator_index >= next_estimator_index:
                    if due_estimator_index > next_estimator_index:
                        worker.dropped_updates += (
                            due_estimator_index - next_estimator_index
                        )
                    table_index = due_estimator_index
                    table_time = float(estimator_table_times[table_index])
                    q_ref, qd_ref, qdd_ref = trajectory.evaluate(table_time)
                    worker.submit({
                        "state": state,
                        "table_index": table_index,
                        "state_time": state_time,
                        "command_time": float(times[k]),
                        "publish_wall": publish_wall,
                        "lateness": lateness,
                        "state_age": state_age,
                        "command_index": k,
                        "q_ref": q_ref,
                        "qd_ref": qd_ref,
                        "qdd_ref": qdd_ref,
                    })
                    next_estimator_index = table_index + 1

                report = worker.last_report
                if report and print_period > 0.0 and time.perf_counter() >= next_print:
                    mass_error = (
                        "" if known_mass is None
                        else float(report["mass_ema"]) - float(known_mass)
                    )
                    known_text = (
                        "" if known_mass is None
                        else f"  err={mass_error:+.3f} kg"
                    )
                    print(
                        f"t={state_time:6.2f}s  mass={report['mass']:6.3f} kg  "
                        f"ema={report['mass_ema']:6.3f} kg{known_text}  "
                        f"conv={report['converged']}"
                    )
                    next_print = time.perf_counter() + print_period
            k += 1

        final_target = robot.make_target(side, trajectory.evaluate(trajectory.duration)[0], other_arm_q)
        for _ in range(2):
            cycle_start = time.perf_counter()
            robot._publish(final_target)
            robot._sleep_cycle(cycle_start)
        estimator_deadline = time.perf_counter() + 2.0
        while not worker.is_idle() and time.perf_counter() < estimator_deadline:
            cycle_start = time.perf_counter()
            robot._publish(final_target)
            robot._sleep_cycle(cycle_start)
        worker.close(timeout=5.0)
    finally:
        if not worker._closed:
            worker.close(timeout=5.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(csv_header())
        writer.writerows(worker.rows)

    estimator_updates = len(worker.rows)
    summary = {
        "side": side,
        "duration_s": trajectory.duration,
        "control_grid_points": int(len(times)),
        "published_points": int(len(times) - missed_deadlines),
        "missed_deadlines": int(missed_deadlines),
        "max_publish_lateness_s": float(max_lateness),
        "estimator_updates": int(estimator_updates),
        "estimator_update_period_s": float(estimator_update_period),
        "estimator_dropped_updates": int(worker.dropped_updates),
        "max_low_state_age_s": float(max_state_age),
        "known_mass_kg": known_mass,
        "final_report": worker.last_report,
    }
    print(
        f"[T1] published {summary['published_points']}/{len(times)} points; "
        f"estimator_updates={estimator_updates}; "
        f"max_lateness={max_lateness * 1e3:.3f} ms"
    )
    if missed_deadlines:
        print(
            f"[WARN] missed {missed_deadlines} control deadlines; "
            "do not use this run for identification"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run online payload identification with the same walking-mode "
            "control path as robot_collect_excitation.py."
        )
    )
    parser.add_argument(
        "network_interface", nargs="?", default=DEFAULT_NETWORK,
        help=f"SDK network address/interface (default: {DEFAULT_NETWORK}).",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="empty prior NPZ")
    parser.add_argument("--traj", default=DEFAULT_TRAJECTORY, help="generated excitation NPZ")
    parser.add_argument("--side", choices=["left", "right"], default=DEFAULT_SIDE)
    parser.add_argument("--method", choices=sorted(MOMENTUM_METHODS), default=None)
    parser.add_argument(
        "--out", default=None,
        help="output CSV; default is a timestamped file under results/",
    )
    parser.add_argument("--known-mass", type=float, default=None)
    parser.add_argument("--simulation", action="store_true")
    args = parser.parse_args()

    if not args.simulation and not args.network_interface:
        parser.error(
            "network_interface is required, e.g. "
            f"python3 online_payload_estimator.py {DEFAULT_NETWORK}"
        )
    if args.known_mass is not None and args.known_mass < 0.0:
        parser.error("--known-mass must be non-negative")

    trajectory_path = resolve_path(args.traj)
    raw = np.load(trajectory_path, allow_pickle=True)
    trajectory_side = (
        str(np.asarray(raw["side"]).item()) if "side" in raw.files else None
    )
    side = args.side or trajectory_side or "left"
    if trajectory_side is not None and trajectory_side != side:
        parser.error(
            f"trajectory side is {trajectory_side!r}, but --side is {side!r}"
        )
    trajectory = QuinticTrajectory(raw["t"], raw["q"], raw["qd"], raw["qdd"])

    cfg = load_json(args.config)
    online_cfg = cfg.get("online", {})
    model_path = resolve_path(args.model)
    model_data = np.load(model_path, allow_pickle=True)
    model_side = str(np.asarray(model_data["side"]).item())
    method = args.method or online_cfg.get("method", "momentum_observer")
    if method not in {"momentum_observer", "momentum_direct", "momentum"}:
        parser.error(
            "online_payload_estimator.py uses the precomputed momentum path; "
            "legacy methods are available only in offline replay"
        )
    estimator_kwargs = dict(
        method=method,
        forgetting=online_cfg.get("forgetting", 0.99),
        p0=online_cfg.get("p0", 1.0),
        process_noise=online_cfg.get("process_noise", 0.0),
        observation_var=online_cfg.get("observation_var", 1.0),
        mass_min=online_cfg.get("mass_min", 0.0),
        mass_max=online_cfg.get("mass_max", 8.0),
        mass_ema_alpha=online_cfg.get("mass_ema_alpha", 0.1),
        exact_project_period=online_cfg.get("exact_project_period", 100),
        convergence_window=online_cfg.get("convergence_mass_std_window", 40),
        convergence_mass_std_kg=online_cfg.get("convergence_mass_std_kg", 0.03),
    )
    model_data.close()
    if model_side != side:
        parser.error(
            f"model side is {model_side!r}, but trajectory/--side is {side!r}"
        )
    output_path = resolve_path(
        args.out
        or f"results/robot_online_payload_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    update_period = float(online_cfg.get("estimator_update_period_s", 0.05))
    estimator_table_times = time_grid(trajectory.duration, update_period)
    table_q = np.empty((len(estimator_table_times), 7))
    table_qd = np.empty_like(table_q)
    table_qdd = np.empty_like(table_q)
    for k, query in enumerate(estimator_table_times):
        table_q[k], table_qd[k], table_qdd[k] = trajectory.evaluate(float(query))
    print(
        f"[INFO] precomputing {len(estimator_table_times)} dynamics rows "
        "before SDK initialization ..."
    )
    estimator = T1OnlinePayloadEstimator(model_path, **estimator_kwargs)
    estimator.prepare_reference_table(
        table_q, table_qd, table_qdd, estimator_table_times
    )
    worker = OnlineEstimatorTableWorker(
        estimator, np.asarray(ARM_INDICES[side]), args.known_mass
    )
    robot = None
    started = False
    try:
        # All Pinocchio work is complete before creating DDS/SDK resources.
        robot = T1OnlineController(
            network_interface=args.network_interface or "",
            simulation=args.simulation,
        )
        print(
            f"[INFO] online {side} payload ID: model={model_path}, "
            f"trajectory={trajectory_path}, output={output_path}"
        )
        print(
            "[INFO] assuming robot is already stable in mw walking mode; "
            "UpperBodyCustomControl will be enabled now."
        )
        robot.start_control(handoff_sec=DEFAULT_HANDOFF_SEC)
        started = True
        summary = execute_online(
            robot,
            worker,
            side,
            trajectory,
            output_path,
            print_period=float(online_cfg.get("print_period", 0.0)),
            known_mass=args.known_mass,
            estimator_update_period=update_period,
            estimator_table_times=estimator_table_times,
        )
        summary.update({
            "model": str(resolve_path(args.model)),
            "trajectory": str(trajectory_path),
            "output_csv": str(output_path),
        })
        report_path = output_path.with_suffix(".json")
        report_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("[FINAL]", json.dumps(worker.last_report, indent=2, ensure_ascii=False)[:2500])
        print(f"[OK] saved {output_path} and {report_path}")
    finally:
        if started and robot is not None:
            robot.stop_control()
        if not worker._closed:
            worker.close(timeout=10.0)


if __name__ == "__main__":
    main()
