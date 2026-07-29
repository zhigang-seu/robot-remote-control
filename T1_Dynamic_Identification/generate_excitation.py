#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Generate a safe Fourier excitation trajectory for one T1 arm.

目标：
使用 Pinocchio computeJointTorqueRegressor 优化末端 payload 相关回归器。
为后续动量观测器负载质量辨识生成动力学激励轨迹。

输出：
    trajectories/left_arm_payload_excitation.npz
    trajectories/left_arm_payload_excitation.csv
    trajectories/left_arm_payload_excitation.mat
    trajectories/left_arm_payload_excitation_report.json
"""

from __future__ import annotations

import math
import re
import sys
import traceback
import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import savemat


# =============================================================================
# =============================================================================
# =============================================================================

MANUAL_PROJECT_ROOT: str | None = None

CONFIG_REL_PATH = "configs/t1_payload_config.json"

SIDE = "left"  # "left" or "right"

OUT_PREFIX: str | None = None

MANUAL_URDF_PATH: str | None = None

# T1 URDF 真实 7 轴手臂关节名
MANUAL_ARM_JOINT_NAMES_BY_SIDE = {
    "left": [
        "Left_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "Left_Wrist_Pitch",
        "Left_Wrist_Yaw",
        "Left_Hand_Roll",
    ],
    "right": [
        "Right_Shoulder_Pitch",
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Roll" if False else "Right_Elbow_Yaw",
        "Right_Wrist_Pitch",
        "Right_Wrist_Yaw",
        "Right_Hand_Roll",
    ],
}

# 末端负载默认挂在最后一个手部 roll 关节对应刚体上
MANUAL_EE_JOINT_NAME_BY_SIDE = {
    "left": "Left_Hand_Roll",
    "right": "Right_Hand_Roll",
}

# flange/tool joint，可以在这里手动指定
MANUAL_EE_JOINT_NAME: str | None = None

# 最终做动量观测器负载辨识必须使用动力学目标
REQUIRE_DYNAMICS_OBJECTIVE = True

# 只在临时测试轨迹导出时改 True
ALLOW_KINEMATIC_FALLBACK = False

# 正常不要改
NO_DYNAMICS_OBJECTIVE = False

# None 表示自动按约 25 Hz 抽样计算动力学回归器
OBJECTIVE_STRIDE: int | None = None

# 想提高轨迹质量可以增大，例如 800 / 200
RANDOM_TRIALS_OVERRIDE = 1200
LOCAL_REFINE_STEPS_OVERRIDE = 300

# False 时只打印简洁错误，不重复刷屏
PRINT_TRACEBACK_ON_ERROR = False


# =============================================================================
# 1. 项目路径和 common 导入
# =============================================================================

def find_project_root() -> Path:
    if MANUAL_PROJECT_ROOT is not None:
        p = Path(MANUAL_PROJECT_ROOT).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"MANUAL_PROJECT_ROOT does not exist: {p}")
        return p

    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents]

    for p in candidates:
        if (p / "t1_payload_id").exists() and (p / "configs").exists():
            return p
        if (p / "generate_excitation.py").exists() and (p / "configs").exists():
            return p

    return here.parent


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from t1_payload_id.common import (
        load_json,
        save_json,
        resolve_project_path,
        trajectory_from_z,
        fourier_nullspace_basis,
        save_csv,
    )
except Exception as exc:
    raise ImportError(
        "Cannot import t1_payload_id.common. 请确认当前文件位于 "
        "T1_Dynamic_Identification 项目中，且项目根目录包含 t1_payload_id 文件夹。\n"
        f"Detected PROJECT_ROOT = {PROJECT_ROOT}"
    ) from exc


# =============================================================================
# 2. 数值工具函数
# =============================================================================

def _finite_or_large(x: float, large: float = 1.0e12) -> float:
    x = float(x)
    return x if np.isfinite(x) else large


def matrix_spectrum(
    Y: np.ndarray,
    col_tol: float = 1.0e-12,
    sv_tol: float = 1.0e-10,
) -> dict[str, Any]:
    """
    对可能秩亏的回归器计算 effective condition number。
    Pinocchio 的 joint torque regressor 是标准惯性参数回归器，
    不是最小基参数回归器，因此直接使用最小奇异值容易出现 0。
    """
    Y = np.asarray(Y, dtype=float)

    if Y.ndim != 2 or Y.size == 0:
        return {
            "shape": list(Y.shape) if hasattr(Y, "shape") else [0, 0],
            "rank": 0,
            "cond_eff": 1.0e12,
            "singular_values": [],
            "kept_columns": 0,
        }

    col_norm = np.linalg.norm(Y, axis=0)
    max_col = max(float(np.max(col_norm)) if col_norm.size else 0.0, 1.0)
    keep = col_norm > col_tol * max_col
    Yk = Y[:, keep]

    if Yk.shape[1] == 0:
        return {
            "shape": list(Y.shape),
            "rank": 0,
            "cond_eff": 1.0e12,
            "singular_values": [],
            "kept_columns": 0,
        }

    Yk = Yk / np.maximum(np.linalg.norm(Yk, axis=0, keepdims=True), 1.0e-12)

    try:
        s = np.linalg.svd(Yk, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "shape": list(Y.shape),
            "rank": 0,
            "cond_eff": 1.0e12,
            "singular_values": [],
            "kept_columns": int(Yk.shape[1]),
        }

    if s.size == 0 or not np.isfinite(s[0]) or s[0] <= 0.0:
        rank = 0
        cond_eff = 1.0e12
    else:
        tol = max(float(s[0]) * sv_tol, 1.0e-12)
        kept_s = s[s > tol]
        rank = int(kept_s.size)
        if kept_s.size == 0:
            cond_eff = 1.0e12
        else:
            cond_eff = float(kept_s[0] / max(kept_s[-1], 1.0e-12))

    return {
        "shape": list(Y.shape),
        "rank": int(rank),
        "cond_eff": _finite_or_large(cond_eff),
        "singular_values": [float(v) for v in s[: min(len(s), 30)]],
        "kept_columns": int(Yk.shape[1]),
    }


def cond_score(Y: np.ndarray) -> float:
    return float(matrix_spectrum(Y)["cond_eff"])


def log_condition_score(
    Y: np.ndarray,
    target_rank: int | None = None,
    rank_penalty: float = 2.0,
) -> float:
    spec = matrix_spectrum(Y)
    cond = max(float(spec["cond_eff"]), 1.0)
    score = math.log10(cond)

    if target_rank is not None:
        rank = int(spec["rank"])
        score += rank_penalty * max(0, int(target_rank) - rank)

    return float(score)


def _natural_key(name: str) -> list[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(name))]


def _side_short(side: str) -> str:
    s = str(side).lower()
    if s in ("l", "left"):
        return "l"
    if s in ("r", "right"):
        return "r"
    raise ValueError(f"Unknown side: {side}")


def _side_long(side: str) -> str:
    s = str(side).lower()
    if s in ("l", "left"):
        return "left"
    if s in ("r", "right"):
        return "right"
    raise ValueError(f"Unknown side: {side}")


def _project_path_or_absolute(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _nested_get(d: dict[str, Any], keys: Iterable[str]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _append_candidate_path(candidates: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (str, Path)) and str(value).strip():
        candidates.append(str(value))


# =============================================================================
# 3. URDF / joint name / alpha 解析
# =============================================================================

def resolve_urdf_path_from_config(cfg: dict[str, Any]) -> Path:
    candidates: list[str] = []

    _append_candidate_path(candidates, MANUAL_URDF_PATH)

    for keys in [
        ("urdf_path",),
        ("urdf",),
        ("robot_urdf",),
        ("model_urdf",),
        ("robot", "urdf_path"),
        ("robot", "urdf"),
        ("model", "urdf_path"),
        ("model", "urdf"),
        ("identification", "urdf_path"),
        ("identification", "urdf"),
        ("pinocchio", "urdf_path"),
        ("pinocchio", "urdf"),
    ]:
        _append_candidate_path(candidates, _nested_get(cfg, keys))

    candidates.extend(
        [
            "urdf/T1_7DofArm_Serial.urdf",
            "urdf/t1_7dof_arm_serial.urdf",
            "urdf/T1.urdf",
            "T1_7DofArm_Serial.urdf",
        ]
    )

    tried: list[str] = []
    for c in candidates:
        try:
            p = _project_path_or_absolute(c)
            tried.append(str(p))
            if p.exists():
                return p.resolve()
        except Exception:
            tried.append(str(c))

    raise FileNotFoundError(
        "Cannot resolve T1 URDF path. Tried:\n  - " + "\n  - ".join(tried)
    )


def available_1dof_joint_names(model: Any) -> list[str]:
    out: list[str] = []
    for jid in range(1, model.njoints):
        try:
            if int(model.nqs[jid]) == 1 and int(model.nvs[jid]) == 1:
                out.append(str(model.names[jid]))
        except Exception:
            pass
    return out


def _candidate_joint_names_from_config(cfg: dict[str, Any], side: str) -> list[str] | None:
    side_l = _side_long(side)
    side_s = _side_short(side)

    manual = MANUAL_ARM_JOINT_NAMES_BY_SIDE.get(side_l)
    if manual is None:
        manual = MANUAL_ARM_JOINT_NAMES_BY_SIDE.get(side_s)
    if isinstance(manual, list) and len(manual) == 7:
        return [str(x) for x in manual]

    containers = [
        cfg,
        cfg.get("robot", {}),
        cfg.get("model", {}),
        cfg.get("identification", {}),
        cfg.get("pinocchio", {}),
        cfg.get("excitation", {}),
    ]

    direct_keys = [
        f"{side_l}_arm_joint_names",
        f"{side_s}_arm_joint_names",
        f"arm_{side_s}_joint_names",
        f"{side_l}_joint_names",
        f"{side_s}_joint_names",
        "arm_joint_names",
        "joint_names",
    ]

    by_side_keys = [
        "arm_joint_names_by_side",
        "joint_names_by_side",
        "selected_joint_names_by_side",
        "controlled_joint_names_by_side",
    ]

    for c in containers:
        if not isinstance(c, dict):
            continue

        for key in direct_keys:
            val = c.get(key)
            if isinstance(val, list) and len(val) == 7:
                return [str(x) for x in val]

        for key in by_side_keys:
            val = c.get(key)
            if isinstance(val, dict):
                for sk in (side_l, side_s):
                    vv = val.get(sk)
                    if isinstance(vv, list) and len(vv) == 7:
                        return [str(x) for x in vv]

    return None


def _fallback_joint_name_sets(side: str) -> list[list[str]]:
    side_l = _side_long(side)
    side_s = _side_short(side)

    if side_l == "left":
        t1_real_names = [
            "Left_Shoulder_Pitch",
            "Left_Shoulder_Roll",
            "Left_Elbow_Pitch",
            "Left_Elbow_Yaw",
            "Left_Wrist_Pitch",
            "Left_Wrist_Yaw",
            "Left_Hand_Roll",
        ]
    else:
        t1_real_names = [
            "Right_Shoulder_Pitch",
            "Right_Shoulder_Roll",
            "Right_Elbow_Pitch",
            "Right_Elbow_Yaw",
            "Right_Wrist_Pitch",
            "Right_Wrist_Yaw",
            "Right_Hand_Roll",
        ]

    return [
        t1_real_names,
        [f"joint_arm_{side_s}_{i:02d}" for i in range(1, 8)],
        [f"joint_arm_{side_l}_{i:02d}" for i in range(1, 8)],
        [f"arm_{side_s}_{i:02d}" for i in range(1, 8)],
        [f"arm_{side_l}_{i:02d}" for i in range(1, 8)],
        [f"{side_l}_arm_joint_{i}" for i in range(1, 8)],
        [f"{side_s}_arm_joint_{i}" for i in range(1, 8)],
    ]


def resolve_arm_joint_names(cfg: dict[str, Any], model: Any, side: str) -> list[str]:
    all_names = [str(n) for n in model.names]
    all_name_set = set(all_names)

    from_cfg = _candidate_joint_names_from_config(cfg, side)
    if from_cfg is not None:
        missing = [n for n in from_cfg if n not in all_name_set]
        if missing:
            raise RuntimeError(
                "Joint names were configured manually, but some were not found in Pinocchio model:\n"
                f"  missing = {missing}\n"
                f"  configured names = {from_cfg}\n"
                f"  available 1-DoF joints = {available_1dof_joint_names(model)}"
            )
        return from_cfg

    for cand in _fallback_joint_name_sets(side):
        if all(n in all_name_set for n in cand):
            return cand

    side_l = _side_long(side)
    movable = available_1dof_joint_names(model)

    prefix = "Left_" if side_l == "left" else "Right_"
    preferred_order = [
        "Shoulder_Pitch",
        "Shoulder_Roll",
        "Elbow_Pitch",
        "Elbow_Yaw",
        "Wrist_Pitch",
        "Wrist_Yaw",
        "Hand_Roll",
    ]

    candidates = [
        n for n in movable
        if n.startswith(prefix)
        and any(k.lower() in n.lower() for k in ["shoulder", "elbow", "wrist", "hand"])
    ]

    def arm_order_key(name: str) -> tuple[int, list[Any]]:
        nl = name.lower()
        for i, key in enumerate(preferred_order):
            if key.lower() in nl:
                return i, _natural_key(name)
        return 999, _natural_key(name)

    candidates = sorted(candidates, key=arm_order_key)

    if len(candidates) >= 7:
        return candidates[:7]

    raise RuntimeError(
        "Cannot infer the 7 T1 arm joint names.\n\n"
        "当前 URDF 里的 1-DoF joints 是：\n"
        f"  {movable}\n\n"
        "对于当前 T1 URDF，左臂应该是：\n"
        "  ['Left_Shoulder_Pitch', 'Left_Shoulder_Roll', 'Left_Elbow_Pitch', "
        "'Left_Elbow_Yaw', 'Left_Wrist_Pitch', 'Left_Wrist_Yaw', 'Left_Hand_Roll']\n"
        "右臂应该是：\n"
        "  ['Right_Shoulder_Pitch', 'Right_Shoulder_Roll', 'Right_Elbow_Pitch', "
        "'Right_Elbow_Yaw', 'Right_Wrist_Pitch', 'Right_Wrist_Yaw', 'Right_Hand_Roll']"
    )


def resolve_payload_joint_name(cfg: dict[str, Any], side: str) -> str | None:
    side_l = _side_long(side)
    side_s = _side_short(side)

    if MANUAL_EE_JOINT_NAME is not None and MANUAL_EE_JOINT_NAME.strip():
        return MANUAL_EE_JOINT_NAME.strip()

    manual = MANUAL_EE_JOINT_NAME_BY_SIDE.get(side_l)
    if manual is None:
        manual = MANUAL_EE_JOINT_NAME_BY_SIDE.get(side_s)
    if isinstance(manual, str) and manual.strip():
        return manual.strip()

    for keys in [
        ("payload_joint_name",),
        ("end_effector_joint_name",),
        ("ee_joint_name",),
        ("robot", "payload_joint_name"),
        ("robot", "end_effector_joint_name"),
        ("robot", "ee_joint_name"),
        ("identification", "payload_joint_name"),
        ("identification", "end_effector_joint_name"),
        ("identification", "ee_joint_name"),
    ]:
        v = _nested_get(cfg, keys)
        if isinstance(v, str) and v.strip():
            return v.strip()

    for keys in [
        ("payload_joint_name_by_side",),
        ("end_effector_joint_name_by_side",),
        ("ee_joint_name_by_side",),
        ("robot", "payload_joint_name_by_side"),
        ("robot", "end_effector_joint_name_by_side"),
        ("robot", "ee_joint_name_by_side"),
        ("identification", "payload_joint_name_by_side"),
        ("identification", "end_effector_joint_name_by_side"),
        ("identification", "ee_joint_name_by_side"),
    ]:
        v = _nested_get(cfg, keys)
        if isinstance(v, dict):
            for sk in (side_l, side_s):
                vv = v.get(sk)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()

    return None


def resolve_alpha(cfg: dict[str, Any], side: str, n: int = 7) -> np.ndarray:
    side_l = _side_long(side)
    side_s = _side_short(side)

    containers = [
        cfg.get("identification", {}),
        cfg.get("online", {}),
        cfg.get("friction", {}),
        cfg,
    ]

    for c in containers:
        if not isinstance(c, dict):
            continue

        for key in ["alpha0_by_side", "alpha_by_side", "friction_alpha_by_side"]:
            v = c.get(key)
            if isinstance(v, dict):
                for sk in (side_l, side_s):
                    if sk in v:
                        arr = np.asarray(v[sk], dtype=float).reshape(-1)
                        if arr.size == n:
                            return arr

        for key in ["alpha0", "alpha", "friction_alpha", "nonlinear_friction_alpha"]:
            v = c.get(key)
            if isinstance(v, list):
                arr = np.asarray(v, dtype=float).reshape(-1)
                if arr.size == n:
                    return arr

    return np.ones(n, dtype=float)


# =============================================================================
# 4. Pinocchio T1 动力学回归器
# =============================================================================

class PinocchioT1RegressorBuilder:
    """
    直接使用 Pinocchio 构建 T1 手臂动力学回归器。

    外部输入：
        q, qd, qdd: shape = (7,)

    内部：
        全身 q 用 neutral；
        非选中关节速度/加速度设为 0；
        只抽取当前 7 个手臂关节对应 torque rows。
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        side: str,
        urdf_path: str | Path,
        joint_names: list[str] | None = None,
        payload_joint_name: str | None = None,
        alpha: np.ndarray | None = None,
    ):
        try:
            import pinocchio as pin  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Pinocchio import failed.\n"
                "建议优先使用 conda 安装：\n"
                "  conda create -n t1payload python=3.10 -y\n"
                "  conda activate t1payload\n"
                "  conda install -c conda-forge pinocchio -y\n"
            ) from exc

        self.pin = pin
        self.cfg = cfg
        self.side = _side_long(side)
        self.urdf_path = Path(urdf_path).resolve()

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.reg_data = self.model.createData()
        self.q_neutral = np.asarray(pin.neutral(self.model), dtype=float).copy()

        if joint_names is None:
            joint_names = resolve_arm_joint_names(cfg, self.model, side)

        self.joint_names = [str(n) for n in joint_names]
        self.joint_ids = [int(self.model.getJointId(n)) for n in self.joint_names]

        bad = [
            (n, jid)
            for n, jid in zip(self.joint_names, self.joint_ids)
            if jid <= 0 or jid >= self.model.njoints
        ]
        if bad:
            raise RuntimeError(
                f"Could not map selected joints into Pinocchio model: {bad}\n"
                f"Available 1-DoF joints: {available_1dof_joint_names(self.model)}"
            )

        self.nq_per_joint = np.array([int(self.model.nqs[jid]) for jid in self.joint_ids], dtype=int)
        self.nv_per_joint = np.array([int(self.model.nvs[jid]) for jid in self.joint_ids], dtype=int)

        if not (np.all(self.nq_per_joint == 1) and np.all(self.nv_per_joint == 1)):
            raise RuntimeError(
                "This generator expects seven 1-DoF revolute joints.\n"
                f"joint_names = {self.joint_names}\n"
                f"nq_per_joint = {self.nq_per_joint.tolist()}\n"
                f"nv_per_joint = {self.nv_per_joint.tolist()}"
            )

        self.q_indices = np.array([int(self.model.idx_qs[jid]) for jid in self.joint_ids], dtype=int)
        self.v_indices = np.array([int(self.model.idx_vs[jid]) for jid in self.joint_ids], dtype=int)

        if payload_joint_name is not None:
            payload_jid = int(self.model.getJointId(payload_joint_name))
            if payload_jid <= 0 or payload_jid >= self.model.njoints:
                raise RuntimeError(
                    f"payload_joint_name={payload_joint_name!r} not found in Pinocchio model."
                )
            self.payload_joint_name = str(payload_joint_name)
            self.payload_joint_id = payload_jid
        else:
            self.payload_joint_name = self.joint_names[-1]
            self.payload_joint_id = self.joint_ids[-1]

        self.alpha = np.ones(7, dtype=float) if alpha is None else np.asarray(alpha, dtype=float).reshape(7)

        self.regressor_cols = self._infer_regressor_column_count()
        self.payload_col_slice = self._infer_payload_column_slice()

    def _infer_regressor_column_count(self) -> int:
        q7 = np.zeros(7, dtype=float)
        Y = self.sample_inertial_regressor(q7, q7, q7)
        return int(Y.shape[1])

    def _infer_payload_column_slice(self) -> slice:
        """
        Pinocchio regressor columns are grouped by joint body, 10 columns each.

        """
        ncol = int(self.regressor_cols)
        jid = int(self.payload_joint_id)

        if ncol == 10 * (self.model.njoints - 1):
            st = 10 * (jid - 1)
        elif ncol == 10 * self.model.njoints:
            st = 10 * jid
        else:
            st_no_universe = 10 * (jid - 1)
            st_with_universe = 10 * jid

            if 0 <= st_no_universe and st_no_universe + 10 <= ncol:
                st = st_no_universe
            elif 0 <= st_with_universe and st_with_universe + 10 <= ncol:
                st = st_with_universe
            else:
                raise RuntimeError(
                    "Cannot infer payload inertial-parameter column block in Pinocchio regressor.\n"
                    f"regressor_cols = {ncol}\n"
                    f"model.njoints = {self.model.njoints}\n"
                    f"payload_joint_id = {jid}\n"
                    f"payload_joint_name = {self.payload_joint_name}"
                )

        if not (0 <= st and st + 10 <= ncol):
            raise RuntimeError(
                "Payload regressor column slice out of range.\n"
                f"regressor_cols = {ncol}\n"
                f"start = {st}\n"
                f"payload_joint_id = {jid}\n"
                f"payload_joint_name = {self.payload_joint_name}"
            )

        return slice(st, st + 10)

    def fill_full_state(
        self,
        q7: np.ndarray,
        qd7: np.ndarray,
        qdd7: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q7 = np.asarray(q7, dtype=float).reshape(7)
        qd7 = np.asarray(qd7, dtype=float).reshape(7)
        qdd7 = np.asarray(qdd7, dtype=float).reshape(7)

        q_full = self.q_neutral.copy()
        v_full = np.zeros(self.model.nv, dtype=float)
        a_full = np.zeros(self.model.nv, dtype=float)

        q_full[self.q_indices] = q7
        v_full[self.v_indices] = qd7
        a_full[self.v_indices] = qdd7

        return q_full, v_full, a_full

    def sample_inertial_regressor(
        self,
        q7: np.ndarray,
        qd7: np.ndarray,
        qdd7: np.ndarray,
    ) -> np.ndarray:
        q_full, v_full, a_full = self.fill_full_state(q7, qd7, qdd7)

        Y = self.pin.computeJointTorqueRegressor(
            self.model,
            self.reg_data,
            q_full,
            v_full,
            a_full,
        )

        if Y is None:
            Y = getattr(self.reg_data, "jointTorqueRegressor", None)

        if Y is None:
            raise RuntimeError(
                "pin.computeJointTorqueRegressor returned None and "
                "data.jointTorqueRegressor was not found."
            )

        Y = np.asarray(Y, dtype=float)

        if Y.ndim != 2 or Y.shape[0] != self.model.nv:
            raise RuntimeError(
                f"Unexpected Pinocchio torque regressor shape: {Y.shape}, "
                f"expected first dimension {self.model.nv}"
            )

        return Y[self.v_indices, :]

    def sample_inverse_dynamics(
        self,
        q7: np.ndarray,
        qd7: np.ndarray,
        qdd7: np.ndarray,
    ) -> np.ndarray:
        q_full, v_full, a_full = self.fill_full_state(q7, qd7, qdd7)
        tau_full = self.pin.rnea(self.model, self.data, q_full, v_full, a_full)
        tau_full = np.asarray(tau_full, dtype=float).reshape(self.model.nv)
        return tau_full[self.v_indices].reshape(7)

    def sample_friction_regressor(
        self,
        qd7: np.ndarray,
        alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        非线性摩擦模型：
            tau_f,j = Fc_j * sign(qd_j)
                    + Fv_j * |qd_j|^alpha_j * sign(qd_j)
                    + B_j

        输出：
            shape = 7 x 21
            columns = [Fc1, Fv1, B1, Fc2, Fv2, B2, ..., Fc7, Fv7, B7]
        """
        qd7 = np.asarray(qd7, dtype=float).reshape(7)
        a = self.alpha if alpha is None else np.asarray(alpha, dtype=float).reshape(7)

        F = np.zeros((7, 21), dtype=float)
        for j in range(7):
            s = float(np.sign(qd7[j]))
            F[j, 3 * j + 0] = s
            F[j, 3 * j + 1] = (abs(float(qd7[j])) ** float(a[j])) * s
            F[j, 3 * j + 2] = 1.0

        return F

    def stack_payload_regressor(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)

        blocks: list[np.ndarray] = []
        for i in range(q.shape[0]):
            Yi = self.sample_inertial_regressor(q[i], qd[i], qdd[i])
            blocks.append(Yi[:, self.payload_col_slice])

        return np.vstack(blocks) if blocks else np.zeros((0, 10), dtype=float)

    def stack_full_regressor(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        alpha: np.ndarray | None = None,
        include_friction: bool = True,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)

        blocks: list[np.ndarray] = []
        for i in range(q.shape[0]):
            Yi = self.sample_inertial_regressor(q[i], qd[i], qdd[i])
            if include_friction:
                Fi = self.sample_friction_regressor(qd[i], alpha=alpha)
                Yi = np.hstack([Yi, Fi])
            blocks.append(Yi)

        return np.vstack(blocks) if blocks else np.zeros((0, 0), dtype=float)

    def stack_payload_and_full_regressor(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        alpha: np.ndarray | None = None,
        include_friction: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)

        payload_blocks: list[np.ndarray] = []
        full_blocks: list[np.ndarray] = []

        for i in range(q.shape[0]):
            Yi = self.sample_inertial_regressor(q[i], qd[i], qdd[i])
            payload_blocks.append(Yi[:, self.payload_col_slice])

            if include_friction:
                Fi = self.sample_friction_regressor(qd[i], alpha=alpha)
                full_blocks.append(np.hstack([Yi, Fi]))
            else:
                full_blocks.append(Yi)

        Wp = np.vstack(payload_blocks) if payload_blocks else np.zeros((0, 10), dtype=float)
        Yfull = np.vstack(full_blocks) if full_blocks else np.zeros((0, 0), dtype=float)
        return Wp, Yfull

    def stack_inverse_dynamics(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)

        tau = [
            self.sample_inverse_dynamics(q[i], qd[i], qdd[i])
            for i in range(q.shape[0])
        ]

        return np.vstack(tau) if tau else np.zeros((0, 7), dtype=float)

    def summary(self) -> dict[str, Any]:
        return {
            "urdf_path": str(self.urdf_path),
            "side": self.side,
            "model_nq": int(self.model.nq),
            "model_nv": int(self.model.nv),
            "model_njoints": int(self.model.njoints),
            "joint_names": list(self.joint_names),
            "joint_ids": [int(x) for x in self.joint_ids],
            "q_indices": [int(x) for x in self.q_indices],
            "v_indices": [int(x) for x in self.v_indices],
            "payload_joint_name": self.payload_joint_name,
            "payload_joint_id": int(self.payload_joint_id),
            "payload_col_slice": [
                int(self.payload_col_slice.start),
                int(self.payload_col_slice.stop),
            ],
            "regressor_cols": int(self.regressor_cols),
            "alpha": self.alpha.tolist(),
            "note": (
                "Payload column block is used only for excitation optimization. "
                "The final online estimator uses the direct payload regressor with the momentum-observer residual."
            ),
        }


# =============================================================================
# 5. 目标函数
# =============================================================================

def make_objective_indices(n: int, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    idx = np.arange(0, n, stride, dtype=int)
    if idx.size == 0 or idx[-1] != n - 1:
        idx = np.r_[idx, n - 1]
    return np.unique(idx)


def _select_samples(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=float)[idx]


def objective(
    z: np.ndarray,
    t: np.ndarray,
    q_center: np.ndarray,
    H: int,
    w0: float,
    q_min: np.ndarray,
    q_max: np.ndarray,
    vlim: float,
    alim: float,
    builder: PinocchioT1RegressorBuilder | None,
    obj_idx: np.ndarray,
    alpha: np.ndarray,
    penalty_weight: float,
    hard_penalty: float,
    full_regressor_weight: float,
) -> float:
    try:
        q, qd, qdd, pen = trajectory_from_z(
            z,
            t,
            q_center,
            H,
            w0,
            q_min,
            q_max,
            vlim,
            alim,
        )
    except Exception:
        return 1.0e12

    if not (
        np.all(np.isfinite(q))
        and np.all(np.isfinite(qd))
        and np.all(np.isfinite(qdd))
    ):
        return 1.0e12

    pen = _finite_or_large(float(pen), large=1.0e6)

    if pen > hard_penalty:
        return float(1.0e9 + penalty_weight * pen)

    q_obj = _select_samples(q, obj_idx)
    qd_obj = _select_samples(qd, obj_idx)
    qdd_obj = _select_samples(qdd, obj_idx)

    if builder is None:
        richness = float(
            np.sum(np.std(qd_obj, axis=0))
            + 0.25 * np.sum(np.std(qdd_obj, axis=0))
        )
        return float(1.0 / max(richness, 1.0e-6) + penalty_weight * pen)

    try:
        Wp, Yfull = builder.stack_payload_and_full_regressor(
            q_obj,
            qd_obj,
            qdd_obj,
            alpha=alpha,
            include_friction=True,
        )

        payload_score = log_condition_score(Wp, target_rank=10, rank_penalty=2.5)

        if full_regressor_weight > 0.0:
            full_score = log_condition_score(Yfull, target_rank=None, rank_penalty=0.0)
        else:
            full_score = 0.0

        richness = float(
            np.sum(np.std(qd_obj, axis=0))
            + 0.1 * np.sum(np.std(qdd_obj, axis=0))
        )
        richness_bonus = -0.02 * math.log10(max(richness, 1.0e-6))

        return float(
            payload_score
            + full_regressor_weight * full_score
            + penalty_weight * pen
            + richness_bonus
        )

    except Exception as exc:
        print(f"[warn] regressor objective failed once: {type(exc).__name__}: {exc}")
        return 1.0e12


# =============================================================================
# 6. 主程序
# =============================================================================

def main() -> None:
    print("=" * 90)
    print("[generate_excitation] PyCharm direct-run started")
    print(f"[generate_excitation] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[generate_excitation] CONFIG       = {CONFIG_REL_PATH}")
    print(f"[generate_excitation] SIDE         = {SIDE}")

    cfg_path = PROJECT_ROOT / CONFIG_REL_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = load_json(str(cfg_path))
    ecfg = cfg["excitation"].copy()
    icfg = cfg.get("identification", {})

    side = _side_long(SIDE)

    T = float(ecfg["duration"])
    fs = float(ecfg["sample_rate"])
    H = int(ecfg["n_harmonics"])
    w0 = 2.0 * np.pi * float(ecfg["base_frequency"])

    t = np.arange(0.0, T + 1.0e-12, 1.0 / fs)

    q_center = np.asarray(
        ecfg.get("center_q_by_side", {}).get(side, ecfg["center_q"]),
        dtype=float,
    ).reshape(7)

    q_min = np.asarray(
        ecfg.get("min_q_by_side", {}).get(side, ecfg["min_q"]),
        dtype=float,
    ).reshape(7)

    q_max = np.asarray(
        ecfg.get("max_q_by_side", {}).get(side, ecfg["max_q"]),
        dtype=float,
    ).reshape(7)

    vlim = float(ecfg["velocity_limit_rad_s"])
    alim = float(ecfg["acceleration_limit_rad_s2"])

    rng = np.random.default_rng(int(ecfg.get("random_seed", 100)))

    basis = fourier_nullspace_basis(H)
    zdim = 7 * int(basis.shape[1])

    alpha = resolve_alpha(cfg, side, n=7)

    objective_stride = OBJECTIVE_STRIDE
    if objective_stride is None:
        objective_stride = int(ecfg.get("objective_stride", max(1, round(fs / 25.0))))
    objective_stride = max(1, int(objective_stride))
    obj_idx = make_objective_indices(len(t), objective_stride)

    penalty_weight = float(ecfg.get("constraint_penalty_weight", 1000.0))
    hard_penalty = float(ecfg.get("hard_penalty", 1000.0))
    full_regressor_weight = float(ecfg.get("full_regressor_weight", 0.15))

    random_trials = (
        int(RANDOM_TRIALS_OVERRIDE)
        if RANDOM_TRIALS_OVERRIDE is not None
        else int(ecfg.get("random_trials", 300))
    )

    local_refine_steps = (
        int(LOCAL_REFINE_STEPS_OVERRIDE)
        if LOCAL_REFINE_STEPS_OVERRIDE is not None
        else int(ecfg.get("local_refine_steps", 120))
    )

    print(f"[generate_excitation] duration     = {T:.3f} s")
    print(f"[generate_excitation] sample_rate  = {fs:.1f} Hz")
    print(f"[generate_excitation] samples      = {len(t)}")
    print(f"[generate_excitation] harmonics    = {H}")
    print(f"[generate_excitation] zdim         = {zdim}")
    print(f"[generate_excitation] random       = {random_trials}")
    print(f"[generate_excitation] refine       = {local_refine_steps}")

    builder: PinocchioT1RegressorBuilder | None = None
    urdf_path: Path | None = None

    if not NO_DYNAMICS_OBJECTIVE:
        try:
            urdf_path = resolve_urdf_path_from_config(cfg)
            payload_joint_name = resolve_payload_joint_name(cfg, side)

            builder = PinocchioT1RegressorBuilder(
                cfg=cfg,
                side=side,
                urdf_path=urdf_path,
                joint_names=None,
                payload_joint_name=payload_joint_name,
                alpha=alpha,
            )

            print("[OK] Pinocchio regressor objective enabled.")
            print(f"     URDF              : {urdf_path}")
            print(f"     arm joints        : {builder.joint_names}")
            print(f"     q indices         : {builder.q_indices.tolist()}")
            print(f"     v indices         : {builder.v_indices.tolist()}")
            print(f"     payload joint     : {builder.payload_joint_name}")
            print(f"     payload col slice : {builder.payload_col_slice.start}:{builder.payload_col_slice.stop}")
            print(f"     objective samples : {len(obj_idx)} / {len(t)}  stride={objective_stride}")

        except Exception as exc:
            if ALLOW_KINEMATIC_FALLBACK:
                print(
                    f"[WARN] Pinocchio unavailable or URDF/joint mapping issue: "
                    f"{type(exc).__name__}: {exc}\n"
                    "       Falling back to kinematic-rich trajectory search.\n"
                    "       This is only for smoke testing, not final payload identification."
                )
                builder = None
            else:
                raise RuntimeError(
                    "Pinocchio dynamics objective failed.\n"
                    "最终做动量观测器负载辨识时，不建议 fallback，应该先修复 Pinocchio / URDF / joint name 映射。\n\n"
                    f"Original error: {type(exc).__name__}: {exc}\n\n"
                    "如果只是临时测试轨迹导出，可以把顶部配置改成：\n"
                    "  ALLOW_KINEMATIC_FALLBACK = True\n"
                ) from exc
    else:
        print("[WARN] NO_DYNAMICS_OBJECTIVE=True. 当前轨迹不会做动力学激励优化。")

    best: tuple[float, np.ndarray] | None = None

    print("[generate_excitation] random search started...")
    for i in range(random_trials):
        z = rng.normal(0.0, float(ecfg.get("random_z_std", 0.22)), size=zdim)

        val = objective(
            z=z,
            t=t,
            q_center=q_center,
            H=H,
            w0=w0,
            q_min=q_min,
            q_max=q_max,
            vlim=vlim,
            alim=alim,
            builder=builder,
            obj_idx=obj_idx,
            alpha=alpha,
            penalty_weight=penalty_weight,
            hard_penalty=hard_penalty,
            full_regressor_weight=full_regressor_weight,
        )

        if best is None or val < best[0]:
            best = (float(val), z.copy())
            print(f"random {i:04d}: best score={val:.6g}")

    if best is None:
        raise RuntimeError("No candidate trajectory was generated.")

    best_val, best_z = best

    print("[generate_excitation] local refine started...")
    step = float(ecfg.get("local_refine_initial_step", 0.08))
    step_decay = float(ecfg.get("local_refine_step_decay", 0.995))

    for i in range(local_refine_steps):
        cand = best_z + rng.normal(0.0, step, size=zdim)

        val = objective(
            z=cand,
            t=t,
            q_center=q_center,
            H=H,
            w0=w0,
            q_min=q_min,
            q_max=q_max,
            vlim=vlim,
            alim=alim,
            builder=builder,
            obj_idx=obj_idx,
            alpha=alpha,
            penalty_weight=penalty_weight,
            hard_penalty=hard_penalty,
            full_regressor_weight=full_regressor_weight,
        )

        if val < best_val:
            best_val, best_z = float(val), cand.copy()
            print(f"refine {i:04d}: best score={val:.6g}")

        step *= step_decay

    q, qd, qdd, penalty = trajectory_from_z(
        best_z,
        t,
        q_center,
        H,
        w0,
        q_min,
        q_max,
        vlim,
        alim,
    )

    prefix = OUT_PREFIX or f"trajectories/{side}_arm_payload_excitation"

    out_npz = resolve_project_path(prefix + ".npz")
    out_csv = resolve_project_path(prefix + ".csv")
    out_mat = resolve_project_path(prefix + ".mat")
    out_json = resolve_project_path(prefix + "_report.json")

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz,
        t=t,
        q=q,
        qd=qd,
        qdd=qdd,
        side=np.array(side),
        z=best_z,
        duration=np.array(T),
        sample_rate=np.array(fs),
        penalty=np.array(penalty),
        objective_score=np.array(best_val),
        objective_stride=np.array(objective_stride),
        used_pinocchio=np.array(builder is not None),
    )

    save_csv(out_csv, {"t": t, "q": q, "qd": qd, "qdd": qdd})

    savemat(
        out_mat,
        {
            "t": t,
            "q": q,
            "qd": qd,
            "qdd": qdd,
            "side": side,
            "objective_score": float(best_val),
            "constraint_penalty": float(penalty),
            "used_pinocchio": bool(builder is not None),
        },
    )

    report: dict[str, Any] = {
        "side": side,
        "used_pinocchio": bool(builder is not None),
        "score": float(best_val),
        "constraint_penalty": float(penalty),
        "duration": float(T),
        "sample_rate": float(fs),
        "num_samples": int(len(t)),
        "n_harmonics": int(H),
        "base_frequency": float(ecfg["base_frequency"]),
        "objective_stride": int(objective_stride),
        "objective_num_samples": int(len(obj_idx)),
        "constraint_penalty_weight": float(penalty_weight),
        "hard_penalty": float(hard_penalty),
        "full_regressor_weight": float(full_regressor_weight),
        "q_min_observed": q.min(axis=0).tolist(),
        "q_max_observed": q.max(axis=0).tolist(),
        "qd_abs_max": np.max(np.abs(qd), axis=0).tolist(),
        "qdd_abs_max": np.max(np.abs(qdd), axis=0).tolist(),
        "q_center": q_center.tolist(),
        "q_min_limit": q_min.tolist(),
        "q_max_limit": q_max.tolist(),
        "velocity_limit_rad_s": float(vlim),
        "acceleration_limit_rad_s2": float(alim),
        "alpha_used_for_friction_regressor": alpha.tolist(),
        "files": {
            "npz": str(Path(out_npz).resolve()),
            "csv": str(Path(out_csv).resolve()),
            "mat": str(Path(out_mat).resolve()),
        },
        "note": (
            "Run matlab/validate_excitation_t1.m before executing this on hardware. "
            "For publication-grade momentum-observer payload identification, "
            "used_pinocchio must be true."
        ),
    }

    if urdf_path is not None:
        report["urdf_path"] = str(urdf_path)

    if builder is not None:
        report["pinocchio_builder"] = builder.summary()

        try:
            q_obj = _select_samples(q, obj_idx)
            qd_obj = _select_samples(qd, obj_idx)
            qdd_obj = _select_samples(qdd, obj_idx)

            Wp_obj, Yfull_obj = builder.stack_payload_and_full_regressor(
                q_obj,
                qd_obj,
                qdd_obj,
                alpha=alpha,
                include_friction=True,
            )
            tau_nom_obj = builder.stack_inverse_dynamics(q_obj, qd_obj, qdd_obj)

            report["payload_regressor_condition"] = cond_score(Wp_obj)
            report["payload_regressor_spectrum"] = matrix_spectrum(Wp_obj)
            report["payload_regressor_shape"] = list(Wp_obj.shape)

            report["full_regressor_condition_eff"] = cond_score(Yfull_obj)
            report["full_regressor_spectrum"] = matrix_spectrum(Yfull_obj)
            report["full_regressor_shape"] = list(Yfull_obj.shape)

            report["nominal_inverse_dynamics_tau_abs_max_obj_samples"] = (
                np.max(np.abs(tau_nom_obj), axis=0).tolist()
                if tau_nom_obj.size
                else []
            )

            report_stride = int(ecfg.get("report_full_regressor_stride", max(1, objective_stride)))
            report_idx = make_objective_indices(len(t), report_stride)

            q_rep = _select_samples(q, report_idx)
            qd_rep = _select_samples(qd, report_idx)
            qdd_rep = _select_samples(qdd, report_idx)

            Wp_rep = builder.stack_payload_regressor(q_rep, qd_rep, qdd_rep)

            report["payload_regressor_condition_report_samples"] = cond_score(Wp_rep)
            report["payload_regressor_spectrum_report_samples"] = matrix_spectrum(Wp_rep)
            report["report_regressor_stride"] = int(report_stride)
            report["report_num_samples"] = int(len(report_idx))

        except Exception as exc:
            report["regressor_report_error"] = f"{type(exc).__name__}: {exc}"

    save_json(report, out_json)

    print("[OK] saved files:")
    print(f"  - {out_npz}")
    print(f"  - {out_csv}")
    print(f"  - {out_mat}")
    print(f"  - {out_json}")

    print("[generate_excitation] final summary:")
    print(f"  used_pinocchio      = {bool(builder is not None)}")
    print(f"  score               = {float(best_val):.6g}")
    print(f"  constraint_penalty  = {float(penalty):.6g}")
    print(f"  qd_abs_max          = {np.max(np.abs(qd), axis=0)}")
    print(f"  qdd_abs_max         = {np.max(np.abs(qdd), axis=0)}")

    if builder is not None:
        print(f"  payload_cond        = {report.get('payload_regressor_condition', 'n/a')}")
        spec = report.get("payload_regressor_spectrum", {})
        print(f"  payload_rank        = {spec.get('rank', 'n/a')}")
        print("[OK] Dynamics-optimized excitation generation finished.")
    else:
        print(
            "[WARN] This trajectory was generated without Pinocchio dynamics objective. "
            "Do not use it as final paper-level payload-identification excitation."
        )

    print("=" * 90)


def apply_cli_overrides() -> None:
    global CONFIG_REL_PATH, SIDE, OUT_PREFIX, MANUAL_PROJECT_ROOT, MANUAL_URDF_PATH, PROJECT_ROOT
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--side", choices=["left", "right"], default=None)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--urdf", default=None)
    args = parser.parse_args()
    if args.config is not None:
        CONFIG_REL_PATH = args.config
    if args.side is not None:
        SIDE = args.side
    if args.out_prefix is not None:
        OUT_PREFIX = args.out_prefix
    if args.project_root is not None:
        MANUAL_PROJECT_ROOT = args.project_root
        PROJECT_ROOT = Path(args.project_root).expanduser().resolve()
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    if args.urdf is not None:
        MANUAL_URDF_PATH = args.urdf


if __name__ == "__main__":
    try:
        apply_cli_overrides()
        main()
    except Exception as exc:
        if PRINT_TRACEBACK_ON_ERROR:
            traceback.print_exc()
        else:
            print("=" * 90)
            print(f"[ERROR] {type(exc).__name__}: {exc}")
            print("=" * 90)
        sys.exit(1)
