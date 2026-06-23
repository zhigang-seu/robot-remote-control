# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pinocchio as pin
try:
    import pinocchio.casadi as cpin
    import casadi as ca
except Exception:
    cpin = None
    ca = None
import time
import os
import sys
import pickle

# Set print options
np.set_printoptions(precision=5, suppress=True, linewidth=200)

class WeightedMovingFilter:
    def __init__(self, weights, dim):
        self.weights = np.array(weights) / np.sum(weights)
        self.dim = dim
        self.data_buffer = np.zeros((len(weights), dim))
        self.index = 0
        self.full = False

    def add_data(self, data):
        self.data_buffer[self.index] = data
        self.index = (self.index + 1) % len(self.weights)
        if not self.full and self.index == 0:
            self.full = True
    @property
    def filtered_data(self):
        if self.full:
            return np.dot(self.weights, self.data_buffer)
        else:
            valid_weights = self.weights[:self.index] / np.sum(self.weights[:self.index])
            return np.dot(valid_weights, self.data_buffer[:self.index])

class T17DofArmIK:
    def __init__(self, visualization=False, unit_test=False):
        # Initialize T1 robot dual arm IK solver
        self.visualization = visualization
        self.unit_test = unit_test
        # Set path
        sdk_root_env = os.environ.get('BOOSTER_SDK_ROOT', '').strip()
        local_base = Path(__file__).resolve().parent
        assets_dir = local_base / 'assets'
        if sdk_root_env:
            sdk_assets = Path(sdk_root_env).expanduser().resolve() / 'example' / 'low_level' / 'assets'
            if sdk_assets.exists():
                assets_dir = sdk_assets
        self.urdf_path = str(assets_dir / 'T1_7DofArm_Serial.urdf')
        self.mesh_dir = str(assets_dir)
        # Cache file path
        self.cache_path = str(local_base / 't1_robot_cache.pkl')
        print(">>> Loading URDF file...")
        
        if os.path.exists(self.cache_path) and (not self.visualization):
            print(f">>> Loading cached robot model: {self.cache_path}")
            self.robot = self.load_cache()
        else:
            # Load complete robot model
            self.robot = pin.RobotWrapper.BuildFromURDF(
                self.urdf_path, 
                self.mesh_dir
            )
            self.save_cache()
            print(f">>> Cache saved to {self.cache_path}")
        
        # Print model information
        print(f">>> Total joints of complete model: {self.robot.model.nq}")
        print(f">>> Velocity dimension of complete model: {self.robot.model.nv}")
        
        # Create a mapping of arm joints
        self.arm_joint_names = [
            "Left_Shoulder_Pitch",
            "Left_Shoulder_Roll", 
            "Left_Elbow_Pitch",
            "Left_Elbow_Yaw",
            "Left_Wrist_Pitch",
            "Left_Wrist_Yaw",
            "Left_Hand_Roll",
            "Right_Shoulder_Pitch",
            "Right_Shoulder_Roll",
            "Right_Elbow_Pitch",
            "Right_Elbow_Yaw",
            "Right_Wrist_Pitch",
            "Right_Wrist_Yaw",
            "Right_Hand_Roll"
        ]
        
        # Get arm joint indices
        self.arm_joint_indices = []
        self.arm_velocity_indices = []
        self.arm_joint_ids = []
        print("\n=== Arm Joint Information ===")
        for joint_name in self.arm_joint_names:
            try:
                jid = self.robot.model.getJointId(joint_name)
                if jid < len(self.robot.model.joints):
                    idx_q = self.robot.model.joints[jid].idx_q
                    idx_v = self.robot.model.joints[jid].idx_v
                    self.arm_joint_indices.append(idx_q)
                    if not hasattr(self, "arm_velocity_indices"):
                        self.arm_velocity_indices = []
                    self.arm_velocity_indices.append(idx_v)
                    self.arm_joint_ids.append(jid)
                    print(f"Joint: {joint_name}, ID: {jid}, q_index: {idx_q}, v_index: {idx_v}")
            except Exception as e:
                print(f"Cannot find joint {joint_name}: {e}")
        print(f"\n>>> Found {len(self.arm_joint_indices)} arm joints")
        
        # Set end-effector frame
        self.left_hand_frame_name = "left_hand_link"
        self.right_hand_frame_name = "right_hand_link"
        
        # Get frame ID
        try:
            self.left_hand_id = self.robot.model.getFrameId(self.left_hand_frame_name)
            print(f">>> Left hand frame ID: {self.left_hand_id} (Name: {self.left_hand_frame_name})")
        except:
            print(f">>> Cannot find left hand frame: {self.left_hand_frame_name}")
            self.left_hand_id = None  
        try:
            self.right_hand_id = self.robot.model.getFrameId(self.right_hand_frame_name)
            print(f">>> Right hand frame ID: {self.right_hand_id} (Name: {self.right_hand_frame_name})")
        except:
            print(f">>> Cannot find right hand frame: {self.right_hand_frame_name}")
            self.right_hand_id = None
        # Set initial joint positions
        self.init_q = self.get_initial_joint_positions()
        self.current_q = self.init_q.copy()
        # Set joint limits
        self.joint_lower = []
        self.joint_upper = []
        for idx in self.arm_joint_indices:
            self.joint_lower.append(self.robot.model.lowerPositionLimit[idx])
            self.joint_upper.append(self.robot.model.upperPositionLimit[idx])
        self.joint_lower = np.array(self.joint_lower)
        self.joint_upper = np.array(self.joint_upper)
        print(f"\n>>> Joint limit range:")
        for i, idx in enumerate(self.arm_joint_indices):
            joint_name = self.arm_joint_names[i]
            print(f"  {joint_name}: [{self.joint_lower[i]:.3f}, {self.joint_upper[i]:.3f}]")
        # Velocity-level Jacobian servo parameters.
        # The original file used a CasADi/IPOPT optimization IK here. For teleoperation
        # this is fragile because a single NLP failure makes the controller drop the
        # current target. The public SDK-facing method names are kept, but the default
        # solver below updates joint angles by integrating a damped least-squares
        # Jacobian velocity command:
        #     q_{k+1} = q_k + qdot_k * dt
        #     qdot_k = J# * [Kp * log(T(q_k)^-1 T_target)] + null-space bias
        self.use_casadi_backup = bool(cpin is not None and ca is not None)
        self.ik_dt = 0.015                  # matches the VR solve loop period approximately
        self.ik_servo_iters = 4             # small number: smooth, bounded, real-time friendly
        self.max_linear_vel = 0.8          # m/s, Cartesian command cap inside IK servo
        self.max_angular_vel = 3.0        # rad/s
        self.max_joint_vel = 3.0           # rad/s, per IK output step before controller clipping
        self.linear_gain = 8.0
        self.angular_gain = 6.0
        self.damping = 1e-3
        self.manipulability_threshold = 1e-3
        self.nullspace_gain = 1.5
        self.joint_limit_margin = 0.08      # rad, soft protection near position limits
        self.teleop_accept_partial_solution = True
        self.last_ik_warning_time = 0.0
        self.last_q_arm = self.extract_arm_joints(self.current_q)
        self.last_tau_ff = np.zeros(len(self.arm_joint_indices))
        self._data_servo = self.robot.model.createData()

        # Keep the old IPOPT members only when CasADi is available. They are no longer
        # used in the real-time path, but leaving them available makes this file more
        # backward-compatible with older experiments that may call them manually.
        self.opti = None
        if self.use_casadi_backup:
            try:
                self.cmodel = cpin.Model(self.robot.model)
                self.cdata = self.cmodel.createData()
                self.cq = ca.SX.sym("q", self.robot.model.nq, 1)
                self.cTf_l = ca.SX.sym("tf_l", 4, 4)
                self.cTf_r = ca.SX.sym("tf_r", 4, 4)
                cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
                self.translational_error = ca.Function(
                    "translational_error",
                    [self.cq, self.cTf_l, self.cTf_r],
                    [
                        ca.vertcat(
                            self.cdata.oMf[self.left_hand_id].translation - self.cTf_l[:3, 3],
                            self.cdata.oMf[self.right_hand_id].translation - self.cTf_r[:3, 3],
                        )
                    ],
                )
                self.rotational_error = ca.Function(
                    "rotational_error",
                    [self.cq, self.cTf_l, self.cTf_r],
                    [
                        ca.vertcat(
                            cpin.log3(self.cdata.oMf[self.left_hand_id].rotation @ self.cTf_l[:3, :3].T),
                            cpin.log3(self.cdata.oMf[self.right_hand_id].rotation @ self.cTf_r[:3, :3].T),
                        )
                    ],
                )
                self.opti = ca.Opti()
                self.var_q = self.opti.variable(self.robot.model.nq)
                self.var_q_last = self.opti.parameter(self.robot.model.nq)
                self.param_tf_l = self.opti.parameter(4, 4)
                self.param_tf_r = self.opti.parameter(4, 4)
                self.translational_cost = ca.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
                self.rotation_cost = ca.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
                self.regularization_cost = ca.sumsqr(self.var_q)
                self.smooth_cost = ca.sumsqr(self.var_q - self.var_q_last)
                full_lower = self.robot.model.lowerPositionLimit.copy()
                full_upper = self.robot.model.upperPositionLimit.copy()
                for i in range(self.robot.model.nq):
                    if i not in self.arm_joint_indices:
                        full_lower[i] = self.init_q[i] - 1e-6
                        full_upper[i] = self.init_q[i] + 1e-6
                self.opti.subject_to(self.opti.bounded(full_lower, self.var_q, full_upper))
                self.opti.minimize(
                    50 * self.translational_cost
                    + 1 * self.rotation_cost
                    + 0.02 * self.regularization_cost
                    + 0.1 * self.smooth_cost
                )
                opts = {
                    "expand": True,
                    "detect_simple_bounds": True,
                    "calc_lam_p": False,
                    "print_time": False,
                    "ipopt.sb": "yes",
                    "ipopt.print_level": 0,
                    "ipopt.max_iter": 20,
                    "ipopt.tol": 1e-4,
                    "ipopt.acceptable_tol": 5e-4,
                    "ipopt.acceptable_iter": 5,
                    "ipopt.warm_start_init_point": "yes",
                    "ipopt.derivative_test": "none",
                    "ipopt.jacobian_approximation": "exact",
                }
                self.opti.solver("ipopt", opts)
            except Exception as e:
                print(f">>> CasADi backup IK disabled: {e}")
                self.use_casadi_backup = False
                self.opti = None

        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), len(self.arm_joint_indices))
        self.init_data = self.init_q.copy()
        self.vis = None
        if self.visualization:
            self.setup_visualization()
    
    def save_cache(self):
        data = {"robot_model": self.robot.model}
        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)
    
    def load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)
        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()
        return robot
    
    def get_initial_joint_positions(self):
        # Get initial joint positions
        q = pin.neutral(self.robot.model)
        left_arm_init = np.array([0.1207, -1.3649,  -0.0025, -1.5215, -0.2081, -0.1417, 0.0086,])
        right_arm_init = np.array([0.1207, 1.3649,  -0.0025, 1.5215, -0.2081, 0.1417, 0.0086,])
        for i, idx in enumerate(self.arm_joint_indices[:7]):  # Left arm
            q[idx] = left_arm_init[i]
        for i, idx in enumerate(self.arm_joint_indices[7:]):  # Right arm
            q[idx] = right_arm_init[i]
        return q
    
    def compute_forward_kinematics(self, q, side='left'):
        # Compute forward kinematics
        data = self.robot.data
        if side == 'left' and self.left_hand_id is not None:
            frame_id = self.left_hand_id
        elif side == 'right' and self.right_hand_id is not None:
            frame_id = self.right_hand_id
        else:
            # Return identity matrix if frame not found
            return np.eye(4)
        pin.forwardKinematics(self.robot.model, data, q)
        pin.updateFramePlacement(self.robot.model, data, frame_id)
        return data.oMf[frame_id].homogeneous
    
    def pose_to_xyzrpy(self, pose_matrix):
        position = pose_matrix[:3, 3]
        rotation_matrix = pose_matrix[:3, :3]
        rpy = pin.rpy.matrixToRpy(rotation_matrix)
        return position, rpy
    
    def xyzrpy_to_pose(self, position, rpy):
        rotation_matrix = pin.rpy.rpyToMatrix(rpy[0], rpy[1], rpy[2])
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = position
        return pose_matrix
    
    def print_pose(self, pose_matrix, name="Pose"):
        position, rpy = self.pose_to_xyzrpy(pose_matrix)
        print(f"{name}:")
        print(f"  Position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}] m")
        print(f"  Euler angles(RPY): [{rpy[0]:.4f}, {rpy[1]:.4f}, {rpy[2]:.4f}] rad")
        print(f"  Euler angles(RPY): [{np.degrees(rpy[0]):.2f}, {np.degrees(rpy[1]):.2f}, {np.degrees(rpy[2]):.2f}] deg")
        
        return position, rpy
    
    def setup_visualization(self):
        # Set up visualization
        try:
            from pinocchio.visualize import MeshcatVisualizer
            self.vis = MeshcatVisualizer(
                self.robot.model,
                self.robot.collision_model,
                self.robot.visual_model
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel()
            self.vis.display(self.init_q)
            print(">>> Visualization initialized")
        except Exception as e:
            print(f">>> Visualization initialization failed: {e}")
            self.visualization = False
            self.vis = None
    
    def extract_arm_joints(self, q_full):
        # Extract arm joints from full configuration
        q_arm = np.zeros(len(self.arm_joint_indices))
        for i, idx in enumerate(self.arm_joint_indices):
            q_arm[i] = q_full[idx]
        return q_arm
    
    def set_arm_joints(self, q_full, q_arm):
        # Set arm joints to full configuration
        q_new = q_full.copy()
        for i, idx in enumerate(self.arm_joint_indices):
            q_new[idx] = q_arm[i]
        return q_new
    
    def _normalize_current_q(self, current_q=None):
        """Return a full Pinocchio q vector and copy arm joints from SDK vectors when needed."""
        q = self.current_q.copy()
        if current_q is None:
            return q
        current_q = np.asarray(current_q, dtype=float).reshape(-1)
        if len(current_q) == self.robot.model.nq:
            q = current_q.copy()
        else:
            # Some SDK paths provide the 29-motor full robot vector. Only the arm
            # joint positions are needed by this serial-arm URDF, so copy the
            # overlapping indices and keep the rest at the last valid IK state.
            for idx in self.arm_joint_indices:
                if idx < len(current_q) and idx < len(q):
                    q[idx] = current_q[idx]
        q = np.clip(q, self.robot.model.lowerPositionLimit, self.robot.model.upperPositionLimit)
        return q

    def _pose_error_local(self, current_pose, target_pose):
        """SE(3) body-frame pose error, ordered as [linear, angular]."""
        current_se3 = pin.SE3(current_pose[:3, :3], current_pose[:3, 3])
        target_se3 = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])
        return pin.log(current_se3.inverse() * target_se3).vector

    def _clip_vector_norm(self, vec, max_norm):
        norm = float(np.linalg.norm(vec))
        if norm > max_norm > 0.0:
            return vec * (max_norm / norm)
        return vec

    def _compute_stacked_arm_jacobian_and_error(self, q, left_target, right_target):
        pin.forwardKinematics(self.robot.model, self._data_servo, q)
        pin.computeJointJacobians(self.robot.model, self._data_servo, q)
        pin.updateFramePlacements(self.robot.model, self._data_servo)

        left_pose = self._data_servo.oMf[self.left_hand_id].homogeneous
        right_pose = self._data_servo.oMf[self.right_hand_id].homogeneous

        e_left = self._pose_error_local(left_pose, left_target)
        e_right = self._pose_error_local(right_pose, right_target)

        v_left = self.linear_gain * e_left[:3]
        w_left = self.angular_gain * e_left[3:]
        v_right = self.linear_gain * e_right[:3]
        w_right = self.angular_gain * e_right[3:]

        v_left = self._clip_vector_norm(v_left, self.max_linear_vel)
        v_right = self._clip_vector_norm(v_right, self.max_linear_vel)
        w_left = self._clip_vector_norm(w_left, self.max_angular_vel)
        w_right = self._clip_vector_norm(w_right, self.max_angular_vel)
        desired_twist = np.concatenate([v_left, w_left, v_right, w_right])

        J_left_full = pin.getFrameJacobian(
            self.robot.model, self._data_servo, self.left_hand_id, pin.ReferenceFrame.LOCAL
        )
        J_right_full = pin.getFrameJacobian(
            self.robot.model, self._data_servo, self.right_hand_id, pin.ReferenceFrame.LOCAL
        )
        J_left = J_left_full[:, self.arm_velocity_indices]
        J_right = J_right_full[:, self.arm_velocity_indices]
        J = np.vstack([J_left, J_right])
        return J, desired_twist, e_left, e_right

    def _regularized_pinv(self, J):
        """Damped least-squares pseudo-inverse with adaptive singularity damping."""
        s = np.linalg.svd(J, compute_uv=False)
        manipulability = float(np.prod(np.maximum(s, 1e-8))) if len(s) else 0.0
        damping = self.damping
        if manipulability < self.manipulability_threshold:
            ratio = self.manipulability_threshold / (manipulability + 1e-8)
            damping *= min(ratio, 20.0)
        JJt = J @ J.T
        return J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(JJt.shape[0]), np.eye(JJt.shape[0]))

    def _apply_joint_position_velocity_limits(self, q, qdot_arm, dt):
        qdot_arm = np.asarray(qdot_arm, dtype=float).reshape(-1)
        qdot_arm = np.clip(qdot_arm, -self.max_joint_vel, self.max_joint_vel)

        for i, idx_q in enumerate(self.arm_joint_indices):
            q_min = self.robot.model.lowerPositionLimit[idx_q]
            q_max = self.robot.model.upperPositionLimit[idx_q]
            # Hard one-step feasibility: q + qdot*dt must remain inside limits.
            lower_v = (q_min - q[idx_q]) / max(dt, 1e-6)
            upper_v = (q_max - q[idx_q]) / max(dt, 1e-6)
            qdot_arm[i] = np.clip(qdot_arm[i], lower_v, upper_v)

            # Soft damping near limits to avoid repeatedly hitting clip().
            if q[idx_q] - q_min < self.joint_limit_margin and qdot_arm[i] < 0.0:
                qdot_arm[i] *= max(0.1, (q[idx_q] - q_min) / self.joint_limit_margin)
            if q_max - q[idx_q] < self.joint_limit_margin and qdot_arm[i] > 0.0:
                qdot_arm[i] *= max(0.1, (q_max - q[idx_q]) / self.joint_limit_margin)
        return qdot_arm

    def _jacobian_servo_step(self, q, left_target, right_target, dt):
        J, desired_twist, e_left, e_right = self._compute_stacked_arm_jacobian_and_error(
            q, left_target, right_target
        )
        J_pinv = self._regularized_pinv(J)
        qdot_task = J_pinv @ desired_twist

        # Null-space posture bias. This keeps the 7-DoF arms from drifting while
        # preserving the end-effector velocity as much as possible.
        q_arm = self.extract_arm_joints(q)
        q_des_arm = self.extract_arm_joints(self.init_q)
        qdot0 = -self.nullspace_gain * (q_arm - q_des_arm)
        null_projector = np.eye(len(self.arm_joint_indices)) - J_pinv @ J
        qdot_arm = qdot_task + null_projector @ qdot0
        qdot_arm = self._apply_joint_position_velocity_limits(q, qdot_arm, dt)

        v_full = np.zeros(self.robot.model.nv)
        for i, idx_v in enumerate(self.arm_velocity_indices):
            v_full[idx_v] = qdot_arm[i]
        q_next = pin.integrate(self.robot.model, q, v_full * dt)
        q_next = np.clip(q_next, self.robot.model.lowerPositionLimit, self.robot.model.upperPositionLimit)
        return q_next, e_left, e_right

    def _solve_ik_jacobian_servo(self, left_target, right_target, current_q=None):
        q = self._normalize_current_q(current_q)
        best_q = q.copy()
        best_err = np.inf
        last_e_left = np.zeros(6)
        last_e_right = np.zeros(6)

        for _ in range(max(1, int(self.ik_servo_iters))):
            q, e_left, e_right = self._jacobian_servo_step(q, left_target, right_target, self.ik_dt)
            err_norm = (
                np.linalg.norm(e_left[:3])
                + np.linalg.norm(e_right[:3])
                + 0.20 * (np.linalg.norm(e_left[3:]) + np.linalg.norm(e_right[3:]))
            )
            if err_norm < best_err:
                best_err = err_norm
                best_q = q.copy()
                last_e_left = e_left.copy()
                last_e_right = e_right.copy()

#        q_arm_raw = self.extract_arm_joints(best_q)
#        self.smooth_filter.add_data(q_arm_raw)
        q_arm = self.extract_arm_joints(best_q)
#        q_smooth = self.set_arm_joints(best_q, q_arm)
        self.current_q = best_q.copy()
        self.init_data = best_q.copy()
        self.last_q_arm = q_arm.copy()
        tau_ff_arm = np.zeros(len(self.arm_joint_indices))
        self.last_tau_ff = tau_ff_arm.copy()

        pos_ok = np.linalg.norm(last_e_left[:3]) < 0.025 and np.linalg.norm(last_e_right[:3]) < 0.025
        ori_ok = np.linalg.norm(last_e_left[3:]) < np.radians(15) and np.linalg.norm(last_e_right[3:]) < np.radians(15)
        finite_ok = np.all(np.isfinite(q_arm))
        # For teleoperation, a partial bounded servo step is still a valid command.
        converged = bool(finite_ok and (pos_ok and ori_ok or self.teleop_accept_partial_solution))
        return q_arm, tau_ff_arm, converged

    def solve_ik_optimization(self, left_target, right_target, current_q=None):
        """SDK-compatible IK entry point.

        The name is preserved for existing ROS/SDK code, but the real-time path now
        uses bounded Jacobian velocity servo instead of repeatedly solving an NLP.
        It always returns the best bounded arm command so teleoperation does not
        stutter when an exact pose is temporarily unreachable.
        """
        try:
            return self._solve_ik_jacobian_servo(left_target, right_target, current_q)
        except Exception as e:
            now = time.time()
            if now - self.last_ik_warning_time > 1.0:
                print(f">>> Jacobian servo IK failed, holding last arm command: {e}")
                self.last_ik_warning_time = now
            return self.last_q_arm.copy(), self.last_tau_ff.copy(), False

    # Solve inverse kinematics
    def solve_ik(self, left_target_pose, right_target_pose, current_q=None, visualize=False):
        if current_q is not None:
            self.current_q = self._normalize_current_q(current_q)

        q_arm, tau_ff, solver_ok = self.solve_ik_optimization(
            left_target_pose, right_target_pose, self.current_q
        )
        q_full = self.set_arm_joints(self.current_q, q_arm)
        T_left_sol = self.compute_forward_kinematics(q_full, "left")
        T_right_sol = self.compute_forward_kinematics(q_full, "right")

        pos_error_left = np.linalg.norm(T_left_sol[:3, 3] - left_target_pose[:3, 3])
        pos_error_right = np.linalg.norm(T_right_sol[:3, 3] - right_target_pose[:3, 3])
        R_left_error = T_left_sol[:3, :3].T @ left_target_pose[:3, :3]
        R_right_error = T_right_sol[:3, :3].T @ right_target_pose[:3, :3]
        left_angle_error = np.arccos(np.clip((np.trace(R_left_error) - 1) / 2, -1.0, 1.0))
        right_angle_error = np.arccos(np.clip((np.trace(R_right_error) - 1) / 2, -1.0, 1.0))

        if visualize and self.vis is not None:
            self.vis.display(q_full)

        POS_ERROR_THRESHOLD = 0.03
        POSE_ERROR_THRESHOLD = np.radians(15)
        exact_converged = (
            pos_error_left < POS_ERROR_THRESHOLD
            and pos_error_right < POS_ERROR_THRESHOLD
            and left_angle_error < POSE_ERROR_THRESHOLD
            and right_angle_error < POSE_ERROR_THRESHOLD
            and solver_ok
        )
        # For teleoperation, do not force the caller to drop a bounded partial
        # command. The controller file still prints a throttled warning when this
        # flag is False, but it can safely publish q_arm.
        business_converged = bool(exact_converged or (self.teleop_accept_partial_solution and np.all(np.isfinite(q_arm))))
        return q_arm, tau_ff, business_converged

    def get_current_end_effector_poses(self):
        # Get current end-effector poses
        left_pose = self.compute_forward_kinematics(self.current_q, 'left')
        right_pose = self.compute_forward_kinematics(self.current_q, 'right')
        return left_pose, right_pose


def main():
    print("=== T1 Robot Dual Arm IK Test ===")
    # Create IK solver
    print("\n>>> Initializing IK solver...")
    ik_solver = T17DofArmIK(visualization=False, unit_test=False)
    # Get current end-effector poses
    left_pose, right_pose = ik_solver.get_current_end_effector_poses()
    print("\n>>> Initial pose information:")
    left_pos, left_rpy = ik_solver.print_pose(left_pose, "Initial left hand pose")
    right_pos, right_rpy = ik_solver.print_pose(right_pose, "Initial right hand pose")
    left_pos_current, left_rpy_current = ik_solver.pose_to_xyzrpy(left_pose)
    right_pos_current, right_rpy_current = ik_solver.pose_to_xyzrpy(right_pose)
    # Set new pose
    left_pos_target = left_pos_current + np.array([0.0, 0.0, 0.05])
    left_rpy_target = left_rpy_current + np.array([np.radians(10), np.radians(0), np.radians(0)])
    right_pos_target = right_pos_current + np.array([0.0, 0.0, 0.05])
    right_rpy_target = right_rpy_current + np.array([np.radians(-10), np.radians(0), np.radians(0)])
    # Create target pose matrix
    target_left = ik_solver.xyzrpy_to_pose(left_pos_target, left_rpy_target)
    target_right = ik_solver.xyzrpy_to_pose(right_pos_target, right_rpy_target)
    # Solve IK
    print("\n>>> Solving IK...")
    q0 = ik_solver.get_initial_joint_positions()
    q_arm, tau_ff, converged = ik_solver.solve_ik(target_left, target_right, q0, visualize=True)
    print(f"\n>>> Joint position changes:")
    print(f"Left arm (7 joints): {q_arm[:7]}")
    print(f"Right arm (7 joints): {q_arm[7:]}")
    print(f">>> Converged: {converged}")
    print("\n>>> Test completed!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n>>> Program error: {e}")
        import traceback
        traceback.print_exc()
