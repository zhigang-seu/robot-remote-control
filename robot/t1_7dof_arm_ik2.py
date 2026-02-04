# -*- coding: utf-8 -*-

import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca
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
        self.urdf_path = '/home/master/Workspace/test_lxk/booster_robotics_sdk-main/example/low_level/assets/T1_7DofArm_Serial.urdf'
        self.mesh_dir = '/home/master/Workspace/test_lxk/booster_robotics_sdk-main/example/low_level/assets/'
        # Cache file path
        self.cache_path = "t1_robot_cache.pkl"
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
        self.arm_joint_ids = []
        print("\n=== Arm Joint Information ===")
        for joint_name in self.arm_joint_names:
            try:
                jid = self.robot.model.getJointId(joint_name)
                if jid < len(self.robot.model.joints):
                    idx_q = self.robot.model.joints[jid].idx_q
                    self.arm_joint_indices.append(idx_q)
                    self.arm_joint_ids.append(jid)
                    print(f"Joint: {joint_name}, ID: {jid}, Index: {idx_q}")
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
        # CasADi
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
                    self.cdata.oMf[self.left_hand_id].translation - self.cTf_l[:3,3],
                    self.cdata.oMf[self.right_hand_id].translation - self.cTf_r[:3,3]
                )
            ],
        )
        self.rotational_error = ca.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                ca.vertcat(
                    cpin.log3(self.cdata.oMf[self.left_hand_id].rotation @ self.cTf_l[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.right_hand_id].rotation @ self.cTf_r[:3,:3].T)
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
            50 * self.translational_cost +  
            1 * self.rotation_cost +    
            0.02 * self.regularization_cost + 
            0.1 * self.smooth_cost       
        )
        opts = {
            'expand': True,
            'detect_simple_bounds': True,
            'calc_lam_p': False,
            'print_time': False,
            'ipopt.sb': 'yes',
            'ipopt.print_level': 0,
            'ipopt.max_iter': 30,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 5e-4,
            'ipopt.acceptable_iter': 5,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.derivative_test': 'none',
            'ipopt.jacobian_approximation': 'exact',
        }
        self.opti.solver("ipopt", opts)
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
    
    def solve_ik_optimization(self, left_target, right_target, current_q=None):
        if current_q is not None:
            self.init_data = current_q.copy()
        else:
            self.init_data = self.current_q.copy()
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_tf_l, left_target)
        self.opti.set_value(self.param_tf_r, right_target)
        self.opti.set_value(self.var_q_last, self.init_data) 
        converged = False
        sol_q = None
        
        try:
            sol = self.opti.solve()
            sol_q = self.opti.value(self.var_q)
            converged = True
            self.smooth_filter.add_data(self.extract_arm_joints(sol_q))
            arm_joints_smoothed = self.smooth_filter.filtered_data
            sol_q = self.set_arm_joints(sol_q, arm_joints_smoothed)
            self.init_data = sol_q
            self.current_q = sol_q.copy()
            
        except Exception as e:
            print(f">>> IK optimization failed: {e}")
            sol_q = self.opti.debug.value(self.var_q)
            converged = False
            self.smooth_filter.add_data(self.extract_arm_joints(sol_q))
            arm_joints_smoothed = self.smooth_filter.filtered_data
            sol_q = self.set_arm_joints(sol_q, arm_joints_smoothed)
        tau_ff_arm = np.zeros(len(self.arm_joint_indices))
        q_arm = self.extract_arm_joints(sol_q if sol_q is not None else self.init_data)
        
        return q_arm, tau_ff_arm, converged
    
    # Solve inverse kinematics
    def solve_ik(self, left_target_pose, right_target_pose, current_q=None, visualize=False):
        if current_q is not None:
            self.current_q = current_q
        print("\n>>> Target pose information:")
        left_pos, left_rpy = self.print_pose(left_target_pose, "Left hand target pose")
        right_pos, right_rpy = self.print_pose(right_target_pose, "Right hand target pose")
        q_arm, tau_ff, converged = self.solve_ik_optimization(
            left_target_pose, right_target_pose, self.current_q
        )
        q_full = self.set_arm_joints(self.current_q, q_arm)
        T_left_sol = self.compute_forward_kinematics(q_full, 'left')
        T_right_sol = self.compute_forward_kinematics(q_full, 'right')
        print("\n>>> Pose information after solving:")
        left_sol_pos, left_sol_rpy = self.print_pose(T_left_sol, "Left hand actual pose")
        right_sol_pos, right_sol_rpy = self.print_pose(T_right_sol, "Right hand actual pose")
        pos_error_left = np.linalg.norm(T_left_sol[:3, 3] - left_target_pose[:3, 3])
        pos_error_right = np.linalg.norm(T_right_sol[:3, 3] - right_target_pose[:3, 3])
        R_left_error = np.dot(T_left_sol[:3, :3].T, left_target_pose[:3, :3])
        R_right_error = np.dot(T_right_sol[:3, :3].T, right_target_pose[:3, :3])
        left_angle_error = np.arccos(np.clip((np.trace(R_left_error) - 1) / 2, -1.0, 1.0))
        right_angle_error = np.arccos(np.clip((np.trace(R_right_error) - 1) / 2, -1.0, 1.0))
        print(f"\n>>> IK solution completed")
        print(f">>> Convergence status: {'SUCCESS' if converged else 'FAILED'}")
        print(f"Left hand position error: {pos_error_left:.6f} m")
        print(f"Left hand pose error: {left_angle_error:.6f} rad ({np.degrees(left_angle_error):.2f} deg)")
        print(f"Right hand position error: {pos_error_right:.6f} m")
        print(f"Right hand pose error: {right_angle_error:.6f} rad ({np.degrees(right_angle_error):.2f} deg)")
        # Visualization
        if visualize and self.vis is not None:
            self.vis.display(q_full)
        POS_ERROR_THRESHOLD = 0.02
        POSE_ERROR_THRESHOLD = np.radians(10)
        business_converged = (pos_error_left < POS_ERROR_THRESHOLD and 
                              pos_error_right < POS_ERROR_THRESHOLD and
                              left_angle_error < POSE_ERROR_THRESHOLD and
                              right_angle_error < POSE_ERROR_THRESHOLD and
                              converged) 
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