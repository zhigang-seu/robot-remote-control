import os
import numpy as np
import pinocchio as pin
import threading
import time
from enum import IntEnum
import logging
import sys
import socket
import json
import math
from queue import Queue
from collections import defaultdict
# Import IK solver class
from t1_7dof_arm_ik2 import T17DofArmIK
# SDK imports
from booster_robotics_sdk_python import (
    ChannelFactory, B1LowCmdPublisher, LowCmd, LowCmdType, MotorCmd, 
    B1LowStateSubscriber, LowState
)

logger = logging.getLogger(__name__)
# Control period
CONTROL_DT = 0.002
# Number of joints
B1_JOINT_CNT = 29
# VR glasses IP and port
VR_IP = "192.168.0.8"
VR_PORT = 8000
# Controller parameters
SCALING = 2.0 * 100.0

# Joint index enumeration class
class B1JointIndex(IntEnum):
    # Head
    HEAD_YAW = 0
    HEAD_PITCH = 1

    LEFT_SHOULDER_PITCH = 2
    LEFT_SHOULDER_ROLL = 3
    LEFT_ELBOW_PITCH = 4
    LEFT_ELBOW_YAW = 5
    LEFT_WRIST_PITCH = 6
    LEFT_WRIST_YAW = 7
    LEFT_HAND_ROLL = 8

    RIGHT_SHOULDER_PITCH = 9
    RIGHT_SHOULDER_ROLL = 10
    RIGHT_ELBOW_PITCH = 11
    RIGHT_ELBOW_YAW = 12
    RIGHT_WRIST_PITCH = 13
    RIGHT_WRIST_YAW = 14
    RIGHT_HAND_ROLL = 15

    WAIST = 16

    LEFT_HIP_PITCH = 17
    LEFT_HIP_ROLL = 18
    LEFT_HIP_YAW = 19
    LEFT_KNEE_PITCH = 20
    CRANK_UP_LEFT = 21
    CRANK_DOWN_LEFT = 22

    RIGHT_HIP_PITCH = 23
    RIGHT_HIP_ROLL = 24
    RIGHT_HIP_YAW = 25
    RIGHT_KNEE_PITCH = 26
    CRANK_UP_RIGHT = 27
    CRANK_DOWN_RIGHT = 28

# Motor state class
class MotorState:
    def __init__(self):
        self.q = 0.0  
        self.dq = 0.0  
        
# Robot state class
class B1LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(B1_JOINT_CNT)]

# Thread-safe data buffer
class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()
        
    def get_data(self):
        with self.lock:
            return self.data
            
    def set_data(self, data):
        with self.lock:
            self.data = data

# VR controller data processor
class VRController:
    def __init__(self, robot_controller, scaling=SCALING):
        self.robot_controller = robot_controller
        self.scaling = scaling
        self.running = False
        self.sock = None
        self.left_initial_pos = [0.25, 0.2, 0.0] 
        self.right_initial_pos = [0.25, -0.2, 0.0]
        self.current_state = {
            "left": {
                "position": self.left_initial_pos,
                "rpy": [-1.57, -1.57, 0.0]
            },
            "right": {
                "position": self.right_initial_pos,
                "rpy": [1.57, -1.57, 0.0]
            }
        }
        self.msg_queue = Queue()
        self.buffer = ""
        self.state_lock = threading.Lock()
        
    def connect_vr(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            server_address = (VR_IP, VR_PORT)
            self.sock.connect(server_address)
            print(f"Connected to VR glasses {VR_IP}:{VR_PORT}")
            message = "This is robot!"
            self.sock.sendall(message.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Failed to connect to VR glasses: {e}")
            return False
            
    def receive_data(self):
        try:
            while self.running:
                data = self.sock.recv(1024)
                if not data:
                    print("VR glasses disconnected")
                    break
                self.buffer += data.decode('utf-8', errors='ignore')
                self.parse_buffer_data()
        except Exception as e:
            if self.running: 
                print(f"Error receiving data: {e}")

    def parse_buffer_data(self):
        while True:
            start = self.buffer.find("{")
            if start == -1:
                self.buffer = ""
                break
            bracket_count = 0
            end = -1
            for i in range(start, len(self.buffer)):
                if self.buffer[i] == "{":
                    bracket_count += 1
                elif self.buffer[i] == "}":
                    bracket_count -= 1
                if bracket_count == 0:
                    end = i
                    break
            if end == -1:
                self.buffer = self.buffer[start:]
                break
            json_str = self.buffer[start:end+1]
            self.buffer = self.buffer[end+1:]
            try:
                json_data = json.loads(json_str)
                who = json_data.get("who", "")
                if who in ["left", "right"]:
                    self.msg_queue.put(json_str)
            except json.JSONDecodeError:
                print(f"Invalid JSON data filtered: {json_str[:50]}...")

    def update_hand_state(self, json_data, hand_side):
        x = json_data.get("x", 0.0)
        y = json_data.get("y", 0.0)
        z = json_data.get("z", 0.0)
        roll_deg = json_data.get("roll", 0.0)
        pitch_deg = json_data.get("pitch", 0.0)
        yaw_deg = json_data.get("yaw", 0.0)

        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        rot_matrix_pin = pin.rpy.rpyToMatrix(roll, pitch, yaw)

        rot_y_neg90 = pin.rpy.rpyToMatrix(0.0, -math.pi / 2.0, 0.0)

        if hand_side == "right":
            rot_x = pin.rpy.rpyToMatrix(math.pi / 2.0, 0.0, 0.0)
            rot_compensated = rot_matrix_pin @ rot_y_neg90 @ rot_x
            rpy_normalized = pin.rpy.matrixToRpy(rot_compensated)
            pos = [
                self.right_initial_pos[0] + x / self.scaling,
                self.right_initial_pos[1] + y / self.scaling,
                self.right_initial_pos[2] + z / self.scaling,
            ]
            rpy = [float(rpy_normalized[0]), float(rpy_normalized[1]), float(rpy_normalized[2])]
        else:
            rot_x = pin.rpy.rpyToMatrix(-math.pi / 2.0, 0.0, 0.0)
            rot_compensated = rot_matrix_pin @ rot_y_neg90 @ rot_x
            rpy_normalized = pin.rpy.matrixToRpy(rot_compensated)
            pos = [
                self.left_initial_pos[0] + x / self.scaling,
                self.left_initial_pos[1] + y / self.scaling,
                self.left_initial_pos[2] + z / self.scaling,
            ]
            rpy = [float(rpy_normalized[0]), float(rpy_normalized[1]), float(rpy_normalized[2])]

        with self.state_lock:
            self.current_state[hand_side]["position"] = pos
            self.current_state[hand_side]["rpy"] = rpy

    def get_current_vr_targets(self):
        with self.state_lock:
            left_pos = list(self.current_state.get("left", {}).get("position", self.left_initial_pos))
            left_rpy = list(self.current_state.get("left", {}).get("rpy", [-1.57, -1.57, 0.0]))
            right_pos = list(self.current_state.get("right", {}).get("position", self.right_initial_pos))
            right_rpy = list(self.current_state.get("right", {}).get("rpy", [1.57, -1.57, 0.0]))
        return left_pos, left_rpy, right_pos, right_rpy

    def solve_from_current_state(self):
        left_pos, left_rpy, right_pos, right_rpy = self.get_current_vr_targets()
        return self.calculate_target_arm_joints(left_pos, left_rpy, right_pos, right_rpy)
        
    def process_hand_data(self, json_data, hand_side):
        try:
            self.update_hand_state(json_data, hand_side)
            return self.solve_from_current_state()
        except Exception as e:
            print(f"Error processing controller data: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def calculate_target_arm_joints(self, left_pos, left_rpy, right_pos, right_rpy):
        if not self.robot_controller.use_ik:
            print("IK solver not enabled")
            return None
        try:
            q_arm_14, tau_ff, converged = self.robot_controller.ik_solver.solve_ik(
                self.robot_controller.ik_solver.xyzrpy_to_pose(left_pos, left_rpy),
                self.robot_controller.ik_solver.xyzrpy_to_pose(right_pos, right_rpy),
                current_q=self.robot_controller.get_current_joint_angles(),
                visualize=False
            )
            if not converged:
                print(f"IK solver did not converge! Left pos: {left_pos}, Right pos: {right_pos}")
                return None
            return q_arm_14
        except Exception as e:
            print(f"Error calculating target arm joint angles: {e}")
            return None
    
    def process_messages(self):
        target_arm_joints = None
        if not self.msg_queue.empty():
            try:
                json_str = self.msg_queue.get()
                if not json_str:
                    return None
                json_data = json.loads(json_str)
                who = json_data.get("who", "")
                if who == "right" or who == "left":
                    target_arm_joints = self.process_hand_data(json_data, who)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
            except Exception as e:
                print(f"Error processing message: {e}")
        return target_arm_joints
    
    def start(self):
        if not self.connect_vr():
            return False
        self.running = True
        recv_thread = threading.Thread(target=self.receive_data)
        recv_thread.daemon = True
        recv_thread.start()
        return True
    
    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()

# Robot controller class
class B1RobotController: 
    def __init__(self, simulation_mode=False, control_mode=LowCmdType.PARALLEL, 
                 use_ik=True, visualize_ik=False):
        logger.info("Initializing B1RobotController...")
        self._initialized = False
        self.ctrl_lock = threading.Lock()
        self.simulation_mode = simulation_mode
        self.control_mode = control_mode
        self.control_dt = CONTROL_DT
        self.q_target = np.zeros(B1_JOINT_CNT)
        self.HOME0 = np.array([
            0.00, 0.80,                     
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
            0.0,                        
            -0.1, 0.0, 0.0, 0.2, 0.104, 0.098, 
            -0.1, 0.0, 0.0, 0.2, 0.104, 0.098 
        ])
        self.FIXED_HOME_POSITION = np.array([
            0.00, 0.80,                     
            0.1207, -1.3649,  -0.0025, -1.5215, -0.2081, -0.1417, 0.0086, 
            0.1207, 1.3649,  -0.0025, 1.5215, -0.2081, 0.1417, 0.0086,  
            0.0,                        
            -0.1, 0.0, 0.0, 0.2, 0.104, 0.098, 
            -0.1, 0.0, 0.0, 0.2, 0.104, 0.098 
        ])
#        self.READY_POSITION = np.array([
#            0.00, 0.00,                     
#            0.45, -1.05,  0.00, -1.50, 0.0, 0.0, 0.0, 
#            0.45,  1.05,  0.00,  1.50, 0.0, 0.0, 0.0,
#            0.0,                        
#            -0.1, 0.0, 0.0, 0.2, 0.104, 0.098, 
#            -0.1, 0.0, 0.0, 0.2, 0.104, 0.098 
#        ])
        self.home_positions = self.FIXED_HOME_POSITION.copy()
        # Joint velocity limit (rad/s)
        self.joint_velocity_limit = 0.4
        self.position_tolerance = 0.01
        # PID parameters
        self.kp_data = [
            5.0, 5.0,        
            100., 100., 100., 100.,80., 100., 80., 
            100., 100., 100., 100.,80., 100., 80., 
            100.,                   
            350., 350., 180., 350., 400., 400., 
            350., 350., 180., 350., 400., 400.   
        ]
        self.kd_data = [
            0.1, 0.1,               
            1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 
            1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7,  
            5.0,                      
            2, 2, 2., 2, 1, 1,         
            2, 2, 2., 2, 1, 1,         
        ]
        # Current desired position
        self.current_jpos_des = np.zeros(B1_JOINT_CNT)
        self.weight = 0.0
        self.weight_rate = 0.2
        self.weight_margin = self.weight_rate * self.control_dt
        ChannelFactory.Instance().Init(0 if not simulation_mode else 1)
        self.low_state_subscriber = B1LowStateSubscriber(handler=self._on_state_received)
        self.low_state_subscriber.InitChannel()
        self.low_cmd_publisher = B1LowCmdPublisher()
        self.low_cmd_publisher.InitChannel()
        self.lowstate_buffer = DataBuffer()
        self.motor_cmds = [MotorCmd() for _ in range(B1_JOINT_CNT)]
        self.ctrl_lock = threading.Lock()
        self.low_cmd = LowCmd()
        self.low_cmd.cmd_type = self.control_mode
        self.low_cmd.motor_cmd = self.motor_cmds
        self._subscribe_running = True
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()
        while not self.lowstate_buffer.get_data():
            time.sleep(0.1)
            logger.warning("[B1RobotController] Waiting for robot state subscription...")
        logger.info("[B1RobotController] Robot state subscription successful")
        self.current_jpos_des = self.get_current_joint_angles()
        self._initialize_joint_control()
        self.use_ik = use_ik
        if use_ik:
            try:
                self.ik_solver = T17DofArmIK(visualization=visualize_ik, unit_test=False)
                logger.info("IK solver initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize IK solver: {e}")
                self.use_ik = False
                self.ik_solver = None
        else:
            self.ik_solver = None
        self.vr_controller = VRController(self)
        logger.info("B1RobotController initialization completed!")
        with self.ctrl_lock:
            self.q_target = self.current_jpos_des.copy() 
        self._publish_running = True
        self.publish_thread = threading.Thread(target=self._publish_control_commands)
        self.publish_thread.daemon = True
        self.publish_thread.start()
        self._initialized = True

    def _on_state_received(self, state: LowState):
        lowstate = B1LowState()
        for i in range(B1_JOINT_CNT):
            lowstate.motor_state[i].q = state.motor_state_parallel[i].q
            lowstate.motor_state[i].dq = state.motor_state_parallel[i].dq
        self.lowstate_buffer.set_data(lowstate)
        
    def _subscribe_motor_state(self):
        while self._subscribe_running:
            time.sleep(0.001)

    # Initialize joint control parameters
    def _initialize_joint_control(self):
        logger.info("Initializing joint control parameters...")
        current_q = self.get_current_joint_angles()
        for idx in range(B1_JOINT_CNT):
            self.motor_cmds[idx].q = current_q[idx]
            self.motor_cmds[idx].dq = 0.0
            self.motor_cmds[idx].kp = self.kp_data[idx]
            self.motor_cmds[idx].kd = self.kd_data[idx]
            self.motor_cmds[idx].tau = 0.0
            self.motor_cmds[idx].weight = self.weight
        logger.info("Joint control parameters initialization completed")

    # Limit joint angle rate of change
    def _clip_joint_target(self, target_q, velocity_limit=None):
        if velocity_limit is None:
            velocity_limit = self.joint_velocity_limit
        current_q = self.current_jpos_des
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        clipped_target = current_q + delta / max(motion_scale, 1.0)
        return clipped_target
        
#    def _clip_joint_target(self, target_q, velocity_limit=None):
#        if velocity_limit is None:
#            velocity_limit = self.joint_velocity_limit
#        current_q = self.current_jpos_des
#        max_joint_delta = velocity_limit * self.control_dt
#        clipped_target = np.zeros_like(target_q)
#        for i in range(len(target_q)):
#            error = target_q[i] - current_q[i]
#            delta = max(min(error, max_joint_delta), -max_joint_delta)
#            clipped_target[i] = current_q[i] + delta
#        return clipped_target
    
    def _publish_control_commands(self):
        while self._publish_running:
            if not self._initialized:
                time.sleep(self.control_dt)
                continue
            start_time = time.time()
            with self.ctrl_lock:
                target_q = self.q_target.copy()
                current_target = self.current_jpos_des.copy()
            
            if self.simulation_mode:
                self.current_jpos_des = target_q
            else:
                self.current_jpos_des = self._clip_joint_target(target_q)
            for idx in range(B1_JOINT_CNT):
                self.motor_cmds[idx].q = self.current_jpos_des[idx]
                self.motor_cmds[idx].dq = 0.0     
                self.motor_cmds[idx].kp = self.kp_data[idx]
                self.motor_cmds[idx].kd = self.kd_data[idx]
                self.motor_cmds[idx].tau = 0.0  
                self.motor_cmds[idx].weight = self.weight
            #print(self.current_jpos_des)
            self.low_cmd = LowCmd()  
            self.low_cmd.cmd_type = self.control_mode  
            self.low_cmd.motor_cmd = self.motor_cmds  
            self.low_cmd_publisher.Write(self.low_cmd)
            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.control_dt - elapsed_time)
            time.sleep(sleep_time)
    
    # Control all joint angles
    def ctrl_all_joints(self, q_target):
        if len(q_target) != B1_JOINT_CNT:
            logger.error(f"Incorrect number of target joint angles: expected {B1_JOINT_CNT}, got {len(q_target)}")
            return
        with self.ctrl_lock:
            self.q_target = np.array(q_target)
    
    # Control dual arm joint angles
    def ctrl_dual_arm_14dof(self, arm_target_14):
        if len(arm_target_14) != 14:
            logger.error(f"Incorrect number of arm joint angles: expected 14, got {len(arm_target_14)}")
            return
        with self.ctrl_lock:
            full_target = self.q_target.copy()
            for i in range(7):
                full_target[B1JointIndex.LEFT_SHOULDER_PITCH + i] = arm_target_14[i]
            for i in range(7):
                full_target[B1JointIndex.RIGHT_SHOULDER_PITCH + i] = arm_target_14[7 + i]
            for idx in range(B1_JOINT_CNT):
                if idx < B1JointIndex.LEFT_SHOULDER_PITCH or idx > B1JointIndex.RIGHT_HAND_ROLL:
                    full_target[idx] = self.FIXED_HOME_POSITION[idx]
            self.q_target = full_target
    
    # Get current all joint angles
    def get_current_joint_angles(self):
        lowstate = self.lowstate_buffer.get_data()
        if lowstate:
            return np.array([lowstate.motor_state[i].q for i in range(B1_JOINT_CNT)])
        return np.zeros(B1_JOINT_CNT)
    
    # Get current 14 arm joint angles
    def get_current_arm_joint_angles(self):
        full_q = self.get_current_joint_angles()
        arm_q = np.zeros(14)
        for i in range(7):
            arm_q[i] = full_q[B1JointIndex.LEFT_SHOULDER_PITCH + i]
        for i in range(7):
            arm_q[7 + i] = full_q[B1JointIndex.RIGHT_SHOULDER_PITCH + i]
        return arm_q

    def _move_to_pose_blocking(self, target_positions, duration=10.0, pose_name="target pose"):
        logger.info(f"Robot moving to {pose_name}, duration: {duration} seconds...")
        self.ctrl_all_joints(np.array(target_positions).copy())
        time.sleep(duration)
        print(f"Reached {pose_name}")

    # Return to home0 position
    def go_home0(self, duration=10.0):
        self._move_to_pose_blocking(
            self.HOME0,
            duration=duration,
            pose_name="home0 position"
        )

    # Return to fixed home position
    def go_home(self, duration=10.0):
        self._move_to_pose_blocking(
            self.FIXED_HOME_POSITION,
            duration=duration,
            pose_name="fixed home position"
        )
    
    def start_control(self):
        logger.info("Starting control...")
        self.weight = 0.0
        while self.weight < 1.0:
            self.weight += self.weight_margin
            self.weight = min(self.weight, 1.0)
            time.sleep(self.control_dt)
    
    def stop_control(self):
        logger.info("Stopping control...")
        while self.weight > 0.0:
            self.weight -= self.weight_margin
            self.weight = max(self.weight, 0.0)
            time.sleep(self.control_dt)
    
    def start_vr_control(self):
        if not self.use_ik:
            print("IK solver not enabled, cannot perform VR control")
            return False
        print("Starting VR control...")
        current_q = self.FIXED_HOME_POSITION.copy()
        with self.ctrl_lock:
            self.current_jpos_des = current_q.copy()
            self.q_target = current_q.copy()
        return self.vr_controller.start()
    
    def process_vr_data_and_enqueue(self):
        try:
            # Get target arm joint angles
            target_arm_joints = self.vr_controller.process_messages()
            if target_arm_joints is not None:
                self.ctrl_dual_arm_14dof(target_arm_joints)
                return True
        except Exception as e:
            print(f"Error processing VR data: {e}")
            
        return False
    
    def cleanup(self):
        logger.info("Cleaning up robot controller resources...")
        self._publish_running = False
        self._subscribe_running = False
        self.vr_controller.stop()
        time.sleep(0.1) 
        logger.info("Robot controller resource cleanup completed")
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== B1 Robot Controller Demo ===")
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} networkInterface")
        print("Example: python controller.py enp6s0")
        sys.exit(-1)
    network_interface = sys.argv[1]
    
    try:
        # Create controller instance
        controller = B1RobotController(
            simulation_mode=False,
            use_ik=True,
            visualize_ik=False 
        )
        time.sleep(5)
        # Start control
        controller.start_control()
        controller.go_home(duration=5.0)
        # Wait for user to press Enter to continue
        print("\nPress Enter to continue to VR control...")
        input()
        # Start VR control
        print("\nStarting VR control...")
        if not controller.start_vr_control():
            print("VR control startup failed")
            controller.stop_control()
            controller.cleanup()
            sys.exit(-1)
        print("VR control started, beginning to receive controller data...")
        print("\nEntering main control loop...")
        try:
            target_count = 0
            last_print_time = time.time()
            while True:
                start_time = time.time()
                if controller.process_vr_data_and_enqueue():
                    target_count += 1
                current_time = time.time()
                if current_time - last_print_time > 2.0:
                    print(f"Real-time Targets sent: {target_count}")
                    last_print_time = current_time
                elapsed_time = time.time() - start_time
                sleep_time = max(0, 0.015 - elapsed_time)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nUser interrupted program")
    except Exception as e:
        print(f"\nProgram error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        print("\nCleaning up resources...")
        controller.stop_control()
        controller.cleanup()
        print("\nProgram ended")
