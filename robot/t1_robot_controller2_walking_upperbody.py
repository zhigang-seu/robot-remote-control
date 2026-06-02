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
# Import IK solver class
from t1_7dof_arm_ik2 import T17DofArmIK
# SDK imports
from booster_robotics_sdk_python import (
    ChannelFactory, B1LowCmdPublisher, LowCmd, LowCmdType, MotorCmd,
    B1LowStateSubscriber, LowState, B1LocoClient
)
try:
    from booster_robotics_sdk_python import GetModeResponse, B1LocoApiId
except ImportError:
    GetModeResponse = None
    B1LocoApiId = None

logger = logging.getLogger(__name__)
# Control period
CONTROL_DT = 0.002
# Number of joints
B1_JOINT_CNT = 29

# In walking mode we must not publish leg/locomotion joints.
# UpperBodyCustomControl only consumes upper-body joints on rt/joint_ctrl.
ARM_JOINT_INDICES = list(range(2, 16))
UPPER_BODY_JOINT_INDICES = list(range(0, 17))  # head(0-1), arms(2-15), waist(16)
LOWER_BODY_JOINT_INDICES = [i for i in range(B1_JOINT_CNT) if i not in UPPER_BODY_JOINT_INDICES]

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
        
    def _convert_vr_pose_to_robot_pose(self, json_data, hand_side):
        """Convert one VR hand JSON packet to robot end-effector position/RPY.

        The VR packet uses millimeter-like x/y/z deltas and degree roll/pitch/yaw.
        The resulting position is in the IK base frame and RPY is in radians.
        """
        x = float(json_data.get("x", 0.0))
        y = float(json_data.get("y", 0.0))
        z = float(json_data.get("z", 0.0))
        roll = math.radians(float(json_data.get("roll", 0.0)))
        pitch = math.radians(float(json_data.get("pitch", 0.0)))
        yaw = math.radians(float(json_data.get("yaw", 0.0)))

        rot_matrix_pin = pin.rpy.rpyToMatrix(roll, pitch, yaw)

        if hand_side == "right":
            rot_y_neg90 = pin.rpy.rpyToMatrix(0.0, -math.pi / 2.0, 0.0)
            rot_x_pos90 = pin.rpy.rpyToMatrix(math.pi / 2.0, 0.0, 0.0)
            rot_compensated = rot_matrix_pin @ rot_y_neg90 @ rot_x_pos90
            rpy = pin.rpy.matrixToRpy(rot_compensated)
            pos = [
                self.right_initial_pos[0] + x / self.scaling,
                self.right_initial_pos[1] + y / self.scaling,
                self.right_initial_pos[2] + z / self.scaling,
            ]
        else:
            rot_y_neg90 = pin.rpy.rpyToMatrix(0.0, -math.pi / 2.0, 0.0)
            rot_x_neg90 = pin.rpy.rpyToMatrix(-math.pi / 2.0, 0.0, 0.0)
            rot_compensated = rot_matrix_pin @ rot_y_neg90 @ rot_x_neg90
            rpy = pin.rpy.matrixToRpy(rot_compensated)
            pos = [
                self.left_initial_pos[0] + x / self.scaling,
                self.left_initial_pos[1] + y / self.scaling,
                self.left_initial_pos[2] + z / self.scaling,
            ]

        return [float(v) for v in pos], [float(rpy[0]), float(rpy[1]), float(rpy[2])]

    def update_hand_state(self, json_data, hand_side):
        """Update cached VR target for one hand. Used by the ROS vr_router path."""
        if hand_side not in ("left", "right"):
            return
        try:
            pos, rpy = self._convert_vr_pose_to_robot_pose(json_data, hand_side)
            with self.state_lock:
                self.current_state[hand_side]["position"] = pos
                self.current_state[hand_side]["rpy"] = rpy
        except Exception as e:
            print(f"Error updating {hand_side} hand target: {e}")
            import traceback
            traceback.print_exc()

    def solve_from_current_state(self):
        """Solve 14-DoF arm IK from the cached left/right VR targets."""
        with self.state_lock:
            left_pos = list(self.current_state.get("left", {}).get("position", self.left_initial_pos))
            left_rpy = list(self.current_state.get("left", {}).get("rpy", [-1.57, -1.57, 0.0]))
            right_pos = list(self.current_state.get("right", {}).get("position", self.right_initial_pos))
            right_rpy = list(self.current_state.get("right", {}).get("rpy", [1.57, -1.57, 0.0]))
        return self.calculate_target_arm_joints(left_pos, left_rpy, right_pos, right_rpy)

    def process_hand_data(self, json_data, hand_side):
        """Socket/standalone path: update one hand then solve from both cached targets."""
        self.update_hand_state(json_data, hand_side)
        return self.solve_from_current_state()
        
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
    def _init_channel_factory(self):
        domain = 0 if not self.simulation_mode else 1
        factory = ChannelFactory.Instance()
        try:
            if self.network_interface:
                factory.Init(domain, self.network_interface)
            else:
                factory.Init(domain)
        except TypeError:
            # Compatibility with older SDK bindings that only accept domain_id.
            factory.Init(domain)

    def __init__(self, network_interface="", simulation_mode=False, control_mode=LowCmdType.PARALLEL, 
                 use_ik=True, visualize_ik=False, upper_body_only=True):
        logger.info("Initializing B1RobotController...")
        self._initialized = False
        self.ctrl_lock = threading.Lock()
        self.simulation_mode = simulation_mode
        self.control_mode = control_mode
        self.control_dt = CONTROL_DT
        self.network_interface = network_interface
        self.upper_body_only = upper_body_only
        self.upper_body_custom_enabled = False
        self._publish_enabled = False
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
        self._init_channel_factory()

        # High-level client is only used to open/close UpperBodyCustomControl.
        # The actual joint stream is still LowCmd on rt/joint_ctrl.
        self.loco_client = B1LocoClient()
        self.loco_client.Init()

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

    def _call_upper_body_custom_control(self, start: bool) -> bool:
        """Open or close SDK UpperBodyCustomControl.

        Firmware >= v1.4.0.7 provides:
            int32_t UpperBodyCustomControl(bool start)

        When enabled in walking mode, the robot accepts upper-body joint
        commands from rt/joint_ctrl while locomotion keeps controlling legs.
        """
        if self.simulation_mode:
            self.upper_body_custom_enabled = bool(start)
            return True

        try:
            if hasattr(self.loco_client, "UpperBodyCustomControl"):
                res = self.loco_client.UpperBodyCustomControl(bool(start))
            else:
                # Some intermediate Python bindings exposed the generic RPC sender
                # before adding the convenience method. Try the raw API ID fallback.
                api_id = getattr(B1LocoApiId, "kUpperBodyCustomControl", None) if B1LocoApiId is not None else None
                if api_id is None or not hasattr(self.loco_client, "SendApiRequest"):
                    print("[Error] UpperBodyCustomControl API is not available in this Python SDK.")
                    return False
                res = self.loco_client.SendApiRequest(api_id, json.dumps({"start": bool(start)}))
        except Exception as e:
            print(f"[Error] UpperBodyCustomControl({start}) raised exception: {e}")
            return False

        # Some Python bindings send the request successfully but return None
        # instead of the documented integer result. Treat None as non-fatal.
        if res is None:
            print(f"[Warn] UpperBodyCustomControl({start}) returned None; assuming request was sent.")
        elif res != 0:
            print(f"[Error] UpperBodyCustomControl({start}) failed, error={res}")
            return False
        else:
            print(f"[OK] UpperBodyCustomControl({start})")

        self.upper_body_custom_enabled = bool(start)
        return True

    def _try_print_robot_mode(self):
        """Best-effort mode print. Failure here should not stop teleop."""
        try:
            if GetModeResponse is None:
                return
            gm = GetModeResponse()
            res = self.loco_client.GetMode(gm)
            if res == 0:
                print(f"[Info] current robot mode = {gm.mode}")
        except Exception as e:
            print(f"[Warn] GetMode skipped: {e}")

    def _make_upper_body_target(self, arm_target_14=None, use_fixed_home_arms=False):

        full_target = self.get_current_joint_angles().copy()
        if len(full_target) != B1_JOINT_CNT:
            full_target = self.q_target.copy()

        # Hold waist and legs where they currently are
        for idx in LOWER_BODY_JOINT_INDICES:
            full_target[idx] = self.get_current_joint_angles()[idx]

        full_target[B1JointIndex.HEAD_YAW] = 0.00
        full_target[B1JointIndex.HEAD_PITCH] = 0.80

        if use_fixed_home_arms:
            for idx in ARM_JOINT_INDICES:
                full_target[idx] = self.FIXED_HOME_POSITION[idx]

        if arm_target_14 is not None:
            if len(arm_target_14) != 14:
                raise ValueError(f"expected 14 arm joints, got {len(arm_target_14)}")
            for i in range(7):
                full_target[B1JointIndex.LEFT_SHOULDER_PITCH + i] = arm_target_14[i]
            for i in range(7):
                full_target[B1JointIndex.RIGHT_SHOULDER_PITCH + i] = arm_target_14[7 + i]

        return full_target

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
            if not self._initialized or not self._publish_enabled:
                time.sleep(self.control_dt)
                continue

            start_time = time.time()
            with self.ctrl_lock:
                target_q = self.q_target.copy()

            if self.upper_body_only:
                # Keep lower-body desired values synced to the real robot state,
                # then publish them with zero weight. This prevents accidental
                # leg takeover if the firmware reads a full LowCmd vector.
                current_q = self.get_current_joint_angles()
                for idx in LOWER_BODY_JOINT_INDICES:
                    target_q[idx] = current_q[idx]
                    self.current_jpos_des[idx] = current_q[idx]

            if self.simulation_mode:
                self.current_jpos_des = target_q
            else:
                self.current_jpos_des = self._clip_joint_target(target_q)

            for idx in range(B1_JOINT_CNT):
                self.motor_cmds[idx].q = float(self.current_jpos_des[idx])
                self.motor_cmds[idx].dq = 0.0
                self.motor_cmds[idx].tau = 0.0

                if self.upper_body_only and idx not in UPPER_BODY_JOINT_INDICES:
                    # Do not control legs in walking mode.
                    self.motor_cmds[idx].kp = 0.0
                    self.motor_cmds[idx].kd = 0.0
                    self.motor_cmds[idx].weight = 0.0
                else:
                    self.motor_cmds[idx].kp = self.kp_data[idx]
                    self.motor_cmds[idx].kd = self.kd_data[idx]
                    self.motor_cmds[idx].weight = self.weight

            self.low_cmd = LowCmd()
            self.low_cmd.cmd_type = self.control_mode
            self.low_cmd.motor_cmd = self.motor_cmds
            ok = self.low_cmd_publisher.Write(self.low_cmd)
            if not ok:
                logger.debug("LowCmd Write() returned false")

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
            if self.upper_body_only:
                # In walking mode only update 14 arm joints. Do not force legs to
                # FIXED_HOME_POSITION; locomotion owns them.
                self.q_target = self._make_upper_body_target(arm_target_14=arm_target_14)
            else:
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
        if self.upper_body_only:
            print("[Warn] upper_body_only=True: go_home() would command legs, use go_upper_body_home() instead.")
            return self.go_upper_body_home(duration=duration)
        self._move_to_pose_blocking(
            self.FIXED_HOME_POSITION,
            duration=duration,
            pose_name="fixed home position"
        )

    # Move only the upper body/arms to the fixed home posture.
    def go_upper_body_home(self, duration=5.0):
        logger.info(f"Moving upper body to fixed home arms, duration: {duration} seconds...")
        target = self._make_upper_body_target(use_fixed_home_arms=True)
        self.ctrl_all_joints(target)
        time.sleep(duration)
        print("Reached upper-body home position")
    
    def start_control(self):
        logger.info("Starting upper-body control...")
        self._try_print_robot_mode()
        if self.upper_body_only:
            if not self._call_upper_body_custom_control(True):
                raise RuntimeError("UpperBodyCustomControl(True) failed")

        # Initialize targets from the actual robot state to avoid jumps.
        with self.ctrl_lock:
            current_q = self.get_current_joint_angles()
            self.current_jpos_des = current_q.copy()
            self.q_target = current_q.copy()

        self._publish_enabled = True
        self.weight = 0.0
        while self.weight < 1.0:
            self.weight += self.weight_margin
            self.weight = min(self.weight, 1.0)
            time.sleep(self.control_dt)
    
    def stop_control(self):
        logger.info("Stopping upper-body control...")
        while self.weight > 0.0:
            self.weight -= self.weight_margin
            self.weight = max(self.weight, 0.0)
            time.sleep(self.control_dt)
        self._publish_enabled = False
        if self.upper_body_only and self.upper_body_custom_enabled:
            self._call_upper_body_custom_control(False)
    
    def start_vr_control(self):
        if not self.use_ik:
            print("IK solver not enabled, cannot perform VR control")
            return False
        print("Starting VR control...")
        with self.ctrl_lock:
            if self.upper_body_only:
                current_q = self._make_upper_body_target(use_fixed_home_arms=True)
            else:
                current_q = self.FIXED_HOME_POSITION.copy()
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
        try:
            if self._publish_enabled or self.upper_body_custom_enabled:
                self.stop_control()
        except Exception as e:
            print(f"[Warn] stop_control during cleanup failed: {e}")
        self._publish_running = False
        self._subscribe_running = False
        self.vr_controller.stop()
        time.sleep(0.1)
        try:
            self.low_cmd_publisher.CloseChannel()
        except Exception:
            pass
        try:
            self.low_state_subscriber.CloseChannel()
        except Exception:
            pass
        logger.info("Robot controller resource cleanup completed")
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== B1 Upper-Body Walking Teleop ===")
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} networkInterface")
        print("Example: python controller.py enp6s0")
        sys.exit(-1)

    network_interface = sys.argv[1]
    controller = None

    try:
        # Create controller instance. In walking mode it only publishes upper-body joints.
        controller = B1RobotController(
            network_interface=network_interface,
            simulation_mode=False,
            use_ik=True,
            visualize_ik=False,
            upper_body_only=True
        )
        time.sleep(1.0)

        input("Switch the robot to walking mode, make sure it is stable, then press Enter: ")

        # Start upper-body control while walking. This does not switch to custom mode.
        controller.start_control()
        controller.go_upper_body_home(duration=5.0)

        input("Start the VR device, make sure the upper body is at home, then press Enter: ")

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
                    print(f"Real-time targets sent: {target_count}")
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
        print("\nCleaning up resources...")
        if controller is not None:
            try:
                controller.stop_control()
            except Exception as e:
                print(f"[Warn] stop_control failed: {e}")
            try:
                controller.cleanup()
            except Exception as e:
                print(f"[Warn] cleanup failed: {e}")
        print("\nProgram ended")
