#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import threading
import time
from copy import deepcopy

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from booster_lerobot_bridge.sdk_paths import add_sdk_paths

ARM_NAMES = [*[f'l_arm_j{i}' for i in range(1, 8)], *[f'r_arm_j{i}' for i in range(1, 8)]]


class ArmBridgeNode(Node):
    def __init__(self):
        super().__init__('arm_bridge')

        self.declare_parameter('rate_hz', 200.0)
        self.declare_parameter('simulation_mode', False)
        self.declare_parameter('use_ik', True)
        self.declare_parameter('go_home_duration_s', 5.0)
        self.declare_parameter('auto_enable', False)

        self.declare_parameter('startup_go_home', False)
        self.declare_parameter('startup_home0_delay_s', 10.0)
        self.declare_parameter('startup_fixed_home_delay_s', 5.0)
        self.declare_parameter('ik_rate_hz', 60.0)
        self.declare_parameter('vr_topic_qos_depth', 1)

        add_sdk_paths()
        from t1_robot_controller2 import B1RobotController  # type: ignore

        self.rate_hz = float(self.get_parameter('rate_hz').value)
        simulation_mode = bool(self.get_parameter('simulation_mode').value)
        use_ik = bool(self.get_parameter('use_ik').value)
        self.go_home_duration_s = float(self.get_parameter('go_home_duration_s').value)
        auto_enable = bool(self.get_parameter('auto_enable').value)

        self.startup_go_home = bool(self.get_parameter('startup_go_home').value)
        self.startup_home0_delay_s = float(self.get_parameter('startup_home0_delay_s').value)
        self.startup_fixed_home_delay_s = float(self.get_parameter('startup_fixed_home_delay_s').value)
        self.ik_rate_hz = float(self.get_parameter('ik_rate_hz').value)
        self.vr_topic_qos_depth = int(self.get_parameter('vr_topic_qos_depth').value)

        self.controller = B1RobotController(
            simulation_mode=simulation_mode,
            use_ik=use_ik,
            visualize_ik=False,
        )

        self.enabled = False
        self.go_home_running = False
        self.startup_go_home_done = False
        self.startup_go_home_timer = None

        self.last_action = np.asarray(
            self.controller.get_current_arm_joint_angles(),
            dtype=np.float32
        )

        self.pub_action = self.create_publisher(JointState, '/teleop/arm_action', 10)
        self.pub_state = self.create_publisher(JointState, '/robot/arm_state', 10)

        self.pose_lock = threading.Lock()
        self.action_lock = threading.Lock()
        self.latest_pose = {'left': None, 'right': None}
        self.latest_pose_stamp_ns = {'left': 0, 'right': 0}
        self.latest_ik_target = None
        self.last_ik_time = 0.0
        self.ik_busy = False
        self.ik_drop_count = 0
        self.pose_update_count = 0

        self.create_subscription(String, '/vr/left', self.left_pose_cb, self.vr_topic_qos_depth)
        self.create_subscription(String, '/vr/right', self.right_pose_cb, self.vr_topic_qos_depth)

        self.create_service(Trigger, '~/enable', self.enable_cb)
        self.create_service(Trigger, '~/disable', self.disable_cb)
        self.create_service(Trigger, '~/go_home', self.go_home_cb)

        self.pub_timer = self.create_timer(1.0 / self.rate_hz, self.on_publish_timer)
        ik_period = 1.0 / self.ik_rate_hz if self.ik_rate_hz > 0 else 0.02
        self.ik_timer = self.create_timer(ik_period, self.on_ik_timer)

        if self.startup_go_home:
            self.startup_go_home_timer = self.create_timer(
                self.startup_home0_delay_s,
                self.startup_go_home_cb
            )

        self.get_logger().info(
            'arm_bridge started; '
            f'rate_hz={self.rate_hz}, '
            f'auto_enable={auto_enable}, '
            f'startup_go_home={self.startup_go_home}, '
            f'startup_home0_delay_s={self.startup_home0_delay_s}, '
            f'startup_fixed_home_delay_s={self.startup_fixed_home_delay_s}, '
            f'ik_rate_hz={self.ik_rate_hz}, '
            f'vr_topic_qos_depth={self.vr_topic_qos_depth}'
        )

        if auto_enable:
            try:
                self.enable_robot()
            except Exception as exc:
                self.get_logger().error(f'auto enable failed: {exc}')

    def enable_robot(self):
        if self.enabled:
            return

        self.get_logger().info('Starting arm control weight ramp...')
        self.controller.start_control()

        self.enabled = True
        self.last_action = np.asarray(
            self.controller.get_current_arm_joint_angles(),
            dtype=np.float32
        )
        self.get_logger().info('Arm bridge enabled')

    def disable_robot(self):
        if not self.enabled:
            return

        self.controller.stop_control()
        self.enabled = False
        self.get_logger().info('Arm bridge disabled')

    def do_go_home(self):
        if not self.enabled:
            raise RuntimeError('arm is not enabled')

        if self.go_home_running:
            self.get_logger().warn('go_home already running, skip this request')
            return

        self.go_home_running = True
        try:
            self.get_logger().info('Moving robot to fixed home pose...')
            self.controller.go_home(duration=self.go_home_duration_s)
            self.last_action = np.asarray(
                self.controller.get_current_arm_joint_angles(),
                dtype=np.float32
            )
            self.get_logger().info('go_home finished')
        finally:
            self.go_home_running = False

    def do_startup_home_sequence(self):
        if not self.enabled:
            raise RuntimeError('arm is not enabled')

        if self.go_home_running:
            self.get_logger().warn('startup home sequence already running')
            return

        self.go_home_running = True
        try:
            self.get_logger().info('startup sequence step1: moving to home0...')
            self.controller.go_home0(duration=self.go_home_duration_s)
            self.last_action = np.asarray(
                self.controller.get_current_arm_joint_angles(),
                dtype=np.float32
            )

            self.get_logger().info(
                f'startup sequence waiting {self.startup_fixed_home_delay_s} seconds before fixed home...'
            )
            time.sleep(self.startup_fixed_home_delay_s)

            self.get_logger().info('startup sequence step2: moving to FIXED_HOME_POSITION...')
            self.controller.go_home(duration=self.go_home_duration_s)
            self.last_action = np.asarray(
                self.controller.get_current_arm_joint_angles(),
                dtype=np.float32
            )
            self.get_logger().info('startup home sequence finished')
        finally:
            self.go_home_running = False

    def _do_go_home_background(self):
        try:
            self.do_go_home()
        except Exception as exc:
            self.get_logger().error(f'go_home failed: {exc}')

    def _do_startup_home_sequence_background(self):
        try:
            self.do_startup_home_sequence()
        except Exception as exc:
            self.get_logger().error(f'startup home sequence failed: {exc}')

    def startup_go_home_cb(self):
        if self.startup_go_home_done:
            return

        if self.go_home_running:
            return

        if not self.enabled:
            self.get_logger().info('startup home sequence waiting for arm enable...')
            return

        self.startup_go_home_done = True

        if self.startup_go_home_timer is not None:
            self.startup_go_home_timer.cancel()

        self.get_logger().info('startup home sequence triggered')
        threading.Thread(target=self._do_startup_home_sequence_background, daemon=True).start()

    def enable_cb(self, request, response):
        try:
            self.enable_robot()
            response.success = True
            response.message = 'arm enabled'
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def disable_cb(self, request, response):
        try:
            self.disable_robot()
            response.success = True
            response.message = 'arm disabled'
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def go_home_cb(self, request, response):
        try:
            if not self.enabled:
                response.success = False
                response.message = 'arm is not enabled'
                return response

            if self.go_home_running:
                response.success = False
                response.message = 'go_home already running'
                return response

            threading.Thread(target=self._do_go_home_background, daemon=True).start()
            response.success = True
            response.message = 'go_home started in background'
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _store_pose_msg(self, msg: String, expected_side: str):
        if not self.enabled or self.go_home_running:
            return

        try:
            data = json.loads(msg.data)
            who = data.get('who', '')
            if who != expected_side:
                return
            stamp = self.get_clock().now().nanoseconds
            with self.pose_lock:
                self.latest_pose[expected_side] = deepcopy(data)
                self.latest_pose_stamp_ns[expected_side] = stamp
                self.pose_update_count += 1
        except Exception as exc:
            self.get_logger().error(f'arm pose callback failed: {exc}')

    def left_pose_cb(self, msg: String):
        self._store_pose_msg(msg, 'left')

    def right_pose_cb(self, msg: String):
        self._store_pose_msg(msg, 'right')

    def process_latest_pose(self):
        if not self.enabled or self.go_home_running:
            return
        if self.ik_busy:
            return

        with self.pose_lock:
            left = deepcopy(self.latest_pose['left'])
            right = deepcopy(self.latest_pose['right'])

        if left is None and right is None:
            return

        self.ik_busy = True
        try:
            if left is not None:
                self.controller.vr_controller.update_hand_state(left, 'left')
            if right is not None:
                self.controller.vr_controller.update_hand_state(right, 'right')

            target = self.controller.vr_controller.solve_from_current_state()
            if target is not None and len(target) == 14:
                self.controller.ctrl_dual_arm_14dof(target)
                target_np = np.asarray(target, dtype=np.float32)
                with self.action_lock:
                    self.last_action = target_np
                    self.latest_ik_target = target_np
                self.last_ik_time = time.monotonic()
        except Exception as exc:
            self.get_logger().error(f'arm IK processing failed: {exc}')
        finally:
            self.ik_busy = False

    def _process_latest_pose_background(self):
        self.process_latest_pose()

    def on_ik_timer(self):
        if not self.enabled or self.go_home_running or self.ik_busy:
            return
        threading.Thread(target=self._process_latest_pose_background, daemon=True).start()

    def publish_joint_state(self, pub, names, values, stamp):
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = list(names)
        msg.position = [float(v) for v in np.asarray(values, dtype=np.float32).tolist()]
        pub.publish(msg)

    def on_publish_timer(self):
        try:
            stamp = self.get_clock().now().to_msg()
            state = np.asarray(self.controller.get_current_arm_joint_angles(), dtype=np.float32)
            with self.action_lock:
                action = np.asarray(self.last_action, dtype=np.float32).copy()
            self.publish_joint_state(self.pub_state, ARM_NAMES, state, stamp)
            self.publish_joint_state(self.pub_action, ARM_NAMES, action, stamp)
        except Exception as exc:
            self.get_logger().error(f'arm publish failed: {exc}')

    def destroy_node(self):
        try:
            if self.enabled:
                self.controller.stop_control()
        except Exception:
            pass

        try:
            self.controller.cleanup()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
