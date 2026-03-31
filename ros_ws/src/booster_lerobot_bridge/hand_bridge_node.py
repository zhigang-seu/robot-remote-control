#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from booster_lerobot_bridge.sdk_paths import add_sdk_paths

HAND_NAMES = ['l_hand_grab', 'r_hand_grab']


class HandBridgeNode(Node):
    def __init__(self):
        super().__init__('hand_bridge')

        self.declare_parameter('rate_hz', 200.0)
        self.declare_parameter('grab_threshold', 0.5)

        add_sdk_paths()
        from vr_hand import VRHandController  # type: ignore

        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.grab_threshold = float(self.get_parameter('grab_threshold').value)

        self.controller = VRHandController()
        self.last_action = np.asarray([0.0, 0.0], dtype=np.float32)

        self.pub_action = self.create_publisher(JointState, '/teleop/hand_action', 10)
        self.pub_state = self.create_publisher(JointState, '/robot/hand_state', 10)
        self.create_subscription(String, '/vr/key', self.key_cb, 10)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)
        self.get_logger().info(f'hand_bridge started; rate_hz={self.rate_hz}')

    def _command_hand(self, hand_side: str, grab: bool):
        idx = 0 if hand_side == 'left' else 1
        target = 1.0 if grab else 0.0

        # 边沿检测：状态没变就不重复发命令
        if float(self.last_action[idx]) == target:
            return

        try:
            if grab:
                ret = self.controller.hand_close(hand_side)
            else:
                ret = self.controller.hand_open(hand_side)

            self.last_action[idx] = target

            # 这里继续维护 controller 里的本地状态，避免其它地方读到旧值
            self.controller.current_grab_state[hand_side] = target

            if ret != 0:
                self.get_logger().warning(
                    f'hand command failed: side={hand_side}, target={target}, ret={ret}'
                )
        except Exception as exc:
            self.get_logger().error(f'hand command exception: side={hand_side}, exc={exc}')

    def key_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            if data.get('who', '') != 'key':
                return

            key_type = data.get('key', '')
            trigger_value = float(data.get('value', 0.0))
            grab = trigger_value > self.grab_threshold

            if key_type == 'LeftTrigger':
                self._command_hand('left', grab)
            elif key_type == 'RightTrigger':
                self._command_hand('right', grab)

        except Exception as exc:
            self.get_logger().error(f'hand key callback failed: {exc}')

    def hand_state(self):
        # 临时方案：state 直接等于 action
        return self.last_action.copy()

    def publish_joint_state(self, pub, names, values, stamp):
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = list(names)
        msg.position = [float(v) for v in np.asarray(values, dtype=np.float32).tolist()]
        pub.publish(msg)

    def on_timer(self):
        try:
            stamp = self.get_clock().now().to_msg()

            action = self.last_action.copy()
            state = self.hand_state()

            self.publish_joint_state(self.pub_action, HAND_NAMES, action, stamp)
            self.publish_joint_state(self.pub_state, HAND_NAMES, state, stamp)

        except Exception as exc:
            self.get_logger().error(f'hand publish failed: {exc}')

    def destroy_node(self):
        try:
            self.controller.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()