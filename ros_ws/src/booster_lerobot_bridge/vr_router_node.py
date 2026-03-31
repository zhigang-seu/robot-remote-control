#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import socket
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VRRouterNode(Node):
    def __init__(self):
        super().__init__('vr_router')
        self.declare_parameter('vr_ip', '192.168.0.8')
        self.declare_parameter('vr_port', 8000)
        self.declare_parameter('hello_message', 'This is ROS2 VR router!')

        self.vr_ip = self.get_parameter('vr_ip').value
        self.vr_port = int(self.get_parameter('vr_port').value)
        self.hello_message = self.get_parameter('hello_message').value

        self.pub_left = self.create_publisher(String, '/vr/left', 10)
        self.pub_right = self.create_publisher(String, '/vr/right', 10)
        self.pub_key = self.create_publisher(String, '/vr/key', 10)
        self.pub_all = self.create_publisher(String, '/vr/raw', 10)

        self.sock = None
        self.buffer = ''
        self.running = True
        self.thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.connect()
        self.thread.start()
        self.get_logger().info(f'Connected to VR server {self.vr_ip}:{self.vr_port}')

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((self.vr_ip, self.vr_port))
        self.sock.sendall(self.hello_message.encode('utf-8'))

    def extract_json_objects(self, s: str):
        objs = []
        start_search = 0
        tail_start = 0
        while True:
            start = s.find('{', start_search)
            if start == -1:
                tail_start = len(s)
                break
            depth = 0
            end = -1
            for i in range(start, len(s)):
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                if depth == 0:
                    end = i
                    break
            if end == -1:
                tail_start = start
                break
            objs.append(s[start:end + 1])
            start_search = end + 1
            tail_start = start_search
        return objs, s[tail_start:]

    def route_one(self, raw: str):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        who = msg.get('who', '')
        ros_msg = String()
        ros_msg.data = raw
        self.pub_all.publish(ros_msg)
        if who == 'left':
            self.pub_left.publish(ros_msg)
        elif who == 'right':
            self.pub_right.publish(ros_msg)
        elif who == 'key':
            self.pub_key.publish(ros_msg)

    def recv_loop(self):
        while self.running:
            try:
                data = self.sock.recv(4096)
                if not data:
                    self.get_logger().error('VR socket closed by peer')
                    break
                self.buffer += data.decode('utf-8', errors='ignore')
                objs, tail = self.extract_json_objects(self.buffer)
                self.buffer = tail
                for raw in objs:
                    self.route_one(raw)
            except Exception as exc:
                self.get_logger().error(f'VR receive error: {exc}')
                break
        self.running = False

    def destroy_node(self):
        self.running = False
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VRRouterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
