import numpy as np
import threading
import time
import logging
import sys
import socket
import json
from queue import Queue
from collections import defaultdict


from booster_robotics_sdk_python import ChannelFactory, B1LocoClient
from booster_robotics_sdk_python import BoosterHandType, HandIndex, DexterousFingerParameter

logger = logging.getLogger(__name__)

CONTROL_DT = 0.002
VR_IP = "192.168.0.8"
VR_PORT = 8000
GRAB_THRESHOLD = 0.5 


class VRHandController:
    def __init__(self):
        self.running = False
        self.sock = None
        self.msg_queue = Queue()
        self.current_grab_state = {
            "left": 0.0,
            "right": 0.0
        }

        ChannelFactory.Instance().Init(0)
        self.loco_client = B1LocoClient()
        self.loco_client.Init()


    def connect_vr(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((VR_IP, VR_PORT))
            print(f"Connected to VR glasses {VR_IP}:{VR_PORT}")
            self.sock.sendall("This is hand control robot!".encode('utf-8'))
            return True
        except Exception as e:
            print(f"Failed to connect to VR glasses: {e}")
            return False

 
    def receive_vr_data(self):
        try:
            while self.running:
                data = self.sock.recv(1024)
                if not data:
                    print("VR glasses disconnected")
                    break
                self.msg_queue.put(data)
        except Exception as e:
            if self.running:
                print(f"VR receive error: {e}")


    def extract_json(self, s):
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1 and end > start:
            return s[start:end+1]
        return ""


    def hand_open(self, hand_side):
        try:
            finger_params = []

            for i in range(5):
                param = DexterousFingerParameter(i, 1000, 200, 800)
                finger_params.append(param)

            thumb_param = DexterousFingerParameter(5, 0, 200, 800)
            finger_params.append(thumb_param)

            hand_idx = HandIndex.kLeftHand if hand_side == "left" else HandIndex.kRightHand

            res = self.loco_client.ControlDexterousHand(finger_params, hand_idx, BoosterHandType.kInspireHand)
            if res == 0:
                print(f"{hand_side} hand open success")
                self.current_grab_state[hand_side] = 0.0
            return res
        except Exception as e:
            print(f"{hand_side} hand open error: {e}")
            return -1


    def hand_close(self, hand_side):
        try:
            finger_params = []

            for i in range(4):
                param = DexterousFingerParameter(i, 400, 400, 800)
                finger_params.append(param)

            pinky_param = DexterousFingerParameter(4, 500, 400, 800)
            finger_params.append(pinky_param)

            thumb_param = DexterousFingerParameter(5, 0, 400, 800)
            finger_params.append(thumb_param)

            hand_idx = HandIndex.kLeftHand if hand_side == "left" else HandIndex.kRightHand

            res = self.loco_client.ControlDexterousHand(finger_params, hand_idx, BoosterHandType.kInspireHand)
            if res == 0:
                print(f"{hand_side} hand close success")
                self.current_grab_state[hand_side] = 1.0
            return res
        except Exception as e:
            print(f"{hand_side} hand close error: {e}")
            return -1


    def parse_vr_key_data(self):
        if self.msg_queue.empty():
            return
        
        try:
            data = self.msg_queue.get()
            data_str = data.decode('utf-8')
            json_str = self.extract_json(data_str)
            if not json_str:
                return
            
            json_data = json.loads(json_str)
            who = json_data.get("who", "")

            if who == "key":
                key_type = json_data.get("key", "")
                trigger_value = json_data.get("value", 0.0)

                if key_type == "LeftTrigger":
                    if trigger_value > GRAB_THRESHOLD:
                        self.hand_close("left")
                    elif trigger_value <= GRAB_THRESHOLD:
                        self.hand_open("left")

                if key_type == "RightTrigger":
                    if trigger_value > GRAB_THRESHOLD:
                        self.hand_close("right")
                    elif trigger_value <= GRAB_THRESHOLD:
                        self.hand_open("right")
        except json.JSONDecodeError as e:
            pass
        except Exception as e:
            print(f"Parse VR key data error: {e}")


    def start(self):
        if not self.connect_vr():
            return False
        self.running = True

        recv_thread = threading.Thread(target=self.receive_vr_data, daemon=True)
        recv_thread.start()
        print("Hand control service started! Waiting VR trigger data...")
        return True

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        print("Hand control service stopped")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Only VR Key -> Dexterous Hand Control ===")

    hand_controller = VRHandController()
    
    try:

        if not hand_controller.start():
            sys.exit(-1)
        
        while True:
            hand_controller.parse_vr_key_data()
            time.sleep(CONTROL_DT)
            
    except KeyboardInterrupt:
        print("\nUser stop the program")
    except Exception as e:
        print(f"\nProgram error: {e}")
    finally:

        hand_controller.stop()
        print("\nAll resource released!")