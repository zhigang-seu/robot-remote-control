# robot-remote-control
This project implements VR-based teleoperation control for the T1 7-DOF dual-arm robot, including real-time arm motion control via VR positioning and dexterous hand grip control via VR trigger keys.

# Features
Real-time dual-arm motion control using VR position/orientation data
Dexterous hand grip control (open/close) via VR trigger buttons
Inverse Kinematics (IK) solver for 7DOF arms with Pinocchio
Thread-safe robot state management and control command publishing
Smooth joint motion control with velocity limiting and PID parameters

# Environment Setup
1. Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
2. Create and Activate Conda Environment
conda create -n teleop_lxk python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
conda activate teleop_lxk
3. Additional Dependencies
pip install booster-robotics-sdk-python (adjust based on actual SDK name)

# Teleoperation Workflow
1. Navigate to Project Directory
cd /home/master/Workspace/test_lxk/booster_robotics_sdk-main/example/low_level/
conda activate teleop_lxk
2. Terminal 1: Robot Mode Configuration
python3 change_mode.py 127.0.0.1
After running, input mp and press Enter to enter Preparation Mode
3. Terminal 2: Main Robot Controller
python3 t1_robot_controller2.py 127.0.0.1
After the controller starts:
Switch back to Terminal 1 and input mc to enter Custom Mode
Power on and activate the VR device
Wait for the robot to reach the home position
Press Enter in Terminal 2 to start VR control
4. Terminal 3: VR Hand Grip Contro
python3 vr_hand.py 127.0.0.1
5. Start Teleoperation

##########################################################################
#数据集采集（按顺序执行）
开机，192.168.10.101无线连运控板，密码123456
#cssh（时间戳对齐）
cssh master@192.168.10.101 booster@192.168.10.102
timedatectl set-time '2026-3-22 1:30:30'
终端1：
sudo nft flush ruleset
cd /home/master/Workspace/test_lxk/ros2_ws
conda activate teleop_lxk
export BOOSTER_SDK_ROOT=/home/master/Workspace/test_lxk/booster_robotics_sdk-main
export CONDA_SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export USER_SITE=$HOME/.local/lib/python3.10/site-packages
export PYTHONPATH=$USER_SITE:$CONDA_SITE:/opt/ros/humble/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source install/setup.bash

终端2：
cd /home/master/Workspace/test_lxk/ros2_ws
conda activate teleop_lxk
export BOOSTER_SDK_ROOT=/home/master/Workspace/test_lxk/booster_robotics_sdk-main
export CONDA_SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export USER_SITE=$HOME/.local/lib/python3.10/site-packages
export PYTHONPATH=$USER_SITE:$CONDA_SITE:/opt/ros/humble/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source install/setup.bash

终端3：
cd /home/master/Workspace/test_lxk/booster_robotics_sdk-main/example/low_level/
conda activate teleop_lxk
python3 change_mode.py 127.0.0.1
输入mp进入准备模式

打开pico头显进入遥操作环境
终端1：
ros2 launch booster_lerobot_bridge bringup.launch.py
5s内！：

终端3：
输入mc进入自定义模式
等机器人到达home位！

终端2：
ros2 service call /start_episode std_srvs/srv/Trigger#开始录制
同时开始遥操作完成抓取
ros2 service call /save_episode std_srvs/srv/Trigger#录完保存
##########################################################################

#测试话题正常
ros2 topic echo /teleop/arm_action --once
ros2 topic echo /robot/arm_state --once
ros2 topic echo /teleop/hand_action --once
ros2 topic echo /robot/hand_state --once
ros2 topic echo /camera/camera/color/image_raw --once

ros2 service call /start_episode std_srvs/srv/Trigger#开始录制
ros2 service call /save_episode std_srvs/srv/Trigger#录完保存
ros2 service call /discard_episode std_srvs/srv/Trigger#丢弃


