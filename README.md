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
# Project Structure
booster_robotics_sdk-main/example/low_level/
├── b1_robot_controller2.py  
├── vr_hand.py              
├── t1_7dof_arm_ik2.py    
├── change_mode.py         
└── assets/                 
    └── T1_7DofArm_Serial.urdf
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

