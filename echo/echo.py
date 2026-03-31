import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import os

# ===== 加载模型 =====
import os

# ===== 加载模型 =====
# 使用 os.path.expanduser 自动解析 "~" 符号，确保路径正确
xml_path = os.path.expanduser("~/ytf/robosuite/robosuite/models/assets/robots/t1/robot.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

nu = model.nu

# ===== 映射 =====
qpos_ids, qvel_ids = [], []
for i in range(nu):
    jid = model.actuator_trnid[i, 0]
    qpos_ids.append(model.jnt_qposadr[jid])
    qvel_ids.append(model.jnt_dofadr[jid])
qpos_ids = np.array(qpos_ids)
qvel_ids = np.array(qvel_ids)

name_to_id = {model.actuator(i).name: i for i in range(nu)}
def idx(n): return name_to_id[n]

# ===== PD =====
Kp, Kd = 400, 40
q_des = np.zeros(nu)

# ===== 站姿 (下肢保持不动) =====
stand = {
    "hip": -0.25,
    "knee": 0.5,
    "ankle": -0.25
}

# 根据你的 XML 修改了 set_leg 里的名字拼接逻辑
def set_leg(prefix, hip, knee, ankle):
    hip_name = f"act_{prefix}_Hip_Pitch"
    knee_name = f"act_{prefix}_Knee_Pitch"
    ankle_name = f"act_{prefix}_Ankle_Pitch"
    
    if hip_name in name_to_id: q_des[idx(hip_name)] = hip
    if knee_name in name_to_id: q_des[idx(knee_name)] = knee
    if ankle_name in name_to_id: q_des[idx(ankle_name)] = ankle

# 初始化时锁死双腿
set_leg("leg_Left",  stand["hip"], stand["knee"], stand["ankle"])
set_leg("leg_Right", stand["hip"], stand["knee"], stand["ankle"])

# ===== 加载遥操作数据 =====
# TODO: 确认 parquet 路径
parquet_file_path = "episode_000080.parquet" 
df = pd.read_parquet(parquet_file_path)

if 'timestamp' in df.columns:
    recorded_times = df['timestamp'].values
elif 'frame_index' in df.columns:
    recorded_times = df['frame_index'].values * 0.02 
else:
    recorded_times = np.arange(len(df)) * 0.02
    
recorded_positions = np.stack(df['observation.state'].values)
max_record_time = recorded_times[-1]

# ===== 机械臂关节名称映射 (完全匹配 XML Actuator Name) =====
# 这里的顺序假设 Parquet 里的 14 维数据是 [左臂7个, 右臂7个]
arm_joint_names = [
    "act_Left_Shoulder_Pitch", "act_Left_Shoulder_Roll", "act_Left_Elbow_Pitch", 
    "act_Left_Elbow_Yaw", "act_Left_Wrist_Pitch", "act_Left_Wrist_Yaw", "act_Left_Hand_Roll",
    
    "act_Right_Shoulder_Pitch", "act_Right_Shoulder_Roll", "act_Right_Elbow_Pitch", 
    "act_Right_Elbow_Yaw", "act_Right_Wrist_Pitch", "act_Right_Wrist_Yaw", "act_Right_Hand_Roll"
]

# ===== 仿真循环 =====
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        sim_time = data.time

        # ===== 根据时间插值生成手臂 q_des =====
        if sim_time <= max_record_time:
            for i, j_name in enumerate(arm_joint_names):
                # 确保驱动器名字正确，且没有超出数据维度
                if j_name in name_to_id and i < recorded_positions.shape[1]:
                    q_des[idx(j_name)] = np.interp(sim_time, recorded_times, recorded_positions[:, i])

        # ===== PD torque =====
        q = data.qpos[qpos_ids]
        qd = data.qvel[qvel_ids]
        
        tau = Kp * (q_des - q) - Kd * qd
        data.ctrl[:] = tau

        mujoco.mj_step(model, data)
        viewer.sync()
