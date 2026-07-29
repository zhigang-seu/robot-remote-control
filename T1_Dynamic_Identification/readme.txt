# T1 机械臂动力学与负载辨识流程（mw 模式）

##########################################################################

生成训练轨迹 n：
python3 generate_excitation.py --config configs/t1_payload_config.json --side left --out-prefix trajectories/left_arm_payload_excitation_trajn
生成验证轨迹 m：
先修改 configs/t1_payload_config.json 中 excitation.random_seed，使轨迹 m 与轨迹 n 不同，然后执行：
python3 generate_excitation.py --config configs/t1_payload_config.json --side left --out-prefix trajectories/left_arm_payload_excitation_trajm
程序会分别生成 npz、csv、mat 和报告 json 文件。
检查生成报告，确认：
used_pinocchio = true
constraint_penalty 接近 0
关节位置、速度和加速度均未超过限制

##########################################################################

MATLAB 检查轨迹
打开 validate_excitation_t1.m，修改：
cfg.traj = fullfile('trajectories','left_arm_payload_excitation_trajn.mat');
cfg.side = 'left';
运行脚本，检查关节曲线、末端轨迹、逆动力学力矩和机器人动画。
然后将 cfg.traj 改为：
cfg.traj = fullfile('trajectories','left_arm_payload_excitation_trajm.mat');
再次运行，确认两条轨迹均无碰撞、突变或明显超限后再上真机。

##########################################################################

机器人进入 mw 模式
终端1：
cd /home/master/Workspace/test_lxk/booster_robotics_sdk-main/example/low_level/
conda activate teleop_lxk
python3 change_mode.py 127.0.0.1
先输入 mp，使机器人进入准备模式。
等待机器人站稳后输入 mw，使机器人进入行走模式。

##########################################################################

采集空载辨识数据
保持夹爪空载，不抓取任何物体。
终端2进入项目根目录并激活环境。
使用轨迹 n 采集训练日志：
python3 robot_collect_excitation.py 127.0.0.1 --traj trajectories/left_arm_payload_excitation_trajn.npz --side left --out logs/empty_id_A_leftn.csv
等待程序自动完成以下过程：
开启上半身控制
移动到轨迹起点
执行激励轨迹并记录数据
返回接管前姿态
关闭上半身控制
机器人稳定后，使用轨迹 m 采集验证日志：
python3 robot_collect_excitation.py 127.0.0.1 --traj trajectories/left_arm_payload_excitation_trajm.npz --side left --out logs/empty_id_A_leftm.csv

##########################################################################

预处理空载日志
预处理轨迹 n 日志作为训练集：
python3 preprocess_logs.py --input logs/empty_id_A_leftn.csv --out logs/processed/left_empty_train_trajn.npz --side left
预处理轨迹 m 日志作为验证集：
python3 preprocess_logs.py --input logs/empty_id_A_leftm.csv --out logs/processed/left_empty_val_trajm.npz --side left
预处理结果会同时生成 npz 数据和对应的 json 摘要。

##########################################################################

使用轨迹 n 辨识空载模型
python3 identify_empty_model.py --train logs/processed/left_empty_train_trajn.npz --side left --out models/left_empty_payload_prior_trajn.npz
生成文件：
models/left_empty_payload_prior_trajn.npz
models/left_empty_payload_prior_trajn.json
该模型作为后续在线负载辨识的空载先验模型。

##########################################################################

使用轨迹 m 验证空载模型
python3 validate_empty_model.py --model models/left_empty_payload_prior_trajn.npz --val logs/processed/left_empty_val_trajm.npz --out-dir results/left_empty_validation_trajm
验证结果位于：
results/left_empty_validation_trajm/summary.json
该目录还会生成各关节实测力矩与模型预测力矩的对比图。
重点查看：
status 是否为 good 或 acceptable_with_caveats
验证集 RMSE 是否接近训练集 RMSE
是否存在误差明显偏大的关节

##########################################################################

在线负载辨识
确认空载模型验证通过后，将待辨识物体固定在末端。
机器人重新稳定进入 mw 模式，然后执行：
python3 online_payload_estimator.py 127.0.0.1 --model models/left_empty_payload_prior_trajn.npz --traj trajectories/left_arm_payload_excitation_trajn.npz --side left
如果物体真实质量已知，例如约 350 g，可增加参数用于结果对比：
python3 online_payload_estimator.py 127.0.0.1 --model models/left_empty_payload_prior_trajn.npz --traj trajectories/left_arm_payload_excitation_trajn.npz --side left --known-mass 0.35
结果默认保存在 results 目录，包括：
robot_online_payload_时间戳.csv
robot_online_payload_时间戳.json
CSV 保存在线质量估计过程，JSON 保存最终质量、收敛情况和控制点发布情况。
