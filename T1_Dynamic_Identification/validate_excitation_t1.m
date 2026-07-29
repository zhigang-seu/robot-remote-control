%% Validate a generated T1 arm excitation trajectory against the URDF.
% Usage in MATLAB, from project root:
%   cd t1_payload_identification_project
%   matlab/validate_excitation_t1
clear; clc; close all;

cfg.urdf = fullfile('T1_7DofArm_Serial.urdf');
cfg.traj = fullfile('trajectories','left_arm_payload_excitation.mat');
cfg.side = 'left';              % 'left' or 'right'
cfg.showRobot = true;
cfg.showAnimation = true;
cfg.baseBody = 'Trunk';
if strcmp(cfg.side,'left')
    cfg.eeBody = 'left_hand_link';
else
    cfg.eeBody = 'right_hand_link';
end

S = load(cfg.traj);
t = S.t(:); q = S.q; qd = S.qd; qdd = S.qdd;
robot = importrobot(cfg.urdf);
robot.DataFormat = 'row';
robot.Gravity = [0 0 -9.81];

allJoints = getAllMovableJointNames(robot);
if strcmp(cfg.side,'left')
    armJoints = {'Left_Shoulder_Pitch','Left_Shoulder_Roll','Left_Elbow_Pitch','Left_Elbow_Yaw','Left_Wrist_Pitch','Left_Wrist_Yaw','Left_Hand_Roll'};
else
    armJoints = {'Right_Shoulder_Pitch','Right_Shoulder_Roll','Right_Elbow_Pitch','Right_Elbow_Yaw','Right_Wrist_Pitch','Right_Wrist_Yaw','Right_Hand_Roll'};
end
idx = zeros(1,numel(armJoints));
for i=1:numel(armJoints)
    idx(i)=find(strcmp(allJoints, armJoints{i}),1);
end

N = size(q,1); nAll = numel(allJoints);
qFull = zeros(N,nAll); qdFull = zeros(N,nAll); qddFull = zeros(N,nAll);
qFull(:,idx)=q; qdFull(:,idx)=qd; qddFull(:,idx)=qdd;
xyz = zeros(N,3); tauFull=zeros(N,nAll);
for k=1:N
    Tfk = getTransform(robot, qFull(k,:), cfg.eeBody, cfg.baseBody);
    xyz(k,:) = tform2trvec(Tfk);
    tauFull(k,:) = inverseDynamics(robot, qFull(k,:), qdFull(k,:), qddFull(k,:));
end
tauArm = tauFull(:,idx);

figure('Name','Joint states');
tiledlayout(3,1,'Padding','compact');
nexttile; plot(t,q); grid on; ylabel('q [rad]'); title('Joint position'); legend(armJoints,'Interpreter','none');
nexttile; plot(t,qd); grid on; ylabel('qd [rad/s]'); title('Joint velocity');
nexttile; plot(t,qdd); grid on; ylabel('qdd [rad/s^2]'); xlabel('t [s]'); title('Joint acceleration');

figure('Name','End-effector path'); plot3(xyz(:,1),xyz(:,2),xyz(:,3),'LineWidth',1.3); grid on; axis equal;
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]'); title('End-effector path');

figure('Name','URDF inverse dynamics torque'); plot(t,tauArm,'LineWidth',1.1); grid on;
xlabel('t [s]'); ylabel('tau [Nm]'); title('URDF inverse dynamics torque'); legend(armJoints,'Interpreter','none');

fprintf('\n===== Constraint check =====\n');
for i=1:numel(armJoints)
    fprintf('%s: q=[%.3f %.3f], |qd|max=%.3f, |qdd|max=%.3f, |tau|max=%.3f Nm\n', ...
        armJoints{i}, min(q(:,i)), max(q(:,i)), max(abs(qd(:,i))), max(abs(qdd(:,i))), max(abs(tauArm(:,i))));
end

if cfg.showRobot
    figure('Name','Initial pose'); show(robot,qFull(1,:),'Frames','off','Visuals','on'); axis equal; view(135,20);
end
if cfg.showAnimation
    figure('Name','Animation');
    for k=1:10:N
        show(robot,qFull(k,:),'Frames','off','Visuals','on','PreservePlot',false); axis equal; view(135,20); drawnow;
    end
end

function names = getAllMovableJointNames(robot)
    names = {};
    for i=1:numel(robot.Bodies)
        j = robot.Bodies{i}.Joint;
        if ~strcmp(j.Type,'fixed')
            names{end+1} = j.Name; %#ok<AGROW>
        end
    end
end
