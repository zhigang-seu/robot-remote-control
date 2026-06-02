from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    network_interface = LaunchConfiguration('network_interface')
    vr_ip = LaunchConfiguration('vr_ip')
    dataset_root = LaunchConfiguration('dataset_root')

    return LaunchDescription([
        DeclareLaunchArgument('network_interface', default_value=''),
        DeclareLaunchArgument('vr_ip', default_value='192.168.0.8'),
        DeclareLaunchArgument('dataset_root', default_value='/home/master/Workspace/test_lxk/booster_vla_dataset_mw'),

        Node(
            package='booster_lerobot_bridge',
            executable='vr_router_node',
            name='vr_router',
            output='screen',
            parameters=[{'vr_ip': vr_ip, 'vr_port': 8000}],
        ),
        Node(
            package='booster_lerobot_bridge',
            executable='arm_bridge_node',
            name='arm_bridge',
            output='screen',
            parameters=[{
                'rate_hz': 200.0,
                'simulation_mode': False,
                'use_ik': True,
                'go_home_duration_s': 5.0,
                # Safer for mw: first switch robot to walking mode, then call /arm_bridge/enable.
                'auto_enable': False,
                'network_interface': network_interface,
                'upper_body_only': True,
                'startup_go_home': False,
                'startup_home_delay_s': 2.0,
                'ik_rate_hz': 60.0,
                'vr_topic_qos_depth': 1,
            }],
        ),
        Node(
            package='booster_lerobot_bridge',
            executable='hand_bridge_node',
            name='hand_bridge',
            output='screen',
            parameters=[{'rate_hz': 200.0, 'grab_threshold': 0.5, 'vr_topic_qos_depth': 1}],
        ),
        Node(
            package='booster_lerobot_bridge',
            executable='lerobot_recorder_node',
            name='lerobot_recorder',
            output='screen',
            parameters=[{
                'root': dataset_root,
                'repo_id': 'booster_vla_dataset_mw',
                'task': 'pick up the object and place it into the target area while walking mode upper-body teleop is enabled',
                'fps': 30,
                'camera_key': 'head_rgb',
                'robot_type': 'booster_bimanual_mw_upperbody',
                'max_age_ms': 15.0,
                'lowdim_buffer_size': 100,
            }],
        ),
    ])
