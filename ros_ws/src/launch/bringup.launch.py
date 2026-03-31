from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='booster_lerobot_bridge',
            executable='vr_router_node',
            name='vr_router',
            output='screen',
            parameters=[{'vr_ip': '192.168.0.8', 'vr_port': 8000}],
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
                'auto_enable': True,
                'startup_go_home': True,
                'startup_home0_delay_s': 10.0,
                'startup_fixed_home_delay_s': 5.0,
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
                'root': '/home/master/Workspace/test_lxk/booster_vla_dataset',
                'repo_id': 'booster_vla_dataset',
                'task': 'pick up the object and place it into the target area',
                'fps': 30,
                'camera_key': 'head_rgb',
                'robot_type': 'booster_bimanual',
                'max_age_ms': 10.0,
                'lowdim_buffer_size': 100,
            }]
        ),
    ])
