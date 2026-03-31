from setuptools import setup

package_name = 'booster_lerobot_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/bringup.launch.py']),
        ('share/' + package_name, ['README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenAI',
    maintainer_email='support@example.com',
    description='ROS2 bridge between Booster SDK teleop/control scripts, RealSense topics, and LeRobot recording',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'vr_router_node = booster_lerobot_bridge.vr_router_node:main',
            'arm_bridge_node = booster_lerobot_bridge.arm_bridge_node:main',
            'hand_bridge_node = booster_lerobot_bridge.hand_bridge_node:main',
            'lerobot_recorder_node = booster_lerobot_bridge.lerobot_recorder_node:main',
        ],
    },
)
