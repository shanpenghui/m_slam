import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    remappings = [
        ('/scan', '/scan'),
        ('/odom', '/odom'),
    ]
    config = os.path.join(
        get_package_share_directory(package_name='m_vins'),
        'cfg',
        'launch_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='m_vins',
            executable='m_vins_node',
            output='screen',
            parameters=[config],
            remappings=remappings
        )
    ])
