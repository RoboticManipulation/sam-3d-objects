from setuptools import setup
from glob import glob
import os

package_name = 'sam3d_ros2_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SAM3D User',
    maintainer_email='user@example.com',
    description='ROS2 bridge for SAM 3D Objects using ZeroMQ',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'action_server = sam3d_ros2_bridge.action_server:main',
            'test_client = sam3d_ros2_bridge.test_client:main',
        ],
    },
)
