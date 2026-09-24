import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arx_mission'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # glob is not recursive: any new subdirectory under param/ needs its
        # own line here, or its yaml is simply not installed and the node runs
        # on its declared defaults without saying anything.
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'param'), glob('param/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='boat1470',
    maintainer_email='claudeseat1@intronics.co.th',
    description='Mission sequencing for TurtleBot3 AutoRace.',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_lane = arx_mission.control_lane:main',
            'detect_intersection_sign = arx_mission.detect_intersection_sign:main',
            'detect_lane = arx_mission.detect_lane:main',
            'detect_traffic_light = arx_mission.detect_traffic_light:main',
            'mission_control = arx_mission.mission_control:main',
        ],
    },
)
