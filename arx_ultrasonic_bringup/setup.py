import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arx_ultrasonic_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'param'), glob('param/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='boat1470',
    maintainer_email='claudeseat1@intronics.co.th',
    description='HC-SR04 on the Pi for the level crossing bar.',
    license='Apache 2.0',
    entry_points={
        'console_scripts': [
            'detect_level_ultrasonic = arx_ultrasonic_bringup.detect_level_ultrasonic:main',
        ],
    },
)
