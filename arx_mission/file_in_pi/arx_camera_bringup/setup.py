import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arx_camera_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='boat1470',
    maintainer_email='claudeseat1@intronics.co.th',
    description='Robot-only launch files for TurtleBot3 AutoRace.',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': []},
)
