from setuptools import find_packages, setup

package_name = 'franka_openpi'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/openpi_franka.launch.xml',
            'launch/openpi_franka.launch.py',
            'launch/act_franka.launch.py',
            'launch/groot_franka.launch.py',
            'launch/camera_launcher.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='saifahmadgit',
    maintainer_email='saif.ahmad98745@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'openpi_client_node = franka_openpi.openpi_client_node:main',
            'groot_client_node = franka_openpi.groot_client_node:main',
            'act_client_node = franka_openpi.act_client_node:main',
            'hand_eye_calibrate = franka_openpi.hand_eye_calibrate:main',
            'camera_viewer = franka_openpi.camera_viewer:main',
        ],
    },
)
