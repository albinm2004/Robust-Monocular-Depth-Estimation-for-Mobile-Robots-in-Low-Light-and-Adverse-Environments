from setuptools import find_packages, setup

package_name = 'depth_estimation_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/depth_estimation_node.launch.py']),
        ('share/' + package_name + '/config', ['config/depth_estimation_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Albin',
    maintainer_email='albinmathew2004@gmail.com',
    description='Robust monocular depth estimation node for the Sherpa RP (Depth Anything V2 + enhancement).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_estimation_node = depth_estimation_node.depth_node:main',
        ],
    },
)
