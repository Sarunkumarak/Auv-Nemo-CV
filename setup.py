from setuptools import setup

package_name = 'balloon_tags'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sarunkumarak',
    maintainer_email='your@email.com',
    description='Balloon and marker detection ROS2 node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'balloon_tags = balloon_tags.balloon_tags:main',
        ],
    },
)