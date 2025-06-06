from setuptools import find_packages, setup
from glob import glob

package_name = 'dacs_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ("share/" + package_name, glob("launch_folder/dacs_parametric_launch.py")), # add launch.py
        ("share/" + package_name, glob("resource/rviz_config.rviz")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        "console_scripts": [  # add entry point
            "agent = dacs_project.agent:main",
            "visualizer = dacs_project.visualizer:main"
        ],
    },
)
