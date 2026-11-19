from setuptools import setup

package_name = 'mppi_racer'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ADP team',
    maintainer_email='todo@todo.todo',
    description='MPPI integrated planner and controller',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': ['mppi_racer = mppi_racer.node:main']},
)
