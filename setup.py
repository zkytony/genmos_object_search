# ## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

IN_ROS_WS = False

if IN_ROS_WS:
    from setuptools import setup
    from catkin_pkg.python_setup import generate_distutils_setup

    # fetch values from package.xml
    setup_args = generate_distutils_setup(
        packages=['sloop_ros', 'sloop'],
        package_dir={'': 'src'}
    )

    setup(**setup_args)

else:
    from setuptools import setup

    setup(name='sloop-ros',
          packages=['sloop_ros', 'sloop'],
          package_dir={'': 'src'},
          version='0.0',
          description='SLOOP ROS',
          python_requires='>3.6',
          install_requires=[
              'pyyaml',
              'numpy',
          ],
          author='Kaiyu Zheng',
          author_email='kaiyutony@gmail.com'
         )
