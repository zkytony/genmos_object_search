set -o xtrace
# install some thirdparty dependencies
if [[ ! $PWD = *ros2 ]]; then
    echo "You must be in the 'genmos_object_search_ros2' ROS2 package directory."
    return 1
fi

function python_package_installed {
    python -c "import $1; assert $1.__file__ is not None" 2> /dev/null
    if [ $? -eq 0 ]; then
        true && return
    else
        false
    fi
}

genmos_ros2_root=$(pwd)
if ! python_package_installed tf_transformations; then
    pip install -e $genmos_ros2_root/thirdparty/tf_transformations
    pip install -e $genmos_ros2_root/thirdparty/ros2_numpy
    # run tests
    python $genmos_ros2_root/thirdparty/ros2_numpy/test/test_geometry.py
    python $genmos_ros2_root/thirdparty/ros2_numpy/test/test_images.py
    python $genmos_ros2_root/thirdparty/ros2_numpy/test/test_occupancygrids.py
    python $genmos_ros2_root/thirdparty/ros2_numpy/test/test_pointclouds.py
    python $genmos_ros2_root/thirdparty/ros2_numpy/test/test_quat.py
fi
set +o xtrace
