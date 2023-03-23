#!/bin/bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.
# run this to setup genmos_object_search automatically in your bdai container

# make sure we are inside docker container
if [ ! -f /.dockerenv ]; then
    color_echo "$RED" "Aborted! You are not inside a docker container!"
    exit 1
fi

current_pwd=$(pwd)
function exit_on_error {
    cd $current_pwd
    exit 1
}

# make sure we are in the expected path (i.e. root of genmos_object_search)
bdai_path=/workspaces/bdai
if [[ ! $current_pwd = *genmos_object_search ]]; then
    cd $bdai_path/ws/src/external/genmos_object_search
fi
genmos_repo_root=$(pwd)

# source tools
. "./scripts/tools.sh"

# install genmos_object_search python package
echo_info "Installing genmos_object_search python package..."
if ! python_package_installed genmos_object_search; then
    pip install Cython
    cd genmos_object_search
    pip install -e .
    if ! python_package_installed genmos_object_search; then
        echo_error "pip install genmos_object_search failed. Abort."
        exit_on_error
    fi
    echo_success "pip install genmos_object_search successful."
else
    echo -e "genmos_object_search is already installed."
fi

# build protos
echo_info "Building genmos protos..."
pip install grpcio grpcio-tools
cd $genmos_repo_root/genmos_object_search/src
source build_proto.sh
cd $genmos_repo_root
if [ $? -eq 1 ]; then
    echo_error "Error building genmos protos. Abort"
    exit_on_error
fi
echo_success "Build genmos protos successful."

# run pytest
echo_info "Running genmos pytest..."
echo_info "If an Open3D window pops up, press 'Q' to quit it so that the test can proceed"
cd $genmos_repo_root/genmos_object_search/tests/genmos_object_search/pytests
python -m pytest
if [ $? -eq 1 ]; then
    echo_error "Error on pytest in genmos. Abort"
    exit_on_error
fi
echo_success "genmos package pytest passed."

# setup ROS 2 package
# The source of the package should be present through a symlink in bdai/projects/search/ws/src
# We just need to build this package here.
echo_info "Setting up ROS 2 package..."
cd $genmos_repo_root
if [ ! -f $bdai_path/projects/search/ws/src/genmos_object_search_ros2 ]; then
    echo_error "genmos ROS2 package does not exist under $bdai_path/projects/search. Abort."
    exit_on_error
fi
useros2
cd $bdai_path/projects/search/ws
colcon build --packages-select genmos_object_search_ros2


# Run a simulation test under ROS2
if confirm "Run simulation search test in ROS2?"; then
    echo_error "Not yet implemented."
fi

# End by going back to where the user was
cd $current_pwd
