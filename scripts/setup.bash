# run this to setup genmos_object_search automatically in your bdai container
if [ ! -f /.dockerenv ]; then
    color_echo "$RED" "Aborted! You are not inside a docker container!"
    exit 1
fi

# make sure we are in the expected path (i.e. root of genmos_object_search)
bdai_path=/workspaces/bdai
current_pwd=$(pwd)
if [[ ! $current_pwd = *genmos_object_search ]]; then
    cd $bdai_path/ws/src/external/genmos_object_search
fi

# source tools
. "./scripts/tools.sh"

# install genmos_object_search python package
pip install Cython
cd genmos_object_search
pip install -e .
if ! python_package_installed genmos_object_search; then
    echo "pip install genmos_object_search failed. Abort."
    exit 1
fi
echo_success "pip install genmos_object_search successful."
