# Copyright [2023] Boston Dynamics AI Institute, Inc.
# Echo styles
export YELLOW='\033[1;33m'
export RED='\033[0;31m'
export BROWN='\033[0;33m'
export LRED='\033[1;31m'
export LBLUE='\033[1;34m'
export LGREEN='\033[1;32m'
export LGRAY='\033[0;37m'
export NC='\033[0m' # No Color
export BOLD=$(tput bold)
export NORMAL=$(tput sgr0)

function color_echo
{
    echo -e "${1}${2}${NC}"
}

function echo_warning
{
    color_echo "$YELLOW" "$1"
}

function echo_error
{
    color_echo "$RED" "$1"
}

function echo_info
{
    color_echo "$LBLUE" "$1"
}

function echo_success
{
    color_echo "$GREEN" "$1"
}

function confirm
{
    read -p "$1 [y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
	true && return
    fi
    false
}

# https://unix.stackexchange.com/questions/70615/bash-script-echo-output-in-box
function box_out()
{
    local s=("$@") b w
    for l in "${s[@]}"; do
        ((w<${#l})) && { b="$l"; w="${#l}"; }
    done
    tput setaf 3
    echo " -${b//?/-}-"
    for l in "${s[@]}"; do
        printf '| %s%*s%s |\n' "$(tput setaf 4)" "-$w" "$l" "$(tput setaf 3)"
    done
    echo " -${b//?/-}-"
    tput sgr 0
}

function get_major_version {
    # getting ubuntu version (from my dotfiles)
    # $1: a string of form xx.yy where xx is major version,
    #     and yy is minor version.
    v="$(echo ${1} | sed -e 's/\.[0-9]*//')"
    return $(expr $v + 0)
}

function get_minor_version {
    # $1: a string of form xx.yy where xx is major version,
    #     and yy is minor version.
    v="$(echo ${1} | sed -e 's/[0-9]*\.//')"
    return $(expr $v + 0)
}

function ubuntu_version {
    version="$(lsb_release -r | sed -e 's/[\s\t]*Release:[\s\t]*//')"
    echo "$version"
}

function ubuntu_version_greater_than {
    version=$(ubuntu_version)
    get_major_version $version
    major=$?
    get_minor_version $version
    minor=$?
    get_major_version $1
    given_major=$?
    get_minor_version $1
    given_minor=$?

    (( $major > $given_major )) || { (( $major == $given_major )) && (( $minor > $given_minor )); }
}

function ubuntu_version_less_than {
    version=$(ubuntu_version)
    get_major_version $version
    major=$?
    get_minor_version $version
    minor=$?
    get_major_version $1
    given_major=$?
    get_minor_version $1
    given_minor=$?

    (( $major < $given_major )) || { (( $major == $given_major )) && (( $minor < $given_minor )); }
}

function ubuntu_version_equal {
    if ! ubuntu_version_less_than $1; then
	if ! ubuntu_version_greater_than $1; then
	    true && return
	fi
    fi
    false
}


function useros() {
    if ubuntu_version_equal 20.04; then
        source /opt/ros/noetic/setup.bash
        true && return
    elif ubuntu_version_equal 16.04; then
        source /opt/ros/kinetic/setup.bash
        true && return
    else
        echo -e "No suitable ROS version installed"
        false
    fi
}


function useros2() {
    # sources the ROS2 setup.bash; nothing bdai-specific
    if [ -f /.dockerenv ]; then
        if ubuntu_version_equal 22.04; then
            source /opt/ros/humble/setup.bash
            true && return
        else
            echo -e "No suitable ROS version installed"
            false
        fi
    else
        color_echo "$RED" "Command only works inside docker container"
        false
    fi
}


function rosactions() {
    # reference: https://answers.ros.org/question/222748/list-action-servers/?answer=222759#post-id-222759
    rostopic list | grep -o -P '^.*(?=/feedback)'
}


function check_exists_and_update_submodule {
    if [ -d $1 ]; then
        git submodule update --init --recursive $1
    fi
}

function update_git_submodules {
    # update submodules (clone necessary stuff)
    if confirm "Update git submodules? NOTE: YOU MAY LOSE PROGRESS IF YOUR COMMIT POINTER IS BEHIND SUBMODULE'S LATEST COMMIT."; then
        git submodule update --init --recursive
    fi
}


function get_rlab_interface {
    (echo -e "import netifaces as ni;" ;
     echo -e "for intf in ni.interfaces():" ;
     echo -e "    addrs = ni.ifaddresses(intf);" ;
     echo -e "    if ni.AF_INET not in addrs:" ;
     echo -e "        continue" ;
     echo -e "    ip = addrs[ni.AF_INET][0]['addr'];" ;
     echo -e "    if ip.startswith('138.16.161'):" ;
     echo -e "        print(intf)" ) | python
}

function get_rlab_ip {
    (echo -e "import netifaces as ni;" ;
     echo -e "for intf in ni.interfaces():" ;
     echo -e "    addrs = ni.ifaddresses(intf);" ;
     echo -e "    if ni.AF_INET not in addrs:" ;
     echo -e "        continue" ;
     echo -e "    ip = addrs[ni.AF_INET][0]['addr'];" ;
     echo -e "    if ip.startswith('138.16.161'):" ;
     echo -e "        print(ip)" ) | python
}


# Returns true if this is the first time
# we build the ROS packages related to a robot
function first_time_build
{
    if [ ! -e "$1/src/.DONE_SETUP" ]; then
        # has not successfully setup
        true && return
    else
        false
    fi
}


function build_ros_ws
{
    cd $1
    if catkin_make; then
        echo "$1 SETUP DONE." >> src/.DONE_SETUP
    else
        rm src/.DONE_SETUP
    fi
    cd ..
}

function split_by {
    # Convenient function
    # usage: split_by <string> <separator>
    # example: split_by 'a--b'  '--'
    # The output will be stored in an array called $substrings
    # The size of the array is stored in $len. So, in the
    # above example, $len=2, $substrings[0]=a and $substrings[1]=b
    # Reference: https://unix.stackexchange.com/a/378549/193800
    string=$1
    separator=$2

    tmp=${string//"$separator"/$'\2'}
    IFS=$'\2' read -a arr <<< "$tmp"
    len=0
    for substr in "${arr[@]}" ; do
        substrings[$len]=$substr
        ((len++))
    done
}

function match_var_arg {
    # usage: match_single_var_arg <arg>
    # If <arg> is of format --variable=value
    # return true. Otherwise, return false.
    regex='--[a-zA-Z0-9\-]+=(.*)'
    if [[ $1 =~ $regex ]]; then
        true && return
    else
        false
    fi
}

function parse_var_arg {
    # usage: parse_single_var_arg <arg>
    # If <arg> is of format --variable=value
    # then parse and set the variables $var_name
    # and $var_value respectively. Otherwise,
    # return false.
    if match_var_arg $1; then
        split_by $1 '--'
        split_by ${substrings[1]} '='
        var_name=${substrings[0]}
        var_value=${substrings[1]}
        true && return
    else
        false
    fi
}


function is_flag {
    # usage: is_option <arg>
    # returns true if <arg> is of format -x or --x
    # where x could be a single character or multiple characters.
    # Note that --x=y will not be accepted because it is a
    # variable not a flag.
    if match_var_arg $1; then
        false
    else
        if [[ $1 == -* ]]; then
            true && return
        fi
    fi
}


function ping_success {
    # usage: ping_success <IP>
    # Tries to ping the given IP address (first argument)
    # returns true if success
    res=$(timeout 0.5 ping -c1 $1)
    # If successful, res will not be empty
    if [ -n "$res" ]; then
        true && return
    else
        false
    fi
}

function in_venv {
    # returns true if you are in virtualenv.
    if [[ "$VIRTUAL_ENV" != "" ]]
    then
        true && return
    else
        false
    fi
}


function python_package_installed {
    python -c "import $1; assert $1.__file__ is not None" 2> /dev/null
    if [ $? -eq 0 ]; then
        true && return
    else
        false
    fi
}
