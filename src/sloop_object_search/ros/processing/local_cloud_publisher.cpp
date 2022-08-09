#include <iostream>
#include <string>

#include <ros/ros.h>

#include "local_cloud_publisher.h"

using std::string;

LocalCloudPublisher::LocalCloudPublisher():
    nh_(ros::NodeHandle("~")){
    string global_cloud_topic = nh_.param<string>("global_cloud_topic", "global_points");
    std::cout << global_cloud_topic << std::endl;
}

void LocalCloudPublisher::run() {}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "local_cloud_publisher");
    LocalCloudPublisher locPub = LocalCloudPublisher();
    locPub.run();
}
