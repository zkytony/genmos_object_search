#include <iostream>
#include <string>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include "local_cloud_publisher.h"

using std::string;

void globalCloudCallback(const sensor_msgs::PointCloud2 &msg) {
    std::cout << "RECEIVED MESSAGE!" << std::endl;
}

LocalCloudPublisher::LocalCloudPublisher():
    nh_(ros::NodeHandle("~")) {
    string global_cloud_topic = nh_.param<string>("global_cloud_topic", "global_points");
    std::cout << "global point cloud topic: " << global_cloud_topic << std::endl;

    pub_rate_ = nh_.param<double>("local_cloud_publish_rate", 4.0);

    pcl_global_sub_ = nh_.subscribe(global_cloud_topic, 10, globalCloudCallback);
}

void LocalCloudPublisher::run() {
    ros::Rate loop_rate(this->pub_rate_);
    while (this->nh_.ok()) {
        ros::spinOnce();
        loop_rate.sleep();
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "local_cloud_publisher");
    LocalCloudPublisher locPub = LocalCloudPublisher();
    locPub.run();
}
