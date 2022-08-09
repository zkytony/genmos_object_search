#include <iostream>
#include <string>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/PointCloud2.h>

#include "local_cloud_publisher.h"

using std::string;
using sensor_msgs::PointCloud2;
using geometry_msgs::PoseStamped;

void LocalCloudPublisher::cloudPoseCallback(const PointCloud2 &cloud, const PoseStamped &pose) {
    std::cout << "RECEIVED MESSAGE!" << std::endl;
}

LocalCloudPublisher::LocalCloudPublisher()
    : nh_(ros::NodeHandle("~")) {

    pub_rate_ = nh_.param<double>("local_cloud_publish_rate", 4.0);
    string global_cloud_topic = nh_.param<string>("global_cloud_topic", "global_points");
    string robot_pose_topic = nh_.param<string>("robot_pose_topic", "robot_pose");

    pcl_global_sub_ = new message_filters::Subscriber<PointCloud2>(nh_, global_cloud_topic, 10);
    robot_pose_sub_ = new message_filters::Subscriber<PoseStamped>(nh_, robot_pose_topic, 10);

    sync_ = new CloudPoseSync(CloudPoseSyncPolicy(10), *pcl_global_sub_, *robot_pose_sub_);
    sync_->registerCallback(&LocalCloudPublisher::cloudPoseCallback, this);
}

void LocalCloudPublisher::run() {
    ros::Rate loop_rate(this->pub_rate_);
    while (this->nh_.ok()) {
        ros::spinOnce();
        loop_rate.sleep();
    }
}

LocalCloudPublisher::~LocalCloudPublisher() {
    delete pcl_global_sub_;
    delete robot_pose_sub_;
    delete sync_;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "local_cloud_publisher");
    LocalCloudPublisher locPub = LocalCloudPublisher();
    locPub.run();
}
