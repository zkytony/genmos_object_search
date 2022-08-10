#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "local_cloud_publisher.h"

using std::string;
using sensor_msgs::PointCloud2;
using geometry_msgs::PoseStamped;
using std::vector;

void LocalCloudPublisher::cloudPoseCallback(const PointCloud2 &cloud, const PoseStamped &pose_stamped) {
    std::cout << "Received message!" << std::endl;
    double robot_x = pose_stamped.pose.position.x;
    double robot_y = pose_stamped.pose.position.y;
    double robot_z = pose_stamped.pose.position.z;

    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(cloud, pcl_cloud);
    pcl::PointCloud<pcl::PointXYZ> points_in_region;

    for (pcl::PointXYZ p : pcl_cloud.points) {

        if ((abs(p.x - robot_x) <= this->region_size_x_/2)
            && (abs(p.y - robot_y) <= this->region_size_y_/2)
            && (abs(p.z - robot_z) <= this->region_size_z_/2)) {
            points_in_region.push_back(p);
        }
    }

    sensor_msgs::PointCloud2 pcl_msg;
    pcl::toROSMsg(points_in_region, pcl_msg);
    pcl_msg.header.frame_id = cloud.header.frame_id;
    this->pcl_local_pub_.publish(pcl_msg);
    std::cout << "Published points!" << std::endl;
}

LocalCloudPublisher::LocalCloudPublisher()
    : nh_(ros::NodeHandle("~")) {

    region_size_x_ = nh_.param<double>("region_size_x", 2.0);  // meters
    region_size_y_ = nh_.param<double>("region_size_y", 2.0);  // meters
    region_size_z_ = nh_.param<double>("region_size_z", 1.5);  // meters

    string global_cloud_topic = nh_.param<string>("global_cloud_topic", "global_points");
    string robot_pose_topic = nh_.param<string>("robot_pose_topic", "robot_pose");
    pcl_global_sub_ = new message_filters::Subscriber<PointCloud2>(nh_, global_cloud_topic, 10);
    robot_pose_sub_ = new message_filters::Subscriber<PoseStamped>(nh_, robot_pose_topic, 10);

    string region_cloud_topic = nh_.param<string>("region_cloud_topic", "region_points");
    pcl_local_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(region_cloud_topic, 10, true);

    sync_ = new CloudPoseSync(CloudPoseSyncPolicy(10), *pcl_global_sub_, *robot_pose_sub_);
    sync_->registerCallback(&LocalCloudPublisher::cloudPoseCallback, this);
}

void LocalCloudPublisher::run() {
    ros::spin();
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
