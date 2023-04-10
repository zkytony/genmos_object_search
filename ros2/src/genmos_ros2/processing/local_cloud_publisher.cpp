#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>

#include <rclcpp/rclcpp.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "local_cloud_publisher.h"
#include "utils/ros2_utils.h"

using std::string;
using sensor_msgs::msg::PointCloud2;
using geometry_msgs::msg::PoseStamped;
using std::vector;
using namespace std::placeholders;

void LocalCloudPublisher::cloudPoseCallback(const PointCloud2::SharedPtr &cloud,
                                            const PoseStamped::SharedPtr &pose_stamped) {
    std::cout << "Received message!" << std::endl;
    double robot_x = pose_stamped->pose.position.x;
    double robot_y = pose_stamped->pose.position.y;
    double robot_z = pose_stamped->pose.position.z;

    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*cloud, pcl_cloud);
    pcl::PointCloud<pcl::PointXYZ> points_in_region;

    Uniform uniform(0.0, 1.0);

    for (pcl::PointXYZ p : pcl_cloud.points) {
        if (uniform.sample() < retain_ratio_) {
            if ((abs(p.x - robot_x) <= this->region_size_x_/2)
                && (abs(p.y - robot_y) <= this->region_size_y_/2)
                && (abs(p.z - robot_z) <= this->region_size_z_/2)) {
                points_in_region.push_back(p);
            }
        }
    }

    PointCloud2 pcl_msg;
    pcl::toROSMsg(points_in_region, pcl_msg);
    pcl_msg.header.frame_id = cloud->header.frame_id;
    pcl_msg.header.stamp = this->now();
    this->pcl_local_pub_->publish(pcl_msg);
    std::cout << "Published points!" << std::endl;
}

LocalCloudPublisher::LocalCloudPublisher(const rclcpp::NodeOptions &options)
  : Node("local_cloud_publisher") {

    this->declare_parameter("region_size_x", 2.0);
    this->declare_parameter("region_size_y", 2.0);
    this->declare_parameter("region_size_z", 1.5);
    this->declare_parameter("global_cloud_topic", "global_points");
    this->declare_parameter("robot_pose_topic", "robot_pose");
    this->declare_parameter("robot_pose_latched", true);
    this->declare_parameter("region_cloud_topic", "region_points");
    this->declare_parameter("retain_ratio", 0.7);

    region_size_x_ = this->get_parameter("region_size_x").get_parameter_value().get<double>();  // meters
    region_size_y_ = this->get_parameter("region_size_y").get_parameter_value().get<double>();  // meters
    region_size_z_ = this->get_parameter("region_size_z").get_parameter_value().get<double>();  // meters

    retain_ratio_ = this->get_parameter("retain_ratio").get_parameter_value().get<double>();  // percentage of points to keep

    string global_cloud_topic = this->get_parameter("global_cloud_topic").get_parameter_value().get<string>();
    string robot_pose_topic   = this->get_parameter("robot_pose_topic").get_parameter_value().get<string>();
    bool robot_pose_latched = this->get_parameter("robot_pose_latched").get_parameter_value().get<bool>();

    pcl_global_sub_ = std::make_shared<message_filters::Subscriber<PointCloud2>>(this, global_cloud_topic);

    // if the robot pose topic is latched on the publisher side, we need to make the subscriber side
    // have the same QoS profile.
    rmw_qos_profile_t qos_profile = rmw_qos_profile_default;
    if (robot_pose_latched) {
        qos_profile = rmw_qos_profile_latch;
    }
    robot_pose_sub_ = std::make_shared<message_filters::Subscriber<PoseStamped>>(this, robot_pose_topic, qos_profile);

    string region_cloud_topic = this->get_parameter("region_cloud_topic").get_parameter_value().get<string>();
    pcl_local_pub_ = this->create_publisher<PointCloud2>(region_cloud_topic, 10);

    sync_ = std::make_shared<CloudPoseSync>(CloudPoseSyncPolicy(10), *pcl_global_sub_, *robot_pose_sub_);
    sync_->registerCallback(&LocalCloudPublisher::cloudPoseCallback, this);
}

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocalCloudPublisher>(rclcpp::NodeOptions()));
    rclcpp::shutdown();
    return 0;
}
