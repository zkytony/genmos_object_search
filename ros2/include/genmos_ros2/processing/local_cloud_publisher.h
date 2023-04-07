#ifndef LOCAL_CLOUD_PUBLISHER_H
#define LOCAL_CLOUD_PUBLISHER_H

#include <string>

#include <rclcpp/rclcpp.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include "utils/math_utils.h"

using std::string;
using sensor_msgs::msg::PointCloud2;
using geometry_msgs::msg::PoseStamped;

using namespace message_filters;
using namespace message_filters::sync_policies;

typedef ApproximateTime<PointCloud2, PoseStamped> CloudPoseSyncPolicy;
typedef Synchronizer<CloudPoseSyncPolicy> CloudPoseSync;

class LocalCloudPublisher : public rclcpp::Node {
public:
    LocalCloudPublisher(const rclcpp::NodeOptions &options);

    void cloudPoseCallback(const PointCloud2 &cloud, const PoseStamped &pose);

    ~LocalCloudPublisher();

private:
    rclcpp::Publisher<PointCloud2>::SharedPtr pcl_local_pub_;
    message_filters::Subscriber<PointCloud2> *pcl_global_sub_;
    message_filters::Subscriber<PoseStamped> *robot_pose_sub_;
    CloudPoseSync *sync_;

    double region_size_x_;
    double region_size_y_;
    double region_size_z_;
    double retain_ratio_;
};

#endif
