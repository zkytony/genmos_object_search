#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <yaml-cpp/yaml.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "point_cloud_merger.h"

void PointCloudMerger::cloudCallback(const PointCloud2 &cloud1,
                                     const PointCloud2 &cloud2,
                                     const PointCloud2 &cloud3,
                                     const PointCloud2 &cloud4,
                                     const PointCloud2 &cloud5,
                                     const PointCloud2 &cloud6,
                                     const PointCloud2 &cloud7,
                                     const PointCloud2 &cloud8) {
  std::cout << "Received messages!" << std::endl;
}


PointCloudMerger::PointCloudMerger()
    : nh_(ros::NodeHandle("~")) {

    nh_.param("input_topics", input_topics_, std::string("[]"));
    nh_.param("output_frame_id", output_frame_id_, std::string("[]"));

    YAML::Node topics = YAML::Load(input_topics_);
    input_topics_size_ = topics.size();

    // string global_cloud_topic = nh_.param<string>("global_cloud_topic", "global_points");
    // string robot_pose_topic = nh_.param<string>("robot_pose_topic", "robot_pose");
    // pcl_global_sub_ = new message_filters::Subscriber<PointCloud2>(nh_, global_cloud_topic, 10);
    // robot_pose_sub_ = new message_filters::Subscriber<PoseStamped>(nh_, robot_pose_topic, 10);

    // string region_cloud_topic = nh_.param<string>("region_cloud_topic", "region_points");
    // pcl_local_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(region_cloud_topic, 10, true);

    // sync_ = new CloudPoseSync(CloudPoseSyncPolicy(10), *pcl_global_sub_, *robot_pose_sub_);
    // sync_->registerCallback(&PointCloudMerger::cloudPoseCallback, this);
}

void PointCloudMerger::run() {
    ros::spin();
}

PointCloudMerger::~PointCloudMerger() {
    delete pcl_sub_;
    delete sync_;
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "point_cloud_merger");
    PointCloudMerger pclMerg = PointCloudMerger();
    pclMerg.run();
}
