// referenced points_concat_filter.cpp in autoware_ai_perception
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
#include <tf2_ros/transform_listener.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

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
    assert(2 <= input_topics_size_ && input_topics_size_ <= 8);

    PointCloud2 cloud_msgs[8] = {cloud1, cloud2, cloud3, cloud4,
        cloud5, cloud6, cloud7, cloud8};
    pcl::PointCloud<pcl::PointXYZ> pcl_clouds[8];

    try {
        for (size_t i = 0; i < input_topics_size_; i++) {
            PointCloud2 cloud = cloud_msgs[i];
            pcl::fromROSMsg(cloud, pcl_clouds[i]);
            // transforms point cloud using a tf listener; Reference:
            // https://docs.ros.org/en/noetic/api/pcl_ros/html/namespacepcl__ros.html#a3b83738943b1ca42be91e5ecd1b3f8e1
            tf_buffer_.lookupTransform(
                output_frame_id_, cloud_msgs[i].header.frame_id, ros::Time(0), ros::Duration(1.0));
            pcl_ros::transformPointCloud(
                output_frame_id_, ros::Time(0), pcl_clouds[i], cloud_msgs[i].header.frame_id, pcl_clouds[i], tf_buffer_);
        }
    } catch (tf2::TransformException &ex) {
        ROS_WARN("%s",ex.what());
        ros::Duration(1.0).sleep();
        return;
    }

    // merge points
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_merged(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < input_topics_size_; i++) {
        *pcl_cloud_merged += pcl_clouds[i];
    }

    // publsh points
    sensor_msgs::PointCloud2 pcl_merged_msg;
    pcl::toROSMsg(*pcl_cloud_merged, pcl_merged_msg);
    pcl_merged_msg.header.frame_id = output_frame_id_;
    pcl_merged_msg.header.stamp = ros::Time::now();
    pcl_merged_pub_.publish(pcl_merged_msg);
    std::cout << "Published merged cloud!" << std::endl;
}

PointCloudMerger::PointCloudMerger()
    : nh_(ros::NodeHandle("~")), tf_listener_(tf_buffer_) {

    input_topics_ = nh_.param<std::string>("input_topics", std::string("[]"));
    nh_.param("output_frame_id", output_frame_id_, std::string("[]"));

    YAML::Node topics = YAML::Load(input_topics_);
    input_topics_size_ = topics.size();
    std::cout << input_topics_ << std::endl;

    if (input_topics_size_ < 2 || 8 < input_topics_size_) {
        ROS_ERROR("The size of input_topics must be between 2 and 8");
        ros::shutdown();
    }
    for (size_t i = 0; i < 8; ++i) {
        if (i < input_topics_size_) {
            ROS_INFO("%s", ("subscribing to " + topics[i].as<std::string>()).c_str());
            pcl_subs_[i] =
                new message_filters::Subscriber<PointCloud2>(nh_, topics[i].as<std::string>(), 1);
        } else {
            pcl_subs_[i] =
                new message_filters::Subscriber<PointCloud2>(nh_, topics[0].as<std::string>(), 1);
        }
    }
    sync_ = new CloudSync(CloudSyncPolicy(10),
                          *pcl_subs_[0], *pcl_subs_[1], *pcl_subs_[2], *pcl_subs_[3],
                          *pcl_subs_[4], *pcl_subs_[5], *pcl_subs_[6], *pcl_subs_[7]);
    sync_->registerCallback(&PointCloudMerger::cloudCallback, this);
    pcl_merged_pub_ = nh_.advertise<PointCloud2>("/points_merged", 1);
}

void PointCloudMerger::run() {
    ros::spin();
}

PointCloudMerger::~PointCloudMerger() {
    for (size_t i = 0; i < 8; ++i) {
        delete pcl_subs_[i];
    }
    delete sync_;
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "point_cloud_merger");
    PointCloudMerger pclMerg;
    pclMerg.run();
}
