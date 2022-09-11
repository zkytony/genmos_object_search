// referenced points_concat_filter.cpp in autoware_ai_perception
#ifndef POINT_CLOUD_MERGER_H
#define POINT_CLOUD_MERGER_H

#include <string>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>

#include "utils/math_utils.h"

using std::string;
using sensor_msgs::PointCloud2;
using geometry_msgs::PoseStamped;

using namespace message_filters;
using namespace message_filters::sync_policies;

typedef ApproximateTime<PointCloud2, PointCloud2, PointCloud2,
                        PointCloud2, PointCloud2, PointCloud2,
                        PointCloud2, PointCloud2> CloudSyncPolicy;
typedef Synchronizer<CloudSyncPolicy> CloudSync;

class PointCloudMerger {
public:
    PointCloudMerger();

    void run();

    void cloudCallback(const PointCloud2 &cloud1,
                       const PointCloud2 &cloud2,
                       const PointCloud2 &cloud3,
                       const PointCloud2 &cloud4,
                       const PointCloud2 &cloud5,
                       const PointCloud2 &cloud6,
                       const PointCloud2 &cloud7,
                       const PointCloud2 &cloud8);

    ~PointCloudMerger();

private:
    ros::NodeHandle nh_;
    ros::Publisher pcl_merged_pub_;
    message_filters::Subscriber<PointCloud2> *pcl_sub_;
    CloudSync *sync_;

    std::string input_topics_;
    int input_topics_size_;
    std::string output_frame_id_;
};


#endif
