#ifndef LOCAL_CLOUD_PUBLISHER_H
#define LOCAL_CLOUD_PUBLISHER_H

#include <string>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>

using std::string;
using sensor_msgs::PointCloud2;
using geometry_msgs::PoseStamped;

using namespace message_filters;
using namespace message_filters::sync_policies;

typedef ApproximateTime<PointCloud2, PoseStamped> CloudPoseSyncPolicy;
typedef Synchronizer<CloudPoseSyncPolicy> CloudPoseSync;

class LocalCloudPublisher {
public:
    LocalCloudPublisher();

    void run();

    void cloudPoseCallback(const PointCloud2 &cloud, const PoseStamped &pose);

    ~LocalCloudPublisher();

private:
    ros::NodeHandle nh_;
    ros::Publisher pcl_local_pub_;
    message_filters::Subscriber<PointCloud2> *pcl_global_sub_;
    message_filters::Subscriber<PoseStamped> *robot_pose_sub_;
    CloudPoseSync *sync_;
    double pub_rate_;
};

#endif
