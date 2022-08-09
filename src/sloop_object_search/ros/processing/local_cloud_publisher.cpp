#include <iostream>
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

typedef Synchronizer<ApproximateTime<PointCloud2, PoseStamped>> CloudPoseSync;

void cloudPoseCallback(const PointCloud2 &cloud, const PoseStamped &pose_stamped) {

    double x = pose_stamped.pose.position.x;
    double y = pose_stamped.pose.position.y;
    double z = pose_stamped.pose.position.z;

    std::cout << "RECEIVED MESSAGE!" << std::endl;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "local_cloud_publisher");

    ros::NodeHandle nh(ros::NodeHandle("~"));

    string global_cloud_topic = nh.param<string>("global_cloud_topic", "global_points");
    string robot_pose_topic = nh.param<string>("robot_pose_topic", "robot_pose");

    message_filters::Subscriber<PointCloud2> pcl_global_sub(nh, global_cloud_topic, 10);
    message_filters::Subscriber<PoseStamped> robot_pose_sub(nh, robot_pose_topic, 10);

    CloudPoseSync sync(10, pcl_global_sub, robot_pose_sub);
    sync.registerCallback(cloudPoseCallback);

    ros::spin();
}
