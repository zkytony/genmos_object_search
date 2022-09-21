#ifndef OCCUPANCY_CLOUD_H
#define OCCUPANCY_CLOUD_H

#include <string>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>

using std::string;
using std::vector;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;


class OccupancyToCloud {
public:
  OccupancyToCloud();

  void run();

  void occupancyGridCallback(const nav_msgs::OccupancyGrid &ocg);

private:
  ros::NodeHandle nh_;
  ros::Publisher pcl_pub_;
  ros::Subscriber ocg_sub_;
  PointCloud cloud_;
  double z_;  // z coordinate of published cloud
  string pcl_frame_id_;
};

#endif
