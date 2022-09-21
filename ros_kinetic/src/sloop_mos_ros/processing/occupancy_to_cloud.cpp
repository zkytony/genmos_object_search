#include <iostream>
#include <vector>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>

#include "occupancy_to_cloud.h"

using std::string;
using std::vector;

OccupancyToCloud::OccupancyToCloud()
  : nh_(ros::NodeHandle("~")) {

  ocg_sub_ = nh_.subscribe("grid_input", 10, &OccupancyToCloud::occupancyGridCallback, this);
  pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cloud_output", 10, true);  // latch

  pcl_rate_ = 4.0;
}

void OccupancyToCloud::occupancyGridCallback(const nav_msgs::OccupancyGrid &msg) {
  std::cout << "HELLO" << std::endl;
}

void OccupancyToCloud::run() {
  ros::spin();
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "occupancy_to_cloud");
    OccupancyToCloud o2c = OccupancyToCloud();
    o2c.run();
}
