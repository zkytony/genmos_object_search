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

  ocg_sub_ = nh_.saubscribe("grid_input", 10, &OccupancyToCloud::occupancyGridCallback, this);
  pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cloud_output", 10, true);  // latch
  z_ = nh_.param<double>("points_z", 1.0);  // meters
  z_density_ = nh_.param<double>("density_z", 0.1);  // fills the z
  z_floor_ = nh_.param<double>("floor_z", 0.02);  // height of floor
  pcl_frame_id_ = nh_.param<string>("pcl_frame_id", "map");  // meters
  pcl_updated_ = false;
}

void OccupancyToCloud::occupancyGridCallback(const nav_msgs::OccupancyGrid &ocg_msg) {

  ROS_INFO("received occupancy grid. Origin:");
  std::cout << ocg_msg.info.origin << std::endl;

  double origin_x = ocg_msg.info.origin.position.x;
  double origin_y = ocg_msg.info.origin.position.y;

  PointCloud cloud;
  for (size_t col = 0; col < ocg_msg.info.width; col++) {
    for (size_t row = 0; row < ocg_msg.info.height; row++) {

      double x = origin_x + col * ocg_msg.info.resolution + ocg_msg.info.resolution / 2;
      double y = origin_y + row * ocg_msg.info.resolution + ocg_msg.info.resolution / 2;

      if (ocg_msg.data[row*ocg_msg.info.width + col] > 0) {
        // we have an obstacle
        double z = z_;
        while (z > 0.0) {
          pcl::PointXYZ point(x, y, z);
          cloud.push_back(point);
          z -= z_density_;
        }
      } else if (ocg_msg.data[row*ocg_msg.info.width + col] == 0) {
        // free space. We will add points for the floor
        pcl::PointXYZ point(x, y, z_floor_);
        cloud.push_back(point);
      }
    }
  }
  this->cloud_ = cloud;
  pcl_updated_ = true;
  sensor_msgs::PointCloud2 pcl_msg;
  pcl::toROSMsg(this->cloud_, pcl_msg);
  pcl_msg.header.frame_id = this->pcl_frame_id_;
  this->pcl_pub_.publish(pcl_msg);
  ROS_INFO("published point cloud");
}

void OccupancyToCloud::run() {
  ros::Rate loop_rate(5.0);
  sensor_msgs::PointCloud2 pcl_msg;
  while (this->nh_.ok()) {
    if (pcl_updated_) {
      pcl::toROSMsg(this->cloud_, pcl_msg);
      pcl_updated_ = false;
    }
    pcl_msg.header.frame_id = this->pcl_frame_id_;
    pcl_msg.header.stamp = ros::Time::now();
    if (this->cloud_.size() > 0) {
      this->pcl_pub_.publish(pcl_msg);
    }
    ros::spinOnce();
    loop_rate.sleep();
  }
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "occupancy_to_cloud");
  OccupancyToCloud o2c = OccupancyToCloud();
  o2c.run();
}
