#ifndef LOCAL_CLOUD_PUBLISHER_H
#define LOCAL_CLOUD_PUBLISHER_H

#include <ros/ros.h>

class LocalCloudPublisher {
public:
    LocalCloudPublisher();
    void run();
private:
    ros::NodeHandle nh_;
    ros::Subscriber pcl_global_sub_;
    ros::Publisher pcl_local_pub_;
    double pub_rate_;

};


#endif
