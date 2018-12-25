#include <ros/ros.h>

#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/io/pcd_io.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/features/normal_3d.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>


#include <iostream>
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "std_msgs/Int32.h"
#include "velodyne_msgs/VelodynePacket.h"
#include "velodyne_msgs/VelodyneScan.h"
#include "velodyne_pointcloud/rawdata.h"
#include "pcl_conversions/pcl_conversions.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/String.h"
#include "velodyne_pointcloud/point_types.h"
#include "velodyne_pointcloud/calibration.h"
#include "velodyne_pointcloud/pointcloudXYZIR.h"
#include "tf2_msgs/TFMessage.h"
#include "nav_msgs/Odometry.h"
#include <tf/transform_broadcaster.h>
#include <Eigen/Geometry>
#include "tf_conversions/tf_eigen.h"
#include <tf/transform_datatypes.h>
#include <ros/ros.h>

#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

class Odom_class
{
private:
    ros::NodeHandle n_;
    ros::Publisher odom_pub_;
    ros::Subscriber sub_ ;

public:

    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud;
    //pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Matrix4f  pairTransform;
    Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity () ;




    Eigen::Matrix3f rotation_mat  , rotation_pair_mat;
    ros::Time current_time, last_time;

    Odom_class()
    {
        //Topic you want to publish
        odom_pub_ = n_.advertise<nav_msgs::Odometry>("odom", 1);

        //Topic you want to subscribe
        sub_ = n_.subscribe("velodyne_points", 1, &Odom_class::callback, this);
        /*
        //t = 138.371
        GlobalTransform(0,0) = -0.9455184 ;
        GlobalTransform(0,1) = 0.3255688 ;
        GlobalTransform(1,0) = -0.3255688;
        GlobalTransform(1,1) = -0.9455184;

        GlobalTransform(0,3) = -1.43/3;
        GlobalTransform(1,3) = +0.148/3;
        */

        /*
        //t = 327.365
        GlobalTransform(0,0) = 0.6946586;
        GlobalTransform(0,1) = 0.7193395 ;
        GlobalTransform(1,0) = -0.7193395;
        GlobalTransform(1,1) = 0.6946586;

        GlobalTransform(0,3) = -4.967/3;
        GlobalTransform(1,3) = +3.94/3;
        */



    }

    void callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input)
    {

        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*input,pcl_pc2);
        pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2(pcl_pc2,*new_cloud);

        //pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
        if (prev_cloud)
        {


            pairAlign(prev_cloud, new_cloud, /*temp , */ pairTransform, true);



            /*
            Eigen::Quaterniond rotation_temp_quat(rotation_temp_mat);
            rotation_temp_quat.x() = 0;
            rotation_temp_quat.y() = 0 ;

            std::cout << "rotation_temp_quat.z()" << rotation_temp_quat.z() << std::endl;

            Eigen::AngleAxisd rotation_temp_angle_axis(rotation_temp_mat);
            Eigen::Vector3d rot_axis = rotation_temp_angle_axis.axis();
            Eigen::AngleAxisd::Scalar rot_angle  = rotation_temp_angle_axis.angle();
            */
            tf::Matrix3x3 rotation_pair_tf;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    rotation_pair_tf[i][j] = pairTransform(i, j);
                }
            }

            tfScalar yaw,pitch,roll;

            rotation_pair_tf.getEulerYPR( yaw , pitch, roll);
            pitch = 0 ;
            roll = 0 ;
            yaw  = - yaw * 6 ;
            rotation_pair_tf.setRPY(roll , pitch ,yaw) ;

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    pairTransform(i, j) = rotation_pair_tf[i][j] ;
                }
            }






            GlobalTransform = GlobalTransform * pairTransform;



            tf::TransformBroadcaster odom_broadcaster;
            double x, y, orient_w, orient_z;

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    rotation_mat(i, j) = GlobalTransform(i, j);
                }
            }


            tf::Matrix3x3 rotation_mat_tf;
            //tf::matrixEigenToTF(rotation_mat, rotation_mat_tf);

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    rotation_mat_tf[i][j] = rotation_mat(i, j);
                }
            }

            tf::Quaternion tf_q;
            rotation_mat_tf.getRotation(tf_q);
            geometry_msgs::Quaternion msg_q;
            quaternionTFToMsg(tf_q, msg_q);

            /*
            msg_q.x = 0 ;
            msg_q.y = 0 ;
            msg_q.z = msg_q.z * 5;
            */
            //first, we'll publish the transform over tf
            geometry_msgs::TransformStamped odom_trans;

            current_time = ros::Time::now();
            odom_trans.header.stamp = current_time;
            odom_trans.header.frame_id = "odom";
            odom_trans.child_frame_id = "base_link";

            odom_trans.transform.translation.x = GlobalTransform(0, 3);
            odom_trans.transform.translation.y = GlobalTransform(1, 3);
            odom_trans.transform.translation.z = 0.0;
            odom_trans.transform.rotation = msg_q;

            //send the transform
            //odom_broadcaster.sendTransform(odom_trans);

            //next, we'll publish the odometry message over ROS
            nav_msgs::Odometry odom;
            odom.header.stamp = current_time;
            odom.header.frame_id = "odom";

            std::cout << "Rototion--" << msg_q  << std::endl;
            //set the position
            odom.pose.pose.position.x = -GlobalTransform(0, 3) * 3;
            odom.pose.pose.position.y = -GlobalTransform(1, 3);
            odom.pose.pose.position.z = 0.0;
            odom.pose.pose.orientation = msg_q;

            //set the velocity
            odom.child_frame_id = "base_link";
            odom.twist.twist.linear.x = 0;
            odom.twist.twist.linear.y = 0;
            odom.twist.twist.angular.z = 0;

            //publish the message
            odom_pub_.publish(odom);
            std::cout << "............." << std::endl;
        }
        prev_cloud = new_cloud;
        //.... do something with the input and generate the output...

    }

    void pairAlign (const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src, const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt, /* pcl::PointCloud<pcl::PointXYZ>::Ptr output, */ Eigen::Matrix4f &final_transform, bool downsample = false)
    {
        //
        // Downsample for consistency and speed
        // \note enable this for large datasets
        pcl::PointCloud<pcl::PointXYZ>::Ptr src (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr tgt (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> grid;
        if (downsample)
        {
            grid.setLeafSize (3, 3, 3);
            grid.setInputCloud (cloud_src);
            grid.filter (*src);

            grid.setInputCloud (cloud_tgt);
            grid.filter (*tgt);
        }
        else
        {
            src = cloud_src;
            tgt = cloud_tgt;
        }

        // Align
        pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> reg;
        reg.setTransformationEpsilon (1e-8);
        // Set the maximum distance between two correspondences (src<->tgt) to 10cm
        // Note: adjust this based on the size of your datasets
        reg.setMaxCorrespondenceDistance (0.3);
        reg.setMaximumIterations (40);
        // Set the point representation
        //reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));

        reg.setInputSource (src);
        reg.setInputTarget (tgt);



        //
        // Run the same optimization in a loop and visualize the results
        Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
        pcl::PointCloud<pcl::PointXYZ>::Ptr reg_result = src;


        /*
        for (int i = 0; i < 2; ++i)
        {
            //PCL_INFO ("Iteration Nr. %d.\n", i);

            // save cloud for visualization purpose
            src = reg_result;

            // Estimate
            reg.setInputSource (src);
            reg.align (*reg_result);

            //accumulate transformation between each Iteration
            Ti = reg.getFinalTransformation () * Ti;

            //if the difference between this transformation and the previous one
            //is smaller than the threshold, refine the process by reducing
            //the maximal correspondence distance
            if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
            {
                reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);
                std::cout << "condition" <<std::endl;
            }
            prev = reg.getLastIncrementalTransformation ();

            // visualize current state
            //showCloudsRight(points_with_normals_tgt, points_with_normals_src);
        }


        //
        // Get the transformation from target to source
        targetToSource = Ti.inverse();

        //
        // Transform target back in source frame
        //pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);

        //add the source to the transformed target
        // *output += *cloud_src;
        final_transform = targetToSource;


        */
        std::cout << "align start" << std::endl;

        reg.align (*reg_result);

        std::cout << "align stop" << std::endl;

        final_transform = reg.getFinalTransformation ();

        std::cout << final_transform << std::endl;


    }

};//End of class Odom_class

int main(int argc, char **argv)
{
    //Initiate ROS
    ros::init(argc, argv, "odom_node");

    //Create an object of class Odom_subandpub that will take care of everything
    Odom_class odomObject;

    ros::spin();

    return 0;
}