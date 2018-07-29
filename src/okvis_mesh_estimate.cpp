/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Jun 26, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file okvis_node_synchronous.cpp
 * @brief This file includes the synchronous ROS node implementation.

          This node goes through a rosbag in order and waits until all processing is done
          before adding a new message to algorithm

 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <memory>
#include <functional>

#include "sensor_msgs/Imu.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop
#include <okvis/Subscriber.hpp>
#include <okvis/Publisher.hpp>
#include <okvis/RosParametersReader.hpp>
#include <okvis/ThreadedKFVio.hpp>

#include "rosbag/bag.h"
#include "rosbag/chunked_file.h"
#include "rosbag/view.h"

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PolygonMesh.h>

#include <image_transport/image_transport.h>

#include <cv_bridge/cv_bridge.h>

#include <ros_sensor_streams/asl_rgbd_offline_stream.h>

#include <flame/params.h>

#include <flame/flame.h>
#include <flame/utils/image_utils.h>
#include <flame/utils/stats_tracker.h>
#include <flame/utils/load_tracker.h>


#include "mesh_estimate/utils.h"

namespace bfs = boost::filesystem;
namespace fu = flame::utils;
using namespace flame_ros;



// this is just a workbench. most of the stuff here will go into the Frontend class.
int main(int argc, char **argv) {

    ros::init(argc, argv, "okvis_mesh_estimate");

    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = 0; // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
    FLAGS_colorlogtostderr = 1;

    if ( argc != 3) {
        LOG(ERROR) <<
                   "Usage: ./" << argv[0] << " configuration-yaml-file bag-to-read-from";
        return -1;
    }

    okvis::Duration deltaT(0.0);


    // set up the node
    ros::NodeHandle nh("okvis_node");
    ros::NodeHandle pnh_("~");


    flame::Params params_;

    // Target processing rate. Can artificially slow down processing so that ROS can
    // keep up.
    float rate_;

    // Frame ID of the camera in frame camera_world_frame_id.
    std::string camera_frame_id_;

    // Frame id of the world in camera (Right-Down-Forward) coordinates.
    std::string camera_world_frame_id_;

    int subsample_factor_; // Process one out of this many images.
    int poseframe_subsample_factor_; // Create a poseframe every this number of images.
    int resize_factor_;

    // Save truth stats.
    std::string output_dir_;
    bool pass_in_truth_; // Pass truth into processing.


    // Publishes mesh.
    bool publish_mesh_;
    ros::Publisher mesh_pub_;

    // Publishes depthmap and stuff.
    image_transport::ImageTransport it_(nh);
    bool publish_idepthmap_;
    bool publish_depthmap_;
    bool publish_features_;
    image_transport::CameraPublisher idepth_pub_;
    image_transport::CameraPublisher depth_pub_;
    image_transport::CameraPublisher features_pub_;

    // Publish pointcloud.
    bool publish_cloud_;
    ros::Publisher cloud_pub_;

    // Publishes statistics.
    bool publish_stats_;
    ros::Publisher stats_pub_;
    ros::Publisher nodelet_stats_pub_;
    int load_integration_factor_;

    // Publishes debug images.
    image_transport::Publisher debug_wireframe_pub_;
    image_transport::Publisher debug_features_pub_;
    image_transport::Publisher debug_detections_pub_;
    image_transport::Publisher debug_matches_pub_;
    image_transport::Publisher debug_normals_pub_;
    image_transport::Publisher debug_idepthmap_pub_;



    std::string pose_path;
    getParamOrFail(pnh_, "pose_path", &pose_path);

    std::string rgb_path;
    getParamOrFail(pnh_, "rgb_path", &rgb_path);

    std::string depth_path;
    getParamOrFail(pnh_, "depth_path", &depth_path);

    std::string world_frame_str;
    getParamOrFail(pnh_, "world_frame", &world_frame_str);

    ros_sensor_streams::ASLRGBDOfflineStream::WorldFrame world_frame;
    if (world_frame_str == "RDF") {
        world_frame = ros_sensor_streams::ASLRGBDOfflineStream::WorldFrame::RDF;
    } else if (world_frame_str == "FLU") {
        world_frame = ros_sensor_streams::ASLRGBDOfflineStream::WorldFrame::FLU;
    } else if (world_frame_str == "FRD") {
        world_frame = ros_sensor_streams::ASLRGBDOfflineStream::WorldFrame::FRD;
    } else if (world_frame_str == "RFU") {
        world_frame = ros_sensor_streams::ASLRGBDOfflineStream::WorldFrame::RFU;
    } else {
        ROS_ERROR("Unknown world frame!\n");
        return 1;
    }

    /*==================== Input Params ====================*/
    getParamOrFail(pnh_, "input/camera_frame_id", &camera_frame_id_);
    getParamOrFail(pnh_, "input/camera_world_frame_id", &camera_world_frame_id_);
    getParamOrFail(pnh_, "input/subsample_factor", &subsample_factor_);
    getParamOrFail(pnh_, "input/poseframe_subsample_factor",
                   &poseframe_subsample_factor_);

    getParamOrFail(pnh_, "input/resize_factor", &resize_factor_);

    getParamOrFail(pnh_, "input/rate", &rate_);

    /*==================== Output Params ====================*/
    getParamOrFail(pnh_, "output/quiet", &params_.debug_quiet);
    getParamOrFail(pnh_, "output/mesh", &publish_mesh_);
    getParamOrFail(pnh_, "output/idepthmap", &publish_idepthmap_);
    getParamOrFail(pnh_, "output/depthmap", &publish_depthmap_);
    getParamOrFail(pnh_, "output/cloud", &publish_cloud_);
    getParamOrFail(pnh_, "output/features", &publish_features_);
    getParamOrFail(pnh_, "output/stats", &publish_stats_);
    getParamOrFail(pnh_, "output/load_integration_factor",
                   &load_integration_factor_);
    getParamOrFail(pnh_, "output/scene_color_scale", &params_.scene_color_scale);
    getParamOrFail(pnh_, "output/filter_oblique_triangles",
                   &params_.do_oblique_triangle_filter);

    double oblique_normal_thresh;
    getParamOrFail(pnh_, "output/oblique_normal_thresh", &oblique_normal_thresh);
    params_.oblique_normal_thresh = oblique_normal_thresh;

    getParamOrFail(pnh_, "output/oblique_idepth_diff_factor",
                   &params_.oblique_idepth_diff_factor);
    getParamOrFail(pnh_, "output/oblique_idepth_diff_abs",
                   &params_.oblique_idepth_diff_abs);

    getParamOrFail(pnh_, "output/filter_long_edges",
                   &params_.do_edge_length_filter);

    double edge_length_thresh;
    getParamOrFail(pnh_, "output/edge_length_thresh", &edge_length_thresh);
    params_.edge_length_thresh = edge_length_thresh;

    getParamOrFail(pnh_, "output/filter_triangles_by_idepth",
                   &params_.do_idepth_triangle_filter);

    double min_triangle_idepth;
    getParamOrFail(pnh_, "output/min_triangle_idepth", &min_triangle_idepth);
    params_.min_triangle_idepth = min_triangle_idepth;

//    getParamOrFail(pnh_, "output/max_angular_rate", &max_angular_rate_);

    /*==================== Debug Params ====================*/
    getParamOrFail(pnh_, "debug/wireframe", &params_.debug_draw_wireframe);
    getParamOrFail(pnh_, "debug/features", &params_.debug_draw_features);
    getParamOrFail(pnh_, "debug/detections", &params_.debug_draw_detections);
    getParamOrFail(pnh_, "debug/matches", &params_.debug_draw_matches);
    getParamOrFail(pnh_, "debug/normals", &params_.debug_draw_normals);
    getParamOrFail(pnh_, "debug/idepthmap", &params_.debug_draw_idepthmap);
    getParamOrFail(pnh_, "debug/text_overlay", &params_.debug_draw_text_overlay);
    getParamOrFail(pnh_, "debug/flip_images", &params_.debug_flip_images);

    /*==================== Threading Params ====================*/
    getParamOrFail(pnh_, "threading/openmp/num_threads", &params_.omp_num_threads);
    getParamOrFail(pnh_, "threading/openmp/chunk_size", &params_.omp_chunk_size);

    /*==================== Features Params ====================*/
    getParamOrFail(pnh_, "features/do_letterbox", &params_.do_letterbox);
    getParamOrFail(pnh_, "features/detection/min_grad_mag", &params_.min_grad_mag);
    params_.fparams.min_grad_mag = params_.min_grad_mag;

    double min_error;
    getParamOrFail(pnh_, "features/detection/min_error", &min_error);
    params_.min_error = min_error;

    getParamOrFail(pnh_, "features/detection/win_size", &params_.detection_win_size);

    int win_size;
    getParamOrFail(pnh_, "features/tracking/win_size", &win_size);
    params_.zparams.win_size = win_size;
    params_.fparams.win_size = win_size;

    getParamOrFail(pnh_, "features/tracking/max_dropouts", &params_.max_dropouts);

    double epipolar_line_var;
    getParamOrFail(pnh_, "features/tracking/epipolar_line_var",
                   &epipolar_line_var);
    params_.zparams.epipolar_line_var = epipolar_line_var;

    /*==================== Regularizer Params ====================*/
    getParamOrFail(pnh_, "regularization/do_nltgv2", &params_.do_nltgv2);
    getParamOrFail(pnh_, "regularization/nltgv2/adaptive_data_weights",
                   &params_.adaptive_data_weights);
    getParamOrFail(pnh_, "regularization/nltgv2/rescale_data", &params_.rescale_data);
    getParamOrFail(pnh_, "regularization/nltgv2/init_with_prediction",
                   &params_.init_with_prediction);
    getParamOrFail(pnh_, "regularization/nltgv2/idepth_var_max",
                   &params_.idepth_var_max_graph);
    getParamOrFail(pnh_, "regularization/nltgv2/data_factor", &params_.rparams.data_factor);
    getParamOrFail(pnh_, "regularization/nltgv2/step_x", &params_.rparams.step_x);
    getParamOrFail(pnh_, "regularization/nltgv2/step_q", &params_.rparams.step_q);
    getParamOrFail(pnh_, "regularization/nltgv2/theta", &params_.rparams.theta);
    getParamOrFail(pnh_, "regularization/nltgv2/min_height", &params_.min_height);
    getParamOrFail(pnh_, "regularization/nltgv2/max_height", &params_.max_height);
    getParamOrFail(pnh_, "regularization/nltgv2/check_sticky_obstacles",
                   &params_.check_sticky_obstacles);



    // publisher
    okvis::Publisher publisher(nh);

    // read configuration file
    std::string configFilename(argv[1]);

    okvis::RosParametersReader vio_parameters_reader(configFilename);
    okvis::VioParameters parameters;
    vio_parameters_reader.getParameters(parameters);

    okvis::ThreadedKFVio okvis_estimator(parameters, params_);

    okvis_estimator.setFullStateCallback(std::bind(&okvis::Publisher::publishFullStateAsCallback,&publisher,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4));
//  okvis_estimator.setLandmarksCallback(std::bind(&okvis::Publisher::publishLandmarksAsCallback,&publisher,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3));
    okvis_estimator.setStateCallback(std::bind(&okvis::Publisher::publishStateAsCallback,&publisher,std::placeholders::_1,std::placeholders::_2));
    okvis_estimator.setBlocking(true);
    publisher.setParameters(parameters); // pass the specified publishing stuff

    // extract the folder path
    std::string bagname(argv[2]);
    size_t pos = bagname.find_last_of("/");
    std::string path;
    if (pos == std::string::npos)
        path = ".";
    else
        path = bagname.substr(0, pos);

    const unsigned int numCameras = parameters.nCameraSystem.numCameras();

    // setup files to be written
    publisher.setCsvFile(path + "/okvis_estimator_output.csv");
    publisher.setLandmarksCsvFile(path + "/okvis_estimator_landmarks.csv");
    okvis_estimator.setImuCsvFile(path + "/imu0_data.csv");
    for (size_t i = 0; i < numCameras; ++i) {
        std::stringstream num;
        num << i;
        okvis_estimator.setTracksCsvFile(i, path + "/cam" + num.str() + "_tracks.csv");
    }

    // open the bag
    rosbag::Bag bag(argv[2], rosbag::bagmode::Read);
    // views on topics. the slash is needs to be correct, it's ridiculous...
    std::string imu_topic("/imu0");
    rosbag::View view_imu(
            bag,
            rosbag::TopicQuery(imu_topic));
    if (view_imu.size() == 0) {
        LOG(ERROR) << "no imu topic";
        return -1;
    }
    rosbag::View::iterator view_imu_iterator = view_imu.begin();
    LOG(INFO) << "No. IMU messages: " << view_imu.size();

    std::vector<std::shared_ptr<rosbag::View> > view_cams_ptr;
    std::vector<rosbag::View::iterator> view_cam_iterators;
    std::vector<okvis::Time> times;
    okvis::Time latest(0);
    for(size_t i=0; i<numCameras;++i) {
        std::string camera_topic("/cam"+std::to_string(i)+"/image_raw");
        std::shared_ptr<rosbag::View> view_ptr(
                new rosbag::View(
                        bag,
                        rosbag::TopicQuery(camera_topic)));
        if (view_ptr->size() == 0) {
            LOG(ERROR) << "no camera topic";
            return 1;
        }
        view_cams_ptr.push_back(view_ptr);
        view_cam_iterators.push_back(view_ptr->begin());
        sensor_msgs::ImageConstPtr msg1 = view_cam_iterators[i]
                ->instantiate<sensor_msgs::Image>();
        times.push_back(
                okvis::Time(msg1->header.stamp.sec, msg1->header.stamp.nsec));
        if (times.back() > latest)
            latest = times.back();
        LOG(INFO) << "No. cam " << i << " messages: " << view_cams_ptr.back()->size();
    }

    for(size_t i=0; i<numCameras;++i) {
        if ((latest - times[i]).toSec() > 0.01)
            view_cam_iterators[i]++;
    }

    int counter = 0;
    okvis::Time start(0.0);
    while (ros::ok()) {
        ros::spinOnce();
        okvis_estimator.display();

        // check if at the end
        if (view_imu_iterator == view_imu.end()){
            std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
            char k = 0;
            while(k==0 && ros::ok()){
                k = cv::waitKey(1);
                ros::spinOnce();
            }
            return 0;
        }
        for (size_t i = 0; i < numCameras; ++i) {
            if (view_cam_iterators[i] == view_cams_ptr[i]->end()) {
                std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
                char k = 0;
                while(k==0 && ros::ok()){
                    k = cv::waitKey(1);
                    ros::spinOnce();
                }
                return 0;
            }
        }

        // add images
        okvis::Time t;
        for(size_t i=0; i<numCameras;++i) {
            sensor_msgs::ImageConstPtr msg1 = view_cam_iterators[i]
                    ->instantiate<sensor_msgs::Image>();
            cv::Mat filtered(msg1->height, msg1->width, CV_8UC1);
            memcpy(filtered.data, &msg1->data[0], msg1->height * msg1->width);
            t = okvis::Time(msg1->header.stamp.sec, msg1->header.stamp.nsec);
            if (start == okvis::Time(0.0)) {
                start = t;
            }

            // get all IMU measurements till then
            okvis::Time t_imu=start;
            do {
                sensor_msgs::ImuConstPtr msg = view_imu_iterator
                        ->instantiate<sensor_msgs::Imu>();
                Eigen::Vector3d gyr(msg->angular_velocity.x, msg->angular_velocity.y,
                                    msg->angular_velocity.z);
                Eigen::Vector3d acc(msg->linear_acceleration.x,
                                    msg->linear_acceleration.y,
                                    msg->linear_acceleration.z);

                t_imu = okvis::Time(msg->header.stamp.sec, msg->header.stamp.nsec);

                // add the IMU measurement for (blocking) processing
                if (t_imu - start > deltaT)
                    okvis_estimator.addImuMeasurement(t_imu, acc, gyr);

                view_imu_iterator++;
            } while (view_imu_iterator != view_imu.end() && t_imu <= t);

            // add the image to the frontend for (blocking) processing
            if (t - start > deltaT)
                okvis_estimator.addImage(t, i, filtered);

            view_cam_iterators[i]++;
        }
        ++counter;

        // display progress
        if (counter % 20 == 0) {
            std::cout
                    << "\rProgress: "
                    << int(double(counter) / double(view_cams_ptr.back()->size()) * 100)
                    << "%  " ;
        }

    }

    std::cout << std::endl;
    return 0;
}
