#include <okvis/estimate_pipeline.hpp>
#include <map>

#include <glog/logging.h>

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ImuError.hpp>

#include <flame/mesh_estimator.hpp>

namespace okvis {
    // Constructor.
    EstimatePipeline::EstimatePipeline(okvis::VioParameters& parameters, flame::Params mesh_parameter)
            : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
              imu_params_(parameters.imu),
              repropagationNeeded_(false),
              lastAddedImageTimestamp_(okvis::Time(0, 0)),
              estimator_(),
              frontend_(parameters.nCameraSystem.numCameras()),
              parameters_(parameters),
              meshParams(mesh_parameter){
        init();
    }

    EstimatePipeline::~EstimatePipeline() {

    }


    void EstimatePipeline::init() {
        assert(parameters_.nCameraSystem.numCameras() > 0);
        numCameras_ = parameters_.nCameraSystem.numCameras();
        numCameraPairs_ = 1;

        frontend_.setBriskDetectionOctaves(parameters_.optimization.detectionOctaves);
        frontend_.setBriskDetectionThreshold(parameters_.optimization.detectionThreshold);
        frontend_.setBriskDetectionMaximumKeypoints(parameters_.optimization.maxNoKeypoints);

        estimator_.addImu(parameters_.imu);

        // set up windows so things don't crash on Mac OS
        if(parameters_.visualization.displayImages){
//            for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
//                std::stringstream windowname;
//                windowname << "OKVIS camera " << im;
//                cv::namedWindow(windowname.str());
//            }
        }

        displayImages_.resize(numCameras_, cv::Mat());

        /*
         * Mesh estimator
         */

        double width = parameters_.nCameraSystem.cameraGeometry(0)->imageWidth();
        double height = parameters_.nCameraSystem.cameraGeometry(0)->imageHeight();
        Eigen::VectorXd intrinsic;
        parameters_.nCameraSystem.cameraGeometry(0)->getIntrinsics(intrinsic);
        double fx = intrinsic[0];
        double fy = intrinsic[1];
        double cx = intrinsic[2];
        double cy = intrinsic[3];
        double k1 = intrinsic[4];
        double k2 = intrinsic[5];
        double k3 = intrinsic[6];
        double k4 = intrinsic[7];

        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0,0) = fx; K(1,1) = fy; K(0,2) = cx; K(1,2) = cy;
        Eigen::Vector4d distort;
        distort << k1,k2,k3,k4;
        meshEstimatorPtr_
                = std::make_shared<flame::MeshEstimator>(width, height,
                                                         K.cast<float>(), K.inverse().cast<float>(),
                                                         distort.cast<float>(), meshParams);
    }


// Add a new image.
    bool EstimatePipeline::addImage(const okvis::Time & stamp,
                                 const cv::Mat & image0,
                                 const cv::Mat & image1,
                                 const std::vector<cv::KeyPoint> * keypoints,
                                 bool* /*asKeyframe*/) {
        assert(cameraIndex<numCameras_);

        if (lastAddedImageTimestamp_ > stamp
            && fabs((lastAddedImageTimestamp_ - stamp).toSec())
               > parameters_.sensors_information.frameTimestampTolerance) {
            LOG(ERROR)
                    << "Received image from the past. Dropping the image.";
            return false;
        }
        lastAddedImageTimestamp_ = stamp;

        std::shared_ptr<okvis::CameraMeasurement> frame0 = std::make_shared<
                okvis::CameraMeasurement>();
        frame0->measurement.image = image0;
        frame0->timeStamp = stamp;
        frame0->sensorId = 0;
        frame0->measurement.deliversKeypoints = false;

        std::shared_ptr<okvis::CameraMeasurement> frame1 = std::make_shared<
                okvis::CameraMeasurement>();
        frame1->measurement.image = image1;
        frame1->timeStamp = stamp;
        frame1->sensorId = 1;
        frame1->measurement.deliversKeypoints = false;


//        if (blocking_) {
//            cameraMeasurementsReceived_[cameraIndex]->PushBlockingIfFull(frame, 1);
//            return true;
//        } else {
//            cameraMeasurementsReceived_[cameraIndex]->PushNonBlockingDroppingIfFull(
//                    frame, max_camera_input_queue_size);
//            return cameraMeasurementsReceived_[cameraIndex]->Size() == 1;
//        }

        // todo


        cv::imshow("image0", image0);
        cv::imshow("image1", image1);
        cv::waitKey(2);


        return true;
    }


// Add an IMU measurement.
    bool EstimatePipeline::addImuMeasurement(const okvis::Time & stamp,
                                          const Eigen::Vector3d & alpha,
                                          const Eigen::Vector3d & omega) {

        okvis::ImuMeasurement imu_measurement;
        imu_measurement.measurement.accelerometers = alpha;
        imu_measurement.measurement.gyroscopes = omega;
        imu_measurement.timeStamp = stamp;


//        if (blocking_) {
//            imuMeasurementsReceived_.PushBlockingIfFull(imu_measurement, 1);
//            return true;
//        } else {
//            imuMeasurementsReceived_.PushNonBlockingDroppingIfFull(
//                    imu_measurement, maxImuInputQueueSize_);
//            return imuMeasurementsReceived_.Size() == 1;
//        }

        return true;

    }
    // trigger display (needed because OSX won't allow threaded display)
    void EstimatePipeline::display() {
        // draw
        for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
            std::stringstream windowname;
            windowname << "OKVIS camera " << im;
            cv::imshow(windowname.str(), displayImages_[im]);
        }
        cv::waitKey(1);
    }

}