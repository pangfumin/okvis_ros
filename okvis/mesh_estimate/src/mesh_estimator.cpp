#include "flame/mesh_estimator.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

namespace flame {
    MeshEstimator::MeshEstimator(int width, int height,
                                 const Matrix3f& K, const Matrix3f& Kinv,
                                 const Vector4f& distort,
                                 const Params& parameters):
            params_(parameters),
            poseframe_subsample_factor_(6) {

        cv::eigen2cv(K, Kcv_);
        cv::eigen2cv(distort, Dcv_);

        sensor_ = std::make_shared<flame::Flame>(width,
                                                 height,
                                                 K,
                                                 Kinv,
                                                 params_);

    }

    void MeshEstimator::processFrame(const uint32_t img_id, const double time,
                      const Sophus::SE3f& pose, const cv::Mat& img_gray,
                      const cv::Mat1f& depth) {


//        cv::Mat Kcv;
//        eigen2cv(K_, Kcv);
//        cv::undistort(rgb_raw, *rgb, Kcv, cinfo_.D);
        /*==================== Process image ====================*/
        // Convert to grayscale.


        bool is_poseframe = (img_id % poseframe_subsample_factor_) == 0;
        bool update_success = false;

            update_success = sensor_->update(time, img_id, pose, img_gray,
                                             is_poseframe);
        if (!update_success) {
            //ROS_WARN("FlameOffline: Unsuccessful update.\n");
            return;
        }

        // todo : add publish and result into buffer

    }

}