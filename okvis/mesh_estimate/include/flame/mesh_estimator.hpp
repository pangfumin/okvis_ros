#ifndef _MESH_ESTIMATOR_H_
#define _MESH_ESTIMATOR_H_

#include "flame/flame.h"
namespace flame {

    class MeshEstimator {
    public:
        MeshEstimator(int width, int height,
                      const Matrix3f& K, const Matrix3f& Kinv, const Vector4f& distort,
                      const Params& parameters = Params());

        void processFrame(const uint32_t img_id, const double time,
                          const Sophus::SE3f& pose, const cv::Mat& img_gray,
                          const cv::Mat1f& depth);


        std::shared_ptr<flame::Flame> sensor_;
    private:

        // Depth sensor.
        cv::Mat Kcv_, Dcv_;
        flame::Params params_;
        int poseframe_subsample_factor_;
    };
}
#endif