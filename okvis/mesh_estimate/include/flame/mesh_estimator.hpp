#ifndef _MESH_ESTIMATOR_H_
#define _MESH_ESTIMATOR_H_

#include "flame/flame.h"
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/MultiFrame.hpp>
namespace okvis {
    class Estimator;
}




namespace flame {

    class MeshEstimator {
    public:
        MeshEstimator(okvis::Estimator* estimator, int width, int height,
                      const Matrix3f& K0, const Matrix3f& K0inv,
                      const Vector4f& distort0,
                      const Matrix3f& K1, const Matrix3f& K1inv,
                      const Vector4f& distort1,
                      const Params& parameters = Params());

        void processFrame(const double time, int32_t img_id,
                          const okvis::kinematics::Transformation& T_WC0,
                          const cv::Mat& img_gray0,
                          const okvis::kinematics::Transformation& T_WC1,
                          const cv::Mat& img_gray1, bool isKeyframe);


        bool estimateMesh();


        std::shared_ptr<flame::Flame> sensor_;
    private:

        bool updateMesh();
        void detectFeatures(DetectionData& data);
        static void detectFeatures(const Params& params,
                                   const Matrix3f& K,
                                   const Matrix3f& Kinv,
                                   const utils::Frame& fref,
                                   const utils::Frame& fprev,
                                   const Image1f& idepthmap,
                                   const std::vector<Point2f>& curr_feats,
                                   Image1f* error,
                                   std::vector<Point2f>* features,
                                   utils::StatsTracker* stats,
                                   Image3b* debug_img);

        // Update the depth estimates.
        static bool updateFeatureIDepths(const Params& params,
                                         const Matrix3f& K0,
                                         const Matrix3f& K0inv,
                                         const Matrix3f& K1,
                                         const Matrix3f& K1inv,
                                         const okvis::Estimator* estimator,
                                         const utils::Frame& fnew,
                                         const utils::Frame& fnew_right,
                                         const utils::Frame& curr_pf,
                                         std::vector<FeatureWithIDepth>* feats,
                                         utils::StatsTracker* stats,
                                         Image3b* debug_img);


        // Track a single feature in the new image.
        static bool trackFeature(const Params& params,
                                 const Matrix3f& K,
                                 const Matrix3f& Kinv,
                                 const okvis::Estimator* estimator,
                                 const stereo::EpipolarGeometry<float>& epigeo,
                                 const utils::Frame& fnew,
                                 const utils::Frame& curr_pf,
                                 FeatureWithIDepth* feat,
                                 Point2f* flow, float* residual,
                                 Image3b* debug_img);

        static bool trackFeatureRight(const Params& params,
                                      const Matrix3f& K0,
                                      const Matrix3f& K0inv,
                                      const Matrix3f& K1,
                                      const Matrix3f& K1inv,
                                      const okvis::Estimator* estimator,
                                      const stereo::EpipolarGeometry<float>& epigeo,
                                      const utils::Frame& fnew,
                                      const utils::Frame& curr_pf,
                                      FeatureWithIDepth* feat,
                                      cv::Point2f* flow,
                                      float* residual,
                                      Image3b* debug_img);



        /********************* Draw  Functions***************************/
        // Draw feature detections.
        static void drawDetections(const Params& params,
                                   const Image1b& img,
                                   const Image1f& score,
                                   float max_score,
                                   const std::vector<cv::KeyPoint>& kps,
                                   utils::StatsTracker* stats,
                                   Image3b* debug_img);

        utils::StatsTracker stats_;

        // Depth sensor.
        cv::Mat K0cv_, D0cv_;
        cv::Mat K1cv_, D1cv_;
        flame::Params params_;
        int poseframe_subsample_factor_;
        okvis::Estimator* estimator_;

        bool inited_;
        uint32_t num_data_updates_;
        uint32_t num_regularizer_updates_;

        int width_;
        int height_;

        Matrix3f K0_;
        Matrix3f K0inv_;
        Matrix3f K1_;
        Matrix3f K1inv_;


        uint32_t num_imgs_;
        utils::Frame::Ptr fnew_; // New frame.
        utils::Frame::Ptr fnew_right_; // New frame.
        utils::Frame::Ptr fprev_; // Previous frame.

        // Lock when performing an update or accessing internals.
        std::recursive_mutex update_mtx_;

        // PoseFrames.
        FrameIDToFrame pfs_; // Main container for pfs.
        utils::Frame::Ptr curr_pf_; // Pointer to the current poseframe.


        std::vector<FeatureWithIDepth> new_feats_; // Newly detected features.

        Image1f photo_error_; // Photometric error.

        // Raw depth estimates.
        uint32_t feat_count_; // Running count of features. Used to create feature ID.
        std::vector<FeatureWithIDepth> feats_; // Raw features.
        std::vector<FeatureWithIDepth> feats_in_curr_; // Feature projected into current frame.

        // The main optimization graph.
        Graph graph_;
        float graph_scale_; // Scale of input data. IDepths are scaled to have mean 1.
        std::thread graph_thread_; // Thread that optimizes graph.
        std::mutex graph_mtx_; // Protects the graph data.

        // Maps between feature IDs and vertex handles for the current graph.
        FeatureToVtx feat_to_vtx_;
        VtxToFeature vtx_to_feat_;

        utils::Delaunay triangulator_; // Performs Delaunay triangulation.
        std::mutex triangulator_mtx_;  // Locks triangulator.
        std::vector<bool> tri_validity_; // False if triangle is oblique or too big.
        std::vector<Triangle> triangles_curr_; // Local copy of current triangulation.
        std::vector<Edge> edges_curr_; // Local copy of current edges.

        std::vector<Point2f> vtx_; // Positions of regularized depthmesh.
        std::vector<float> vtx_idepths_; // Regularized idepths.
        std::vector<float> vtx_w1_; // Plane parameters.
        std::vector<float> vtx_w2_;
        std::vector<Vector3f> vtx_normals_;

        Image1f idepthmap_; // Dense idepthmap.
        Image1f w1_map_; // Dense plane parameters.
        Image1f w2_map_; // Dense plane parameters.

        // // Debug images.
        Image3b debug_img_detections_;
        Image3b debug_img_wireframe_;
        Image3b debug_img_features_;
        Image3b debug_img_matches_;
        Image3b debug_img_normals_;
        Image3b debug_img_idepthmap_;
    };
}
#endif