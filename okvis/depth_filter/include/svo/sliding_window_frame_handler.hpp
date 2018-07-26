#ifndef SLIDING_WINDOW_FRAME_HANDLER_H
#define SLIDING_WINDOW_FRAME_HANDLER_H

#include <set>
#include <vikit/abstract_camera.h>
#include <svo/frame_handler_base.h>
#include <svo/reprojector.h>
#include <svo/initialization.h>
namespace svo {
    class SlidingWindowFrameHandler : public FrameHandlerBase{
    public:
        SlidingWindowFrameHandler(vk::AbstractCamera* cam0, vk::AbstractCamera* cam1 , int maxKeyframeNum);
        /// Provide an image.
        void addImage(const cv::Mat& img0, const cv::Mat& img1, Sophus::SE3& T_W_C0, Sophus::SE3& T_W_C1,
                      double timestamp, bool isKeyframe);
    private:
        int maxKeyframeNum_;
        vk::AbstractCamera* cam0_;                     //!< Camera model, can be ATAN, Pinhole or Ocam (see vikit).
        vk::AbstractCamera* cam1_;                     //!< Camera model, can be ATAN, Pinhole or Ocam (see vikit).
        Reprojector reprojector0_;                     //!< Projects points from other keyframes into the current frame
        Reprojector reprojector1_;                     //!< Projects points from other keyframes into the current frame
        FramePtr new_frame_;                          //!< Current frame.
        FramePtr last_frame_;                         //!< Last frame, not necessarily a keyframe.
        set<FramePtr> core_kfs_;                      //!< Keyframes in the closer neighbourhood.
        vector< pair<FramePtr,size_t> > overlap_kfs_; //!< All keyframes with overlapping field of view. the paired number specifies how many common mappoints are observed TODO: why vector!?
        initialization::KltHomographyInit klt_homography_init_; //!< Used to estimate pose of the first two keyframes by estimating a homography.
        DepthFilter* depth_filter_;                   //!< Depth estimation algorithm runs in a parallel thread and is used to initialize new 3D points.

        /// Initialize the visual odometry algorithm.
        virtual void initialize();

        /// Processes the first frame and sets it as a keyframe.
        virtual UpdateResult processFirstFrame(Sophus::SE3& T_WC);

        /// Processes all frames after the first frame until a keyframe is selected.
        virtual UpdateResult processSecondFrame(Sophus::SE3& T_WC);

        /// Processes all frames after the first two keyframes.
        virtual UpdateResult processFrame(Sophus::SE3& T_WC, bool isKeyframe);

    };
}


#endif