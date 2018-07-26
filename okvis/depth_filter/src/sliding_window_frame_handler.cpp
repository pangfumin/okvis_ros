#include <svo/sliding_window_frame_handler.hpp>
#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/pose_optimizer.h>
#include <svo/sparse_img_align.h>
#include <vikit/performance_monitor.h>
#include <svo/depth_filter.h>
namespace svo {
    //SlidingWindowFrameHandler(vk::AbstractCamera* cam)
    SlidingWindowFrameHandler::SlidingWindowFrameHandler(vk::AbstractCamera* cam0,
                                                         vk::AbstractCamera* cam1, int maxKeyframeNum) :
            FrameHandlerBase(),
            maxKeyframeNum_(maxKeyframeNum),
            cam0_(cam0),
            cam1_(cam1),
            reprojector0_(cam0_, map_),
            reprojector1_(cam1_, map_),
            depth_filter_(NULL)
    {
        initialize();
    }

    void SlidingWindowFrameHandler::initialize()
    {
        feature_detection::DetectorPtr feature_detector(
                new feature_detection::FastDetector(
                        cam0_->width(), cam0_->height(), Config::gridSize(), Config::nPyrLevels()));
        DepthFilter::callback_t depth_filter_cb = boost::bind(
                &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
        depth_filter_ = new DepthFilter(feature_detector, depth_filter_cb);
        depth_filter_->startThread();
    }

    void SlidingWindowFrameHandler::addImage(const cv::Mat& img0, const cv::Mat& img1,
                                            Sophus::SE3& T_W_C0, Sophus::SE3& T_W_C1,
                                             double timestamp, bool isKeyframe) {
//        if(!startFrameProcessingCommon(timestamp))
//            return;
//
//        // some cleanup from last iteration, can't do before because of visualization
//        core_kfs_.clear();
//        overlap_kfs_.clear();
//
//        // create new frame
//        SVO_START_TIMER("pyramid_creation");
//        new_frame_.reset(new Frame(cam0_, img0.clone(), timestamp));
//        SVO_STOP_TIMER("pyramid_creation");

        // process frame
//        UpdateResult res = RESULT_FAILURE;
//        if(stage_ == STAGE_DEFAULT_FRAME)
//            res = processFrame(T_W_C, isKeyframe);
//        else if(stage_ == STAGE_SECOND_FRAME)
//            res = processSecondFrame(T_W_C);
//        else if(stage_ == STAGE_FIRST_FRAME)
//            res = processFirstFrame(T_W_C);

//        // set last frame
//        last_frame_ = new_frame_;
//        new_frame_.reset();
//        // finish processing
//        finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());

//        cv::imshow( "image0", img0);
//        cv::waitKey(2);

    }


    SlidingWindowFrameHandler::UpdateResult
    SlidingWindowFrameHandler::processFirstFrame(Sophus::SE3& T_WC)
    {
        new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
        if(klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
            return RESULT_NO_KEYFRAME;
        new_frame_->setKeyframe();
        map_.addKeyframe(new_frame_);
        stage_ = STAGE_SECOND_FRAME;
        SVO_INFO_STREAM("Init: Selected first frame.");
        return RESULT_IS_KEYFRAME;
    }

    SlidingWindowFrameHandler::UpdateResult
    SlidingWindowFrameHandler::processSecondFrame(Sophus::SE3& T_WC)
    {
        initialization::InitResult res = klt_homography_init_.addSecondFrame(new_frame_);
        if(res == initialization::FAILURE)
            return RESULT_FAILURE;
        else if(res == initialization::NO_KEYFRAME)
            return RESULT_NO_KEYFRAME;


        new_frame_->setKeyframe();
        double depth_mean, depth_min;
        frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
        depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

        // add frame to map
        map_.addKeyframe(new_frame_);
        stage_ = STAGE_DEFAULT_FRAME;
        klt_homography_init_.reset();
        SVO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");
        return RESULT_IS_KEYFRAME;
    }

    SlidingWindowFrameHandler::UpdateResult
    SlidingWindowFrameHandler::processFrame(Sophus::SE3& T_WC, bool isKeyframe)
    {
//        // Set initial pose TODO use prior
//        new_frame_->T_f_w_ = last_frame_->T_f_w_;
//
//        // sparse image align
//        SVO_START_TIMER("sparse_img_align");
//        SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
//                                 30, SparseImgAlign::GaussNewton, false, false);
//        size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
//        SVO_STOP_TIMER("sparse_img_align");
//        SVO_LOG(img_align_n_tracked);
//        SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);
//
//        // map reprojection & feature alignment
//        SVO_START_TIMER("reproject");
//        reprojector_.reprojectMap(new_frame_, overlap_kfs_);
//        SVO_STOP_TIMER("reproject");
//        const size_t repr_n_new_references = reprojector_.n_matches_;
//        const size_t repr_n_mps = reprojector_.n_trials_;
//        SVO_LOG2(repr_n_mps, repr_n_new_references);
//        SVO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references);
//        if(repr_n_new_references < Config::qualityMinFts())
//        {
//            SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
//            new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
//            tracking_quality_ = TRACKING_INSUFFICIENT;
//            return RESULT_FAILURE;
//        }
//
//        // pose optimization
//        SVO_START_TIMER("pose_optimizer");
//        size_t sfba_n_edges_final;
//        double sfba_thresh, sfba_error_init, sfba_error_final;
//        pose_optimizer::optimizeGaussNewton(
//                Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
//                new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
//        SVO_STOP_TIMER("pose_optimizer");
//        SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
//        SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
//        SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final);
//        if(sfba_n_edges_final < 20)
//            return RESULT_FAILURE;
//
//        // structure optimization
//        SVO_START_TIMER("point_optimizer");
//        optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
//        SVO_STOP_TIMER("point_optimizer");
//
//        // select keyframe
//        core_kfs_.insert(new_frame_);
//        setTrackingQuality(sfba_n_edges_final);
//        if(tracking_quality_ == TRACKING_INSUFFICIENT)
//        {
//            new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
//            return RESULT_FAILURE;
//        }
//        double depth_mean, depth_min;
//        frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
//        if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
//        {
//            depth_filter_->addFrame(new_frame_);
//            return RESULT_NO_KEYFRAME;
//        }
//        new_frame_->setKeyframe();
//        SVO_DEBUG_STREAM("New keyframe selected.");
//
//        // new keyframe selected
//        for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
//            if((*it)->point != NULL)
//                (*it)->point->addFrameRef(*it);
//        map_.point_candidates_.addCandidatePointToFrame(new_frame_);
//
//        // optional bundle adjustment
//#ifdef USE_BUNDLE_ADJUSTMENT
//        if(Config::lobaNumIter() > 0)
//  {
//    SVO_START_TIMER("local_ba");
//    setCoreKfs(Config::coreNKfs());
//    size_t loba_n_erredges_init, loba_n_erredges_fin;
//    double loba_err_init, loba_err_fin;
//    ba::localBA(new_frame_.get(), &core_kfs_, &map_,
//                loba_n_erredges_init, loba_n_erredges_fin,
//                loba_err_init, loba_err_fin);
//    SVO_STOP_TIMER("local_ba");
//    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
//    SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
//                     "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
//  }
//#endif
//
//        // init new depth-filters
//        depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);
//
//        // if limited number of keyframes, remove the one furthest apart
//        if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
//        {
//            FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
//            depth_filter_->removeKeyframe(furthest_frame); // TODO this interrupts the mapper thread, maybe we can solve this better
//            map_.safeDeleteFrame(furthest_frame);
//        }
//
//        // add keyframe to map
//        map_.addKeyframe(new_frame_);

        return RESULT_IS_KEYFRAME;
    }


}
