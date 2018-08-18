#include "flame/mesh_estimator.hpp"

#include <atomic>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include <okvis/Estimator.hpp>

namespace flame {

    MeshEstimator::MeshEstimator(okvis::Estimator* estimator, int width, int height,
                                 const Matrix3f& K0, const Matrix3f& K0inv,
                                 const Vector4f& distort0,
                                 const Matrix3f& K1, const Matrix3f& K1inv,
                                 const Vector4f& distort1,
                                 const Params& parameters):
            params_(parameters),
            poseframe_subsample_factor_(6),
            estimator_(estimator),
            stats_(),
            inited_(false),
            num_data_updates_(0),
            num_regularizer_updates_(0),
            width_(width),
            height_(height),
            K0_(K0),
            K0inv_(K0inv),
            K1_(K1),
            K1inv_(K1inv),
            num_imgs_(0),
            fnew_(nullptr),
            fprev_(nullptr),
            update_mtx_(),
            pfs_(),
            curr_pf_(nullptr),
            new_feats_(),
            photo_error_(height, width, std::numeric_limits<float>::quiet_NaN()),
            feat_count_(0),
            feats_(),
            feats_in_curr_(),
            graph_(),
            graph_scale_(1.0f),
            graph_thread_(),
            graph_mtx_(),
            feat_to_vtx_(),
            vtx_to_feat_(),
            triangulator_(),
            triangulator_mtx_(),
            tri_validity_(),
            vtx_(),
            vtx_idepths_(),
            vtx_w1_(),
            vtx_w2_(),
            vtx_normals_(),
            idepthmap_(height, width, std::numeric_limits<float>::quiet_NaN()),
            w1_map_(height, width, std::numeric_limits<float>::quiet_NaN()),
            w2_map_(height, width, std::numeric_limits<float>::quiet_NaN()),
            debug_img_detections_(height, width),
            debug_img_wireframe_(height, width),
            debug_img_features_(height, width),
            debug_img_matches_(height, width),
            debug_img_normals_(height, width),
            debug_img_idepthmap_(height, width){

        cv::eigen2cv(K0, K0cv_);
        cv::eigen2cv(distort0, D0cv_);

        cv::eigen2cv(K1, K1cv_);
        cv::eigen2cv(distort1, D1cv_);

        sensor_ = std::make_shared<flame::Flame>(estimator,
                width,
                                                 height,
                                                 K0,
                                                 K0inv,
                                                 K1,
                                                 K1inv,
                                                 params_);

    }

    void MeshEstimator::processFrame( const double time, int32_t img_id,
                      const okvis::kinematics::Transformation& T_WC0,
                                      const cv::Mat& img_gray0,
                                      const okvis::kinematics::Transformation& T_WC1,
                                      const cv::Mat& img_gray1,bool isKeyframe) {
//
        /*==================== Process image ====================*/
        cv::Mat img_gray_undist0;
        cv::undistort(img_gray0, img_gray_undist0, K0cv_, D0cv_);

        cv::Mat img_gray_undist1;
        cv::undistort(img_gray1, img_gray_undist1, K1cv_, D1cv_);

        bool is_poseframe = isKeyframe;

        bool update_success = false;

        update_success = sensor_->update(time, img_id,
                T_WC0, img_gray_undist0,
                                        T_WC1, img_gray_undist1,
                                             is_poseframe);
//        if (!update_success) {
//            //ROS_WARN("FlameOffline: Unsuccessful update.\n");
//            return;
//        }
        Image3b wireImage = sensor_->getDebugImageWireframe();
//        Image3b wireImage = sensor_->getDebugImageFeatures();
        cv::imshow("wireImage", wireImage);

        Image3b depthImage = sensor_->getDebugImageInverseDepthMap();
        cv::imshow("depthImage", depthImage);
        cv::waitKey(2);
//
//        // todo : add publish and result into buffer

    }


    bool MeshEstimator::estimateMesh() {
        // Create flame frame
        int64_t cur_frame_id = estimator_->currentFrameId();
        okvis::MultiFramePtr cur_frame = estimator_->multiFrame(cur_frame_id);
        cv::Mat undistort0 = cur_frame->geometry(0)->undistortImage(cur_frame->image(0));
        cv::Mat undistort1 = cur_frame->geometry(1)->undistortImage(cur_frame->image(1));
        okvis::kinematics::Transformation T_WS, T_WC0, T_WC1;
        estimator_->get_T_WS(cur_frame_id, T_WS);
        T_WC0 = T_WS * (*cur_frame->T_SC(0));
        T_WC1 = T_WS * (*cur_frame->T_SC(1));


        fprev_ = fnew_;

        int border = params_.fparams.win_size;
        fnew_ = flame::utils::Frame::create(T_WC0,undistort0,cur_frame->id(),1, border);
        fnew_right_ = flame::utils::Frame::create(T_WC1,undistort1,cur_frame->id(),1, border);

        cur_frame->getDenseFrame(0) = fnew_;
        cur_frame->getDenseFrame(1) = fnew_right_;


//        cv::imshow("left: ", fnew_->img[0]);
//        cv::imshow("right: ", fnew_right_->img[0]);
//        cv::waitKey(2);

        if (estimator_->isKeyframe(cur_frame_id)) {
            curr_pf_ = fnew_;
        }



        if (inited_) {
            bool update =  updateMesh();


            /*==================== Add frames to detection queue ====================*/
            if (estimator_->isKeyframe(cur_frame_id)) {
                // Create detection data.
                DetectionData data;

                // Fill in reference frame info. We need to make a deep copy of idepthmap
                // (and pose, but that copy is deep by default) so that we don't need to
                // worry about locking/unlocking.
                data.ref = *curr_pf_;
                for (int ii = 0; ii < curr_pf_->idepthmap.size(); ++ii) {
                    data.ref.idepthmap[ii] = curr_pf_->idepthmap[ii].clone();
                }

                data.prev = *fprev_;


                // Fill in features projected into poseframe.
                data.ref_xy.resize(feats_in_curr_.size());
                for (int ii = 0; ii < feats_in_curr_.size(); ++ii) {
                    data.ref_xy[ii] = feats_in_curr_[ii].xy;
                }

                detectFeatures(data);

                //cv::imshow("image", debug_img_detections_);
            }



            return true;
        }

         if (estimator_->numFrames() == 1) {
             fprev_ = fnew_;
             return false;
         }


       if (estimator_->isKeyframe(cur_frame_id) && !inited_ ) {
            // Create initial detection data.
            // Use first two keyframe.
            DetectionData data;
            data.ref = *curr_pf_;
            data.prev = *fprev_;
            detectFeatures(data);
            inited_ = true;
            //cv::imshow("detect", debug_img_detections_);
            return false;
        }

        return true;
    }

    bool MeshEstimator::updateMesh() {

        /*==================== Add new features ====================*/
        if ((feats_.size() == 0) && (new_feats_.size() == 0)) {
            // No features to add.
            inited_ = false;
            return false;
        }

        /*==================== Add new features ====================*/
        feats_.insert(feats_.end(), new_feats_.begin(), new_feats_.end());
        new_feats_.clear();


//        /*==================== Update features ====================*/
//        // Update depth estimates.
//        bool idepth_success = updateFeatureIDepths(params_, K0_, K0inv_,
//                                                   K1_, K1inv_,
//                                                   estimator_, *fnew_, *fnew_right_,
//                                                   *curr_pf_, &feats_, &stats_,
//                                                   &debug_img_matches_);
//
//        if (!idepth_success) {
//            // No idepths could be updated.
//            // TODO(wng): Not sure what to do here.
//        }




        return true;
    }


    bool MeshEstimator::updateFeatureIDepths(const Params& params,
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
                                     Image3b* debug_img) {
        stats->tick("update_idepths");

        int debug_feature_radius = 4 * fnew.img[0].cols / 320; // For drawing features.

        if (params.debug_draw_matches) {
            cv::cvtColor(fnew.img[0], *debug_img, cv::COLOR_GRAY2RGB);
        }

        bool left_success = false;
        bool right_success = false;
        int num_total_updates = 0;

        // Count failure types
        std::atomic<int> num_fail_max_var(0);
        std::atomic<int> num_fail_max_dropouts(0);
        std::atomic<int> num_ref_patch(0);
        std::atomic<int> num_amb_match(0);
        std::atomic<int> num_max_cost(0);

        // Debug for right image update
        int left_update_cnt = 0;
        int right_update_cnt = 0;
        std::vector<cv::KeyPoint> kp_curr_pf, kp_new;
        Image3b colorCurr_pf, colorNew_left, colorNew_right;
        cv::cvtColor(curr_pf.img[0], colorCurr_pf, CV_GRAY2BGR);
        cv::cvtColor(fnew.img[0], colorNew_left, CV_GRAY2BGR);
        cv::cvtColor(fnew_right.img[0], colorNew_right, CV_GRAY2BGR);

        std::vector<std::pair<Point2f , Point2f >> flow_pairs;

#pragma omp parallel for num_threads(params.omp_num_threads) schedule(static, params.omp_chunk_size) // NOLINT
        for (int ii = 0; ii < feats->size(); ++ii) {
            FeatureWithIDepth& fii = (*feats)[ii];

            if ( fii.frame_id == curr_pf.id) {
                cv::KeyPoint kp;
                kp.pt = fii.xy;
                kp_curr_pf.push_back(kp);
            }


            bool left_update_success = false;
            bool right_update_success = false;

            /**
             *  Update Using Left Image
             */
            stereo::EpipolarGeometry<float> epigeo_left(K0, K0inv, K0, K0inv);
            // Load geometry.
            okvis::kinematics::Transformation new_pose, fii_pose;
            estimator->get_T_WS(fnew.id, new_pose);
            estimator->get_T_WS(fii.frame_id, fii_pose);
            okvis::kinematics::Transformation T_ref_to_new = new_pose.inverse() * fii_pose;
            okvis::kinematics::Transformation T_new_to_ref = fii_pose.inverse() * new_pose;
            epigeo_left.loadGeometry(T_ref_to_new.hamilton_quaternion().cast<float>(),
                                     T_ref_to_new.r().cast<float>());

            bool go_on_left = true;

            // Check baseline.
            float baseline = T_ref_to_new.r().cast<float>().norm();
            if (baseline < params.min_baseline) {
                // Not enough baseline. Skip.
                go_on_left = false;
            }



            /*==================== Track feature in new image ====================*/
            cv::Point2f left_flow;
            if (go_on_left) {
                float residual;
                bool track_success = trackFeature(params, K0, K0inv, estimator, epigeo_left, fnew,
                                                  curr_pf, &fii, &left_flow, &residual,
                                                  debug_img);


                // Count failure types.
                if (fii.search_status == stereo::inverse_depth_filter::Status::FAIL_REF_PATCH_GRADIENT) {
                    ++num_ref_patch;
                } else if (fii.search_status == stereo::inverse_depth_filter::Status::FAIL_AMBIGUOUS_MATCH) {
                    ++num_amb_match;
                } else if (fii.search_status == stereo::inverse_depth_filter::Status::FAIL_MAX_COST) {
                    ++num_max_cost;
                } else {
                    // Unknown status.
                }

                if (!track_success) {
                    fii.idepth_var *= params.fparams.process_fail_var_factor;
                    if (fii.idepth_var > params.idepth_var_max) {
                        fii.valid = false;
                        ++num_fail_max_var;

                        if (params.debug_draw_matches) {
                            cv::Scalar color(0, 255, 0); // Green for max var.
                            float blah;
                            cv::Point2f fii_cmp;
                            epigeo_left.project(fii.xy, fii.idepth_mu, &fii_cmp, &blah);
                            cv::circle(*debug_img, cv::Point2i(fii_cmp.x + 0.5f, fii_cmp.y + 0.5f),
                                       debug_feature_radius, color);
                        }
                    }

                    fii.num_dropouts++;
                    if (fii.num_dropouts > params.max_dropouts) {
                        fii.valid = false;
                        ++num_fail_max_dropouts;

                        if (params.debug_draw_matches) {
                            cv::Scalar color(255, 0, 0); // Blue for max dropouts.
                            float blah;
                            cv::Point2f fii_cmp;
                            epigeo_left.project(fii.xy, fii.idepth_mu, &fii_cmp, &blah);
                            cv::circle(*debug_img, cv::Point2i(fii_cmp.x + 0.5f, fii_cmp.y + 0.5f),
                                       debug_feature_radius, color);
                        }
                    }

                    go_on_left = false;
                }
            }


            /*==================== Update idepth ====================*/
            // Load stuff into meas model.
            float mu_meas_left, var_meas_left;
            if (go_on_left) {
                stereo::InverseDepthMeasModel model(K0, K0inv, K0, K0inv, params.zparams);
                okvis::MultiFramePtr fii_frame = estimator->multiFrame(fii.frame_id);

                auto& pfii = fii_frame->getDenseFrame(0);
                okvis::kinematics::Transformation fii_pose;
                estimator->get_T_WS(pfii->id, fii_pose);
                model.loadGeometry(fii_pose, new_pose);
                model.loadPaddedImages(pfii->img_pad[0], fnew.img_pad[0],
                                       pfii->gradx_pad[0],
                                       pfii->grady_pad[0],
                                       fnew.gradx_pad[0], fnew.grady_pad[0]);

                // Generate measurement.

                bool sense_success = model.idepth(fii.xy, left_flow, &mu_meas_left, &var_meas_left);

                if (!sense_success) {
                    if (!params.debug_quiet && params.debug_print_verbose_errors) {
                        fprintf(stderr, "FAIL:Sense: u_ref = (%f, %f), id = %f, var = %f\n",
                                fii.xy.x, fii.xy.y, fii.idepth_mu, fii.idepth_var);
                    }

                    cv::Scalar color;
                    fii.idepth_var *= params.fparams.process_fail_var_factor;
                    if (fii.idepth_var > params.idepth_var_max) {
                        fii.valid = false;
                        ++num_fail_max_var;


                        if (params.debug_draw_matches) {
                            cv::Scalar color(0, 255, 0); // Green for max var.
                            float blah;
                            cv::Point2f fii_cmp;
                            epigeo_left.project(fii.xy, fii.idepth_mu, &fii_cmp, &blah);
                            cv::circle(*debug_img, cv::Point2i(fii_cmp.x + 0.5f, fii_cmp.y + 0.5f),
                                       debug_feature_radius, color);
                        }
                    }

                    fii.num_dropouts++;
                    if (fii.num_dropouts > params.max_dropouts) {
                        fii.valid = false;
                        ++num_fail_max_dropouts;

                        if (params.debug_draw_matches) {
                            cv::Scalar color(255, 0, 0); // Blue for max dropouts.
                            float blah;
                            cv::Point2f fii_cmp;
                            epigeo_left.project(fii.xy, fii.idepth_mu, &fii_cmp, &blah);
                            cv::circle(*debug_img, cv::Point2i(fii_cmp.x + 0.5f, fii_cmp.y + 0.5f),
                                       debug_feature_radius, color);
                        }
                    }
                    go_on_left = false;

                }
            }


            // Fuse.
            float mu_post_left, var_post_left;
            if (go_on_left) {
                bool fuse_success =
                        stereo::inverse_depth_filter::update(fii.idepth_mu,
                                                             fii.idepth_var,
                                                             mu_meas_left, var_meas_left,
                                                             &mu_post_left, &var_post_left,
                                                             params.outlier_sigma_thresh);

                if (!fuse_success) {
                    if (!params.debug_quiet && params.debug_print_verbose_errors) {
                        fprintf(stderr, "FAIL:Fuse: mu_meas = %f, var_meas = %f\n",
                                mu_meas_left, var_meas_left);
                    }

                    cv::Scalar color;
                    fii.idepth_var *= params.fparams.process_fail_var_factor;
                    if (fii.idepth_var > params.idepth_var_max) {
                        fii.valid = false;
                        ++num_fail_max_var;

                        if (params.debug_draw_matches) {
                            cv::Scalar color(0, 255, 0); // Green for max var.
                            float blah;
                            cv::Point2f fii_cmp;
                            epigeo_left.project(fii.xy, fii.idepth_mu, &fii_cmp, &blah);
                            cv::circle(*debug_img, cv::Point2i(fii_cmp.x + 0.5f, fii_cmp.y + 0.5f),
                                       debug_feature_radius, color);
                        }
                    }

                    fii.num_dropouts++;
                    if (fii.num_dropouts > params.max_dropouts) {
                        fii.valid = false;
                        ++num_fail_max_dropouts;

                        if (params.debug_draw_matches) {
                            cv::Scalar color(255, 0, 0); // Blue for max dropouts.
                            float blah;
                            cv::Point2f fii_cmp;
                            epigeo_left.project(fii.xy, fii.idepth_mu, &fii_cmp, &blah);
                            cv::circle(*debug_img, cv::Point2i(fii_cmp.x + 0.5f, fii_cmp.y + 0.5f),
                                       debug_feature_radius, color);
                        }
                    }

                    go_on_left = false;
                }
            }

            // update
            if (go_on_left) {
                if (params.do_meas_fusion) {
                    fii.idepth_mu = mu_post_left;
                    fii.idepth_var = var_post_left;
                } else {
                    fii.idepth_mu = mu_meas_left;
                    fii.idepth_var = var_meas_left;
                }

                fii.valid = true;
                fii.num_updates++;
                fii.num_dropouts = 0;
                num_total_updates++;

                left_success = true;
                left_update_success = true;
                left_update_cnt ++;


                // todo(pang): add observation into estimator
            }

            /**
           *  Update Using Right Image
           */
//
//            if (! params.filte_with_stereo_image) continue;
//
//            stereo::EpipolarGeometry<float> epigeo_right(K0, K0inv, K1, K1inv);
//            // Load geometry.
//            okvis::kinematics::Transformation T_ref_to_right = fnew_right.pose.inverse() * pfs.at(fii.frame_id)->pose;
//            okvis::kinematics::Transformation T_right_to_ref = pfs.at(fii.frame_id)->pose.inverse() * fnew_right.pose;
//            epigeo_right.loadGeometry(T_ref_to_right.hamilton_quaternion().cast<float>(),
//                                      T_ref_to_right.r().cast<float>());
//
//            /*==================== Track feature in new image ====================*/
//            cv::Point2f right_flow;
//            float residual;
//            bool track_success = trackFeatureRight(params, K0, K0inv, K1, K1inv, pfs,
//                                                   epigeo_right, fnew_right,
//                                                   curr_pf, &fii, &right_flow,
//                                                   &residual,
//                                                   &colorNew_right);
//
//            if (track_success) {
//
//            } else {
//                continue;
//            }
//
//            // todo: enssential matrix check
//
//            /*==================== Update idepth ====================*/
//            // Load stuff into meas model.
//            stereo::InverseDepthMeasModel model(K0, K0inv, K1, K1inv, params.zparams);
//            auto& pfii = pfs.at(fii.frame_id);
//            model.loadGeometry(pfii->pose, fnew_right.pose);
//            model.loadPaddedImages(pfii->img_pad[0], fnew_right.img_pad[0],
//                                   pfii->gradx_pad[0],
//                                   pfii->grady_pad[0],
//                                   fnew_right.gradx_pad[0], fnew_right.grady_pad[0]);
//
//            // Generate measurement.
//            float mu_meas_right, var_meas_right;
//            bool sense_success = model.idepth(fii.xy, right_flow, &mu_meas_right, &var_meas_right);
//
//            if (!sense_success) {
//                continue;
//            }
//
//            // Fuse.
//            float mu_post_right, var_post_right;
//            bool fuse_success =
//                    stereo::inverse_depth_filter::update(fii.idepth_mu,
//                                                         fii.idepth_var,
//                                                         mu_meas_right, var_meas_right,
//                                                         &mu_post_right, &var_post_right,
//                                                         params.outlier_sigma_thresh);
//
//            if (!fuse_success) {
//                continue;
//            }
//
//            if (params.do_meas_fusion) {
//                fii.idepth_mu = mu_post_right;
//                fii.idepth_var = var_post_right;
//            } else {
//                fii.idepth_mu = mu_meas_right;
//                fii.idepth_var = var_meas_right;
//            }
//
//            fii.valid = true;
//            fii.num_updates++;
//            fii.num_dropouts = 0;
//
//            right_success = true;
//            right_update_success = true;
//            right_update_cnt ++;
//
//
//            // todo(pang): add observation into estimator
//
//
//
//            if (left_update_success && right_update_success) {
//                std::pair<Point2f , Point2f > flow_pair(left_flow, right_flow);
//                flow_pairs.push_back(flow_pair);
//            }
        }

        // Fill in some stats.
        stats->set("num_idepth_updates", num_total_updates);
        stats->set("num_fail_max_var", num_fail_max_var.load());
        stats->set("num_fail_max_dropouts", num_fail_max_dropouts.load());
        stats->set("num_fail_ref_patch_grad", num_ref_patch.load());
        stats->set("num_fail_ambiguous_match", num_amb_match.load());
        stats->set("num_fail_max_cost", num_max_cost.load());

        if (params.debug_draw_matches) {
            if (params.debug_flip_images) {
                // Flip image for display.
                cv::flip(*debug_img, *debug_img, -1);
            }

            if (params.debug_draw_text_overlay) {
                // Print some info.
                char buf[200];
                snprintf(buf, sizeof(buf), "%i updates, %i fails (%i ref_patch_grad, %i, amb_match, %i max_cost)",
                         num_total_updates, feats->size() - num_total_updates,
                         num_ref_patch.load(), num_amb_match.load(), num_max_cost.load());
                float font_scale = 0.6 / (640.0f / debug_img->cols);
                int font_thickness = 2.0f / (640.0f / debug_img->cols) + 0.5f;
                cv::putText(*debug_img, buf,
                            cv::Point(10, debug_img->rows - 5),
                            cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(200, 200, 250),
                            font_thickness, 8);
            }
        }

        stats->tock("update_idepths");
        if (!params.debug_quiet && params.debug_print_timing_update_idepths) {
            printf("Flame/update_idepths(%i) = %f ms\n",
                   feats->size(), stats->timings("update_idepths"));
        }


        cv::drawKeypoints(colorCurr_pf,kp_curr_pf,colorCurr_pf, cv::Scalar(225, 0,0 ));
        ///cv::drawKeypoints(colorNew,kp_new,colorNew, cv::Scalar(0,225, 0 ));


        for (auto pair :  flow_pairs) {
            cv::circle(colorNew_left, pair.first, 1, cv::Scalar(255, 0, 0));
            cv::circle(colorNew_left, pair.second, 1, cv::Scalar(0, 255, 0));
            cv::line(colorNew_left, pair.first, pair.second, cv::Scalar(0, 0,255));
        }

        //cv::imshow("curr_pf", colorCurr_pf);
        cv::imshow("fnew", 0.5*colorNew_left + 0.5*colorNew_right);
        cv::imshow("fnew_right", colorNew_right);
        cv::waitKey(2);



        return left_success || right_success;
    }


    bool MeshEstimator::trackFeature(const Params& params,
                             const Matrix3f& K,
                             const Matrix3f& Kinv,
                             const okvis::Estimator* estimator,
                             const stereo::EpipolarGeometry<float>& epigeo,
                             const utils::Frame& fnew,
                             const utils::Frame& curr_pf,
                             FeatureWithIDepth* feat,
                             cv::Point2f* flow,
                             float* residual,
                             Image3b* debug_img) {
        int debug_feature_radius = fnew.img[0].cols / 320; // For drawing features.
        cv::Point2i debug_feature_offset(debug_feature_radius, debug_feature_radius);

        /*==================== Predict feature in new image ====================*/
        cv::Point2f u_cmp;
        float idepth_cmp, var_cmp;
        bool pred_success =
                stereo::inverse_depth_filter::predict(epigeo,
                                                      params.fparams.process_var_factor,
                                                      feat->xy,
                                                      feat->idepth_mu,
                                                      feat->idepth_var,
                                                      &u_cmp, &idepth_cmp, &var_cmp);

        if (!pred_success) {
            // TOOD(wng): Dropout count.
            return false;
        }

        /*==================== Use LSD-SLAM style direct search ====================*/
        int width = fnew.img[0].cols;
        int height = fnew.img[0].rows;

        int row_offset = 0;
        if (params.do_letterbox) {
            // Only detect features in middle third of image.
            row_offset = height/3;
        }

        int border = params.rescale_factor_max * params.fparams.win_size / 2 + 1;
        cv::Rect valid_region(border, border + row_offset,
                              width - 2*border, height - 2*border - 2*row_offset);

        // Compute rescale factor. This is how much to grow/shrink the reference
        // patch based on the difference in idepth between the reference and
        // comparison frames (i.e. travel along the optical axis).
        float rescale_factor = 1.0f;
        if ((feat->idepth_mu > 0.0f) && (idepth_cmp > 0.0f)) {
            rescale_factor = idepth_cmp / feat->idepth_mu;
        }

        FLAME_ASSERT(!std::isnan(rescale_factor));
        FLAME_ASSERT(rescale_factor > 0);

        if ((rescale_factor <= params.rescale_factor_min) ||
            (rescale_factor >= params.rescale_factor_max)) {
            // Warp on reference patch is too large - i.e. idepth difference between
            // reference frame and comparison frame is too large. Move the feature to
            // the most recent pf.
            bool verbose = false;
            if (verbose) {
                fprintf(stderr, "Flame[FAIL]: bad rescale_factor = %f, prior_idepth = %f, idepth_cmp = %f\n",
                        rescale_factor, feat->idepth_mu, idepth_cmp);
            }

            if (verbose) {
                fprintf(stderr, "Flame[WARNING]: Moving feature from u_ref = (%f, %f) idepth = %f to u_cmp = (%f, %f) idepth = %f\n",
                        feat->xy.x, feat->xy.y, feat->idepth_mu,
                        u_cmp.x, u_cmp.y, idepth_cmp);
            }

            // If this feature has converged already, move it so that it's parent
            // pose is the most recent poseframe frame rather than throw it away.
            stereo::EpipolarGeometry<float> epipf(K, Kinv, K, Kinv);
            okvis::kinematics::Transformation curr_pose, feat_frame_pose;
            estimator->get_T_WS(curr_pf.id, curr_pose);
            estimator->get_T_WS(feat->frame_id, feat_frame_pose);
            okvis::kinematics::Transformation T_old_to_new = curr_pose.inverse() * feat_frame_pose;
            epipf.loadGeometry(T_old_to_new.hamilton_quaternion().cast<float>(),
                               T_old_to_new.r().cast<float>());

            cv::Point2f u_pf;
            float idepth_pf, var_pf;
            bool move_success =
                    stereo::inverse_depth_filter::predict(epipf,
                                                          params.fparams.process_var_factor,
                                                          feat->xy,
                                                          feat->idepth_mu,
                                                          feat->idepth_var,
                                                          &u_pf, &idepth_pf, &var_pf);
            if (!move_success || !valid_region.contains(u_pf)) {
                feat->valid = false;
                if (params.debug_draw_matches) {
                    // Failed move in brown.
                    cv::Point2i u_cmpi(u_cmp.x + 0.5f, u_cmp.y + 0.5f);
                    cv::rectangle(*debug_img, u_cmpi - debug_feature_offset,
                                  u_cmpi + debug_feature_offset, cv::Scalar(0, 51, 102), -1);
                }
                return false;
            }

            feat->frame_id = curr_pf.id;
            feat->xy = u_pf;
            float old_idepth = feat->idepth_mu;
            feat->idepth_mu = idepth_pf;

            // Project idepth variance.
            float var_factor4 = idepth_pf / old_idepth;
            var_factor4 *= var_factor4;
            var_factor4 *= var_factor4;

            if (idepth_pf < 1e-6) {
                // If feat_ref.idepth_mu == 0, then var_factor4 is inf.
                var_factor4 = 1;
            }
            feat->idepth_var *= var_factor4;

            if (params.debug_draw_matches) {
                // Successful move in magenta.
                cv::Point2i u_cmpi(u_cmp.x + 0.5f, u_cmp.y + 0.5f);
                cv::rectangle(*debug_img, u_cmpi - debug_feature_offset,
                              u_cmpi + debug_feature_offset, cv::Scalar(255, 0, 255), -1);
            }

            return false;
        }

        cv::Point2f u_start, u_end, epi;
        bool region_success =
                stereo::inverse_depth_filter::getSearchRegion(params.fparams, epigeo,
                                                              width, height, feat->xy,
                                                              feat->idepth_mu, feat->idepth_var,
                                                              &u_start, &u_end, &epi);
        if (!region_success) {
            if (params.debug_draw_matches) {
                // Failed search region in black.
                cv::Point2i u_cmpi(u_cmp.x + 0.5f, u_cmp.y + 0.5f);
                cv::rectangle(*debug_img, u_cmpi - debug_feature_offset,
                              u_cmpi + debug_feature_offset, cv::Scalar(0, 0, 0), -1);
            }
            return false;
        }

        int padding = (fnew.img_pad[0].rows - fnew.img[0].rows) / 2;
        cv::Point2f offset(padding, padding);

        if (!valid_region.contains(feat->xy)) {
            if (!params.debug_quiet) {
                printf("Flame[WARNING]: Feature outside bounds: feat->xy = %f, %f\n",
                       feat->xy.x, feat->xy.y);
            }
            return false;
        }
        // FLAME_ASSERT(valid_region.contains(feat->xy));

        okvis::MultiFramePtr feat_frame = estimator->multiFrame(feat->frame_id);
        auto search_success =
                stereo::inverse_depth_filter::search(params.fparams, epigeo, rescale_factor,
                                                     feat_frame->getDenseFrame(0)->img_pad[0],
                                                     fnew.img_pad[0],
                                                     feat->xy + offset, u_start + offset,
                                                     u_end + offset, &u_cmp);
        feat->search_status = search_success;

        // Parse output.
        if (search_success != stereo::inverse_depth_filter::SUCCESS) {
            if (params.debug_draw_matches) {
                // Color failure by error status.
                cv::Vec3b color;

                if (search_success == stereo::inverse_depth_filter::FAIL_REF_PATCH_GRADIENT) {
                    color = cv::Vec3b(255, 255, 0); // Cyan.
                    if (feat->num_updates == 0) {
                        color = cv::Vec3b(255, 255, 255); // White.
                    }
                } else if (search_success == stereo::inverse_depth_filter::FAIL_AMBIGUOUS_MATCH) {
                    color = cv::Vec3b(0, 0, 255); // Red.
                } else if (search_success == stereo::inverse_depth_filter::FAIL_MAX_COST) {
                    color = cv::Vec3b(0, 255, 255); // Yellow.
                } else if (search_success != stereo::inverse_depth_filter::SUCCESS) {
                    fprintf(stderr, "inverse_depth_filter::search: Unrecognized status!\n");
                    FLAME_ASSERT(false);
                    return false;
                }

                cv::Point2i u_cmpi(u_cmp.x + 0.5f, u_cmp.y + 0.5f);
                cv::rectangle(*debug_img, u_cmpi - debug_feature_offset,
                              u_cmpi + debug_feature_offset,
                              cv::Scalar(color[0], color[1], color[2]), -1);

                auto colormap = [&color](float a) { return color; };
                utils::applyColorMapLine(u_start, u_end, 1, 1, colormap, 0.5, debug_img);
            }

            return false;
        }

        *flow = u_cmp - offset;

        // if (params.debug_draw_matches) {
        //   cv::Point2i flowi(flow->x + 0.5f, flow->y + 0.5f);
        //   // cv::circle(*debug_img, cv::Point2i(flow->x + 0.5f, flow->y + 0.5f),
        //   //            2, cv::Scalar(0, 255, 0));

        //   // cv::Vec3b color(0, 255, 0);

        //   cv::Vec3b color = utils::blendColor(cv::Vec3b(255, 0, 0),
        //                                       cv::Vec3b(0, 255, 0),
        //                                       feat->num_updates, 0, 30);
        //   // cv::Vec3b color = utils::jet(feat->num_updates, 0, 30);
        //   cv::rectangle(*debug_img, flowi - debug_feature_offset,
        //                 flowi + debug_feature_offset,
        //                 cv::Scalar(color[0], color[1], color[2]), -1);

        //   auto colormap = [&color](float a) { return color; };
        //   utils::applyColorMapLine(u_start, u_end, 1, 1, colormap, 0.5, debug_img);
        // }

        return true;
    }

    bool MeshEstimator::trackFeatureRight(const Params& params,
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
                                  Image3b* debug_img) {
        int debug_feature_radius = fnew.img[0].cols / 320; // For drawing features.
        cv::Point2i debug_feature_offset(debug_feature_radius, debug_feature_radius);

        /*==================== Predict feature in new image ====================*/

        cv::Point2f u_cmp;
        float idepth_cmp, var_cmp;
        bool pred_success =
                stereo::inverse_depth_filter::predict(epigeo,
                                                      params.fparams.process_var_factor,
                                                      feat->xy,
                                                      feat->idepth_mu,
                                                      feat->idepth_var,
                                                      &u_cmp, &idepth_cmp, &var_cmp);
        if (!pred_success) {
            //std::cout<< "right predict failure" << std::endl;
            return false;
        }

        if (feat->frame_id == curr_pf.id) {
//    std::vector<cv::KeyPoint> kps;
//    cv::KeyPoint kp; kp.pt = u_cmp; kps.push_back(kp);
//    cv::drawKeypoints(*debug_img, kps, *debug_img, cv::Scalar(225, 0,0));
        }



        float rescale_factor = 1.0f;
        if ((feat->idepth_mu > 0.0f) && (idepth_cmp > 0.0f)) {
            rescale_factor = idepth_cmp / feat->idepth_mu;
        }

        /*==================== Use LSD-SLAM style direct search ====================*/
        int width = fnew.img[0].cols;
        int height = fnew.img[0].rows;

        int row_offset = 0;
        if (params.do_letterbox) {
            // Only detect features in middle third of image.
            row_offset = height/3;
        }

        int border = params.rescale_factor_max * params.fparams.win_size / 2 + 1;
        cv::Rect valid_region(border, border + row_offset,
                              width - 2*border, height - 2*border - 2*row_offset);



        cv::Point2f u_start, u_end, epi;
        bool region_success =
                stereo::inverse_depth_filter::getSearchRegion(params.fparams, epigeo,
                                                              width, height, feat->xy,
                                                              feat->idepth_mu, feat->idepth_var,
                                                              &u_start, &u_end, &epi);


        if (region_success) {
//    if (feat->frame_id == curr_pf.id) {
//      cv::line(*debug_img, u_start, u_end, cv::Scalar(33,54,234));
//    }
        }else{
            // Failed search region in black.
            cv::Point2i u_cmpi(u_cmp.x + 0.5f, u_cmp.y + 0.5f);
            cv::rectangle(*debug_img, u_cmpi - debug_feature_offset,
                          u_cmpi + debug_feature_offset, cv::Scalar(45, 233, 0), -1);
            return false;
        }

        int padding = (fnew.img_pad[0].rows - fnew.img[0].rows) / 2;
        cv::Point2f offset(padding, padding);

        if (!valid_region.contains(feat->xy)) {
            if (!params.debug_quiet) {
                printf("Flame[WARNING]: Feature outside bounds: feat->xy = %f, %f\n",
                       feat->xy.x, feat->xy.y);
            }
            return false;
        }
        // FLAME_ASSERT(valid_region.contains(feat->xy));
        okvis::MultiFramePtr feat_frame = estimator->multiFrame(feat->frame_id);
        auto search_success =
                stereo::inverse_depth_filter::search(params.fparams, epigeo, rescale_factor,
                                                     feat_frame->getDenseFrame(1)->img_pad[0],
                                                     fnew.img_pad[0],
                                                     feat->xy + offset, u_start + offset,
                                                     u_end + offset, &u_cmp);
        feat->search_status = search_success;

        // Parse output.
        if (search_success == stereo::inverse_depth_filter::SUCCESS) {
//    if (feat->frame_id == curr_pf.id) {
//      std::vector<cv::KeyPoint> kps;
//      cv::KeyPoint kp; kp.pt = u_cmp; kps.push_back(kp);
//      cv::drawKeypoints(*debug_img, kps, *debug_img, cv::Scalar(0,255,0));
//    }

        }else{

            // Color failure by error status.
            cv::Vec3b color;

            if (search_success == stereo::inverse_depth_filter::FAIL_REF_PATCH_GRADIENT) {
                color = cv::Vec3b(255, 255, 0); // Cyan.
                if (feat->num_updates == 0) {
                    color = cv::Vec3b(255, 255, 255); // White.
                }
            } else if (search_success == stereo::inverse_depth_filter::FAIL_AMBIGUOUS_MATCH) {
                color = cv::Vec3b(0, 0, 255); // Red.
            } else if (search_success == stereo::inverse_depth_filter::FAIL_MAX_COST) {
                color = cv::Vec3b(0, 255, 255); // Yellow.
            } else if (search_success != stereo::inverse_depth_filter::SUCCESS) {
                fprintf(stderr, "inverse_depth_filter::search: Unrecognized status!\n");
                FLAME_ASSERT(false);
                return false;
            }

//      cv::Point2i u_cmpi(u_cmp.x + 0.5f, u_cmp.y + 0.5f);
//      cv::rectangle(*debug_img, u_cmpi - debug_feature_offset,
//                    u_cmpi + debug_feature_offset,
//                    cv::Scalar(color[0], color[1], color[2]), -1);
//
//      auto colormap = [&color](float a) { return color; };
//      utils::applyColorMapLine(u_start, u_end, 1, 1, colormap, 0.5, debug_img);
            return false;
        }

        *flow = u_cmp - offset;



        return true;
    }






    void MeshEstimator::detectFeatures(DetectionData& data) {
        // Detect new features if this is a poseframe.
        std::vector<cv::Point2f> new_feats;
        if (params_.continuous_detection ||
            (!params_.continuous_detection && (num_data_updates_ < 1))) {
            detectFeatures(params_, K0_, K0inv_,
                           data.ref, data.prev,  data.ref.idepthmap[0],
                           data.ref_xy, &photo_error_,
                           &new_feats, &stats_, &debug_img_detections_);
        }

        // Add new features to list.
        for (int ii = 0; ii < new_feats.size(); ++ii) {
            FeatureWithIDepth newf;
            newf.id = feat_count_++;
            newf.frame_id = data.ref.id;
            newf.xy = new_feats[ii];
            newf.idepth_var = params_.idepth_var_init;
            newf.valid = true;
            newf.num_updates = 0;

            // Initialize idepth from dense idepthmap if possible.
            newf.idepth_mu =  params_.idepth_init;
            int x = utils::fast_roundf(newf.xy.x);
            int y = utils::fast_roundf(newf.xy.y);
            if (!std::isnan(data.ref.idepthmap[0](y, x))) {
                newf.idepth_mu = data.ref.idepthmap[0](y, x);
            }

            new_feats_.push_back(newf);
        }
    }


    void MeshEstimator::detectFeatures(const Params& params,
                               const Matrix3f& K,
                               const Matrix3f& Kinv,
                               const utils::Frame& fref,
                               const utils::Frame& fprev,
                               const Image1f& idepthmap,
                               const std::vector<cv::Point2f>& curr_feats,
                               Image1f* error,
                               std::vector<cv::Point2f>* features,
                               utils::StatsTracker* stats,
                               Image3b* debug_img) {
        // Sample points.
        stats->tick("detection");

        int width = fref.img[0].cols;
        int height = fref.img[0].rows;

        int row_offset = 0;
        if (params.do_letterbox) {
            // Only detect features in middle third of image.
            row_offset = height/3;
        }

        // Images aren't padded - accessing pixels outside this region may result in a
        // segfault.
        int border = params.rescale_factor_max * params.fparams.win_size / 2 + 1;
        cv::Rect valid_region(border, border + row_offset,
                              width - 2*border, height - 2*border - 2*row_offset);

        stereo::EpipolarGeometry<float> epigeo(K, Kinv, K, Kinv);
        /*==================== Detect features on grid with single pass ====================*/
        // Compute mask where we already have features.
        int win_size = params.detection_win_size;
        int hclvl = utils::fast_ceil(static_cast<float>(height) / win_size);
        int wclvl = utils::fast_ceil(static_cast<float>(width) / win_size);
        Image1b cmask(hclvl, wclvl, 255);

        for (uint32_t ii = 0; ii < curr_feats.size(); ++ii) {
            // Fill in coarse mask.
            uint32_t x_clvl = curr_feats[ii].x / win_size;
            uint32_t y_clvl = curr_feats[ii].y / win_size;
            cmask(y_clvl, x_clvl) = 0;
        }

        // Load epipolar geometry from prev to ref.
        okvis::kinematics::Transformation T_ref_to_prev(fprev.pose.inverse() * fref.pose);
        epigeo.loadGeometry(T_ref_to_prev.hamilton_quaternion().cast<float>(),
                            T_ref_to_prev.r().cast<float>());

        // Coarse pass.
        Image1f score(height, width, std::numeric_limits<float>::quiet_NaN());
        Image1f best_gradsc(hclvl, wclvl, 0.0f);
        std::vector<cv::Point2f> best_pxc(hclvl * wclvl);
        float grad_thresh2 = params.min_grad_mag * params.min_grad_mag;
        for (uint32_t ii = border + row_offset; ii < height - border - row_offset; ++ii) {
            for (uint32_t jj = border; jj < width - border; ++jj) {
                // Check gradient magnitude.
                float gx = fref.gradx[0](ii, jj);
                float gy = fref.grady[0](ii, jj);
                float gmag2 = gx*gx + gy*gy;
                if (gmag2 < grad_thresh2) {
                    continue;
                }

                // Check gradient magnitude in epipolar direction.
                cv::Point2f epi_ref;
                epigeo.referenceEpiline(cv::Point2f(ii, jj), &epi_ref);

                float epigrad = gx * epi_ref.x + gy * epi_ref.y;
                float epigrad2 = epigrad * epigrad;

                if (epigrad2 < grad_thresh2) {
                    // Gradient isn't large enough, skip.
                    continue;
                }

                int ii_clvl = static_cast<float>(ii) / win_size;
                int jj_clvl = static_cast<float>(jj) / win_size;
                int idx_clvl = ii_clvl * wclvl + jj_clvl;

                // Fill in best gradients.
                if (epigrad2 >= best_gradsc(ii_clvl, jj_clvl)) {
                    best_gradsc(ii_clvl, jj_clvl) = epigrad2;
                    best_pxc[idx_clvl] = cv::Point2f(jj, ii);
                }

                // Fill in score (for visualization).
                score(ii, jj) = utils::fast_abs(epigrad);
            }
        }

        // Now extract detections of grid.
        std::vector<cv::KeyPoint> kps;
        for (int ii = 0; ii < hclvl; ++ii) {
            for (int jj = 0; jj < wclvl; ++jj) {
                int idx = ii * wclvl + jj;
                if ((cmask(ii, jj) > 0) && (best_gradsc(ii, jj) > 0)) {
                    kps.push_back(cv::KeyPoint(best_pxc[idx], 1));
                }
            }
        }

        cv::KeyPoint::convert(kps, *features);

        stats->tock("detection");
        if (!params.debug_quiet && params.debug_print_timing_detection) {
            printf("Flame/detection(%i) = %f ms\n",
                   features->size(), stats->timings("detection"));
        }

        if (params.debug_draw_detections) {
            drawDetections(params, fref.img[0], score, 0.005f, kps, stats,
                           debug_img);
        }

        return;
    }



    void MeshEstimator::drawDetections(const Params& params,
                               const Image1b& img,
                               const Image1f& score,
                               float max_score,
                               const std::vector<cv::KeyPoint>& kps,
                               utils::StatsTracker* stats,
                               Image3b* debug_img) {
        Image3b img_rgb;
        cv::cvtColor(img, img_rgb, cv::COLOR_GRAY2RGB);

        auto colormap = [max_score](float v, cv::Vec3b c) {
            if (!std::isnan(v)) {
                return utils::jet(v, 0, max_score);
            } else {
                return c;
            }
        };
        if (debug_img->empty()) {
            debug_img->create(score.rows, score.cols);
        }
        debug_img->setTo(cv::Vec3b(0, 0, 0));

        utils::applyColorMap<float>(score, colormap, debug_img);

        // Blend source image and score image.
        *debug_img = 0.7*(*debug_img) + 0.3*img_rgb;

        // cv::drawKeypoints(*debug_img, kps, *debug_img);

        if (params.debug_flip_images) {
            // Flip image for display.
            cv::flip(*debug_img, *debug_img, -1);
        }

        if (params.debug_draw_text_overlay) {
            // Print some info.
            char buf[200];
            snprintf(buf, sizeof(buf), "%4.1fms, %lu new feats, avg_err = %.2f",
                     stats->timings("update"), kps.size(),
                     stats->stats("avg_photo_score"));
            float font_scale = 0.6 / (640.0f / debug_img->cols);
            int font_thickness = 2.0f / (640.0f / debug_img->cols) + 0.5f;
            cv::putText(*debug_img, buf,
                        cv::Point(10, debug_img->rows - 5),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(200, 200, 250),
                        font_thickness, 8);
        }

        return;
    }


}