#include "cfg_common/slam_config.h"

#include "Eigen/Geometry"

namespace common {
SlamConfig::SlamConfig(const std::string& file_path,
                       const bool online_mode)
    : ConfigBase(file_path),
      online(online_mode) {
    LoadConfigFromFile();
}

void SlamConfig::LoadConfigFromFile() {
    try {
        config_node_ = YAML::LoadFile(config_file_path_.c_str());
        LOG(INFO) << "Loaded the config YAML file " << config_file_path_;
    } catch (const std::exception& ex) {
        LOG(ERROR) << "Failed to open and parse the config YAML file "
                     << config_file_path_ << " with the error: " << ex.what()
                     << ". So default configuration will be used.";
        return;
    }

    config_node_ = config_node_["slam_config"];

    size_t load_option = 0u;
    SetValueBasedOnYamlKey(config_node_,
                             kYamlFieldLoadOption,
                             &load_option);
    load_option_ = static_cast<LoadOption>(load_option);

    // common config.
    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldDataPath,
        &data_path);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldDataRealtimePlaybackRate,
        &data_realtime_playback_rate);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldLogPath,
        &log_path);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFeatureTrackingTestPath,
        &feature_tracking_test_path);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldLogTimeTable,
        &log_time_table);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldMapPath,
        &map_path);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldMapping,
        &mapping);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldCalibYamlConfig,
        &calib_yaml_config);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldAssetsPath,
        &assets_path);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldSuperPointModelFile,
        &superpoint_model_file);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldMaskPath,
        &mask_path);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldDoOctoMapping,
        &do_octo_mapping);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldDoObstacleRemoval,
        &do_obstacle_removal);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldContourLength,
        &contour_length);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldDoExOnlineCalib,
        &do_ex_online_calib);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldDoTimeOnlineCalib,
        &do_time_online_calib);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldWinsdowSize,
        &window_size);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldMainSensorType,
        &main_sensor_type);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldZeroVelocityPixelDiff,
        &zero_velocity_pixel_diff);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldMaxFeatureSizeInOpt,
        &max_feature_size_in_opt);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldVisualSigmaPixel,
        &visual_sigma_pixel);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldDepthSigmaShort,
        &depth_sigma_short);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldDepthSigmaMid,
        &depth_sigma_mid);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldDepthSigmaFar,
        &depth_sigma_far);

    // Compute depth noise params.
    // Moding by y = ax^2 + bx + c
    // e.g.
    // 1) 0.01 = 0.1*0.1*a + 0.1*b + c
    // 2) 0.05 = 5*5*a + 5*b + c
    // 3) 0.15 = 10*10*a + 10*b + c
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> A;
    A << 0.01, 0.1, 1.0,
          25.0, 5.0, 1.0,
          100.0, 10.0, 1.0;
    Eigen::Vector3d b;
    b << depth_sigma_short,
         depth_sigma_mid,
         depth_sigma_far;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> ATA = A.transpose() * A;
    Eigen::Vector3d ATb = A.transpose() * b;
    depth_noise_params = ATA.ldlt().solve(ATb);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldScanSigma,
        &scan_sigma);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldRangeSizePerSubmap,
        &range_size_per_submap);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldMaxSubmapSize,
        &max_submap_size);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldMinMonitorScore,
        &min_monitor_score);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldRelocAcceptScore,
        &reloc_accept_score);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldRealtimeUpdateMap,
        &realtime_update_map);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldTuningMode,
        &tuning_mode);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldDoOutlierRejection,
        &do_outlier_rejection);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldUseDepth,
        &use_depth);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFixDepth,
        &fix_depth);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldOutlierRejectionScale,
        &outlier_rejection_scale);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldVocTrainMode,
        &voc_train_mode);
    
    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFeatureTrackingTestMode,
        &feature_tracking_test_mode);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldSaveColmapModel,
        &save_colmap_model);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldUseIMU,
        &use_imu);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldSigmaGyro,
        &sigma_gyro);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldSigmaAcc,
        &sigma_acc);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldSigmaBg,
        &sigma_bg);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldSigmaBa,
        &sigma_ba);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldUseScanMatching,
        &use_scan_matching);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldUseScanPointCloud,
        &use_scan_pointcloud);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldResolution,
        &resolution);

    // Visual config.
    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFeatureDetectorType,
        &feature_detector_type);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldUseGrids,
        &use_grids);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldNumFeatureToDetect,
        &num_feature_to_detect);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldNumGridsHorizontal,
        &num_grids_horizontal);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldNumGridsVertical,
        &num_grids_vertical);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldNonMaxSuppressionRadius,
        &non_max_suppression_radius);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFreakOrientationNormalized,
        &freak_orientation_normalized);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFreakScaleNormalized,
        &freak_scale_normalized);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFreakPatternScale,
        &freak_pattern_scale);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFreakNumOctaves,
        &freak_num_octaves);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFastScaleFactor,
        &fast_scale_factor);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFastPyramidLevels,
        &fast_pyramid_levels);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFastPatchSize,
        &fast_patch_size);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFastScoreLowerBound,
        &fast_score_lower_bound);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFastCircleThreshold,
        &fast_circle_threshold);
    
    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldGfttQualityLevel,
        &gftt_quality_level);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldGfttMinDistance,
        &gftt_min_distance);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldGfttBlockSize,
        &gftt_block_size);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldGfttUseHarrisDetector,
        &gftt_use_harris_detector);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldGfttK,
        &gftt_k);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldLkCandidatesRatio,
        &lk_candidates_ratio);

#ifdef USE_CNN_FEATURE
    CHECK_EQ(lk_candidates_ratio, 0.0) <<
        "Can not perform lk tracking if use CNN feature, " <<
        "please set lk_candidates_ratio as 0.";
#endif

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldLkMaxStatusTrackLength,
        &lk_max_status_track_length);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldLkTrackDetectedThreshold,
        &lk_track_detected_threshold);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldLkWindowSize,
        &lk_window_size);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldLkMaxPyramidLevels,
        &lk_max_pyramid_levels);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldLkMinEigenvalueThreshold,
        &lk_min_eigenvalue_threshold);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldLkSmallSearchDistancePx,
        &lk_small_search_distance_px);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldLkLargeSearchDistancePx,
        &lk_large_search_distance_px);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldKeypointUncertaintyPx,
        &keypoint_uncertainty_px);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldMinDistanceToImageBorderPx,
        &min_distance_to_image_border_px);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldRansacOutlierPercentage,
        &ransac_outlier_percentage);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldRansacSuccessProbability,
        &ransac_success_probability);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldRansacAngleTolerance,
        &ransac_angle_tolerance);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldRansacSamplePointSize,
        &ransac_sample_point_size);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldLoopFrequency,
        &loop_frequency);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldScanLoopInterDistance,
        &scan_loop_inter_distance);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldScoringFunction,
        &scoring_function);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldMinImageTimeSconds,
        &min_image_time_seconds);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldMinVerifyMatchesNum,
        &min_verify_matches_num);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldFractionBestScores,
        &fraction_best_scores);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldNumNeighbors,
        &num_neighbors);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldNumWordsForNnSearch,
        &num_words_for_nn_search);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldMinInlierCount,
        &min_inlier_count);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldMaxKnnRadius,
        &max_knn_radius);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldLocalSearchRadius,
        &local_search_radius);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldLocalSearchAngles,
        &local_search_angles);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldHardwareThreads,
        &hardware_threads);

    SetValueBasedOnYamlKey(config_node_,
        kYamlFieldPnpNumRansacIters,
        &pnp_num_ransac_iters);

    if (load_option_ == LOAD_AND_PRINT_GLOG) {
        PrintConfigToGlog();
    }
}

void SlamConfig::PrintConfigToGlog() const {
    LOG(INFO) << "Print SLAM Config: ";
    LOG(INFO) << "---------------common config---------------";
    LOG(INFO) << "online" << ": "
              << online;
    LOG(INFO) << kYamlFieldWinsdowSize << ": "
              << window_size;
    LOG(INFO) << kYamlFieldMainSensorType << ": "
              << main_sensor_type;
    LOG(INFO) << kYamlFieldDataPath << ": "
              << data_path;
    LOG(INFO) << kYamlFieldCalibYamlConfig << ": "
              << calib_yaml_config;
    LOG(INFO) << kYamlFieldMapPath << ": "
              << map_path;
    LOG(INFO) << kYamlFieldLogPath << ": "
              << log_path;
    LOG(INFO) << kYamlFieldFeatureTrackingTestPath << ": "
              << feature_tracking_test_path;
    LOG(INFO) << kYamlFieldMaskPath << ": "
              << mask_path;
    LOG(INFO) << kYamlFieldAssetsPath << ": "
              << assets_path;
    LOG(INFO) << kYamlFieldSuperPointModelFile << ": "
              << superpoint_model_file;
    LOG(INFO) << kYamlFieldDataRealtimePlaybackRate << ": "
              << data_realtime_playback_rate;
    LOG(INFO) << kYamlFieldLogTimeTable << ": "
              << log_time_table;
    LOG(INFO) << kYamlFieldMapping << ": "
              << mapping;
    LOG(INFO) << kYamlFieldTuningMode << ": "
              << tuning_mode;
    LOG(INFO) << kYamlFieldDoOctoMapping << ": "
              << do_octo_mapping;
    LOG(INFO) << kYamlFieldVocTrainMode << ": "
              << voc_train_mode;
    LOG(INFO) << kYamlFieldFeatureTrackingTestMode << ": "
              << feature_tracking_test_mode;
    LOG(INFO) << kYamlFieldSaveColmapModel << ": "
              << save_colmap_model;
    LOG(INFO) << kYamlFieldDoObstacleRemoval << ": "
              << do_obstacle_removal;
    LOG(INFO) << kYamlFieldContourLength << ": "
              << contour_length;
    LOG(INFO) << kYamlFieldDoExOnlineCalib << ": "
              << do_ex_online_calib;
    LOG(INFO) << kYamlFieldDoTimeOnlineCalib << ": "
              << do_time_online_calib;
    LOG(INFO) << "---------------vins config---------------";
    LOG(INFO) << kYamlFieldZeroVelocityPixelDiff << ": "
              << zero_velocity_pixel_diff;
    LOG(INFO) << kYamlFieldMaxFeatureSizeInOpt << ": "
              << max_feature_size_in_opt;
    LOG(INFO) << kYamlFieldOutlierRejectionScale << ": "
              << outlier_rejection_scale;
    LOG(INFO) << kYamlFieldVisualSigmaPixel << ": "
              << visual_sigma_pixel;
    LOG(INFO) << kYamlFieldDepthSigmaShort << ": "
              << depth_sigma_short;
    LOG(INFO) << kYamlFieldDepthSigmaMid << ": "
              << depth_sigma_mid;
    LOG(INFO) << kYamlFieldDepthSigmaFar << ": "
              << depth_sigma_far;
    LOG(INFO) << "depth_noise_params: "
              << depth_noise_params.transpose();
    LOG(INFO) << kYamlFieldMinMonitorScore << ": "
              << min_monitor_score;
    LOG(INFO) << kYamlFieldRealtimeUpdateMap << ": "
              << realtime_update_map;
    LOG(INFO) << kYamlFieldDoOutlierRejection << ": "
              << do_outlier_rejection;
    LOG(INFO) << kYamlFieldUseDepth << ": "
              << use_depth;
    LOG(INFO) << kYamlFieldFixDepth << ": "
              << fix_depth;
    LOG(INFO) << kYamlFieldOutlierRejectionScale << ": "
              << outlier_rejection_scale;
    LOG(INFO) << "---------------IMU config---------------";    
    LOG(INFO) << kYamlFieldUseIMU << ": "
              << use_imu;      
    LOG(INFO) << kYamlFieldSigmaGyro << ": "
              << sigma_gyro;
    LOG(INFO) << kYamlFieldSigmaAcc << ": "
              << sigma_acc;
    LOG(INFO) << kYamlFieldSigmaBg << ": "
              << sigma_bg;
    LOG(INFO) << kYamlFieldSigmaBa << ": "
              << sigma_ba;
    LOG(INFO) << "---------------Scan config---------------";
    LOG(INFO) << kYamlFieldUseScanMatching << ": "
              << use_scan_matching;
    LOG(INFO) << kYamlFieldUseScanPointCloud << ": "
              << use_scan_pointcloud;
    LOG(INFO) << kYamlFieldResolution << ": "
              << resolution;
    LOG(INFO) << kYamlFieldScanSigma << ": "
              << scan_sigma;
    LOG(INFO) << kYamlFieldRangeSizePerSubmap << ": "
              << range_size_per_submap;
    LOG(INFO) << kYamlFieldMaxSubmapSize << ": "
              << max_submap_size;
    LOG(INFO) << kYamlFieldRelocAcceptScore << ": "
              << reloc_accept_score;
    LOG(INFO) << "---------------visual config---------------";
    LOG(INFO) << kYamlFieldFeatureDetectorType << ": "
              << feature_detector_type;
    LOG(INFO) << kYamlFieldUseGrids << ": "
              << use_grids;
    LOG(INFO) << kYamlFieldNumFeatureToDetect << ": "
              << num_feature_to_detect;
    LOG(INFO) << kYamlFieldNumGridsHorizontal << ": "
              << num_grids_horizontal;
    LOG(INFO) << kYamlFieldNumGridsVertical << ": "
              << num_grids_vertical;
    LOG(INFO) << kYamlFieldNonMaxSuppressionRadius << ": "
              << non_max_suppression_radius;
    LOG(INFO) << kYamlFieldFreakOrientationNormalized << ": "
              << freak_orientation_normalized;
    LOG(INFO) << kYamlFieldFreakScaleNormalized << ": "
              << freak_scale_normalized;
    LOG(INFO) << kYamlFieldFreakPatternScale << ": "
              << freak_pattern_scale;
    LOG(INFO) << kYamlFieldFreakNumOctaves << ": "
              << freak_num_octaves;
    LOG(INFO) << kYamlFieldFastScaleFactor << ": "
              << fast_scale_factor;
    LOG(INFO) << kYamlFieldFastPyramidLevels << ": "
              << fast_pyramid_levels;
    LOG(INFO) << kYamlFieldFastPatchSize << ": "
              << fast_patch_size;
    LOG(INFO) << kYamlFieldFastScoreLowerBound << ": "
              << fast_score_lower_bound;
    LOG(INFO) << kYamlFieldFastCircleThreshold << ": "
              << fast_circle_threshold;
    LOG(INFO) << kYamlFieldGfttQualityLevel << ": "
              << gftt_quality_level;
    LOG(INFO) << kYamlFieldGfttMinDistance << ": "
              << gftt_min_distance;
    LOG(INFO) << kYamlFieldGfttBlockSize << ": "
              << gftt_block_size;
    LOG(INFO) << kYamlFieldGfttUseHarrisDetector << ": "
              << gftt_use_harris_detector;
    LOG(INFO) << kYamlFieldGfttK << ": "
              << gftt_k;
    LOG(INFO) << kYamlFieldLkCandidatesRatio << ": "
              << lk_candidates_ratio;
    LOG(INFO) << kYamlFieldLkMaxStatusTrackLength << ": "
              << lk_max_status_track_length;
    LOG(INFO) << kYamlFieldLkTrackDetectedThreshold << ": "
              << lk_track_detected_threshold;
    LOG(INFO) << kYamlFieldLkWindowSize << ": "
              << lk_window_size;
    LOG(INFO) << kYamlFieldLkMaxPyramidLevels << ": "
              << lk_max_pyramid_levels;
    LOG(INFO) << kYamlFieldLkMinEigenvalueThreshold << ": "
              << lk_min_eigenvalue_threshold;
    LOG(INFO) << kYamlFieldLkSmallSearchDistancePx << ": "
              << lk_small_search_distance_px;
    LOG(INFO) << kYamlFieldLkLargeSearchDistancePx << ": "
              << lk_large_search_distance_px;
    LOG(INFO) << kYamlFieldKeypointUncertaintyPx << ": "
              << keypoint_uncertainty_px;
    LOG(INFO) << kYamlFieldMinDistanceToImageBorderPx << ": "
              << min_distance_to_image_border_px;
    LOG(INFO) << kYamlFieldRansacOutlierPercentage << ": "
              << ransac_outlier_percentage;
    LOG(INFO) << kYamlFieldRansacSuccessProbability << ": "
              << ransac_success_probability;
    LOG(INFO) << kYamlFieldRansacSamplePointSize << ": "
              << ransac_sample_point_size;
    LOG(INFO) << kYamlFieldRansacAngleTolerance << ": "
              << ransac_angle_tolerance;
    LOG(INFO) << "------------loop closure config------------";
    LOG(INFO) << kYamlFieldLoopFrequency << ": "
              << loop_frequency;
    LOG(INFO) << kYamlFieldScanLoopInterDistance << ": "
              << scan_loop_inter_distance;
    LOG(INFO) << kYamlFieldScoringFunction << ": "
              << scoring_function;
    LOG(INFO) << kYamlFieldMinImageTimeSconds << ": "
              << min_image_time_seconds;
    LOG(INFO) << kYamlFieldMinVerifyMatchesNum << ": "
              << min_verify_matches_num;
    LOG(INFO) << kYamlFieldFractionBestScores << ": "
              << fraction_best_scores;
    LOG(INFO) << kYamlFieldNumNeighbors << ": "
              << num_neighbors;
    LOG(INFO) << kYamlFieldNumWordsForNnSearch << ": "
              << num_words_for_nn_search;
    LOG(INFO) << kYamlFieldMinInlierCount << ": "
              << min_inlier_count;
    LOG(INFO) << kYamlFieldMaxKnnRadius << ": "
              << max_knn_radius;
    LOG(INFO) << kYamlFieldLocalSearchRadius << ": "
              << local_search_radius;
    LOG(INFO) << kYamlFieldLocalSearchAngles << ": "
              << local_search_angles;
    LOG(INFO) << kYamlFieldHardwareThreads << ": "
              << hardware_threads;
    LOG(INFO) << kYamlFieldPnpNumRansacIters << ": "
              << pnp_num_ransac_iters;
}

}
