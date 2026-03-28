#ifndef MVINS_LASER_CONFIG_H_
#define MVINS_LASER_CONFIG_H_

#include "cfg_common/config_base.h"
#include "yaml_common/yaml_util.h"

namespace common{
    constexpr char kYamlFieldLoadOption[] =
            "load_option";
    constexpr char kYamlFieldDataPath[] =
            "data_path";
    constexpr char kYamlFieldDataRealtimePlaybackRate[] =
            "data_realtime_playback_rate";
    constexpr char kYamlFieldLogPath[] =
            "log_path";
    constexpr char kYamlFieldLogTimeTable[] =
            "log_time_table";
    constexpr char kYamlFieldMapPath[] =
            "map_path";
    constexpr char kYamlFieldMapping[] =
            "mapping";
    constexpr char kYamlFieldCalibYamlConfig[] =
            "calib_yaml_config";
    constexpr char kYamlFieldAssetsPath[] =
            "assets_path";
    constexpr char kYamlFieldSuperPointModelFile[] =
            "superpoint_model_file";
    constexpr char kYamlFieldMaskPath[] =
            "mask_path";
    constexpr char kYamlFieldDoOctoMapping[] =
            "do_octo_mapping";
    const char kYamlFieldDoObstacleRemoval[] =
            "do_obstacle_removal";
    const char kYamlFieldContourLength[] =
            "contour_length";
    const char kYamlFieldDoExOnlineCalib[] =
            "do_ex_online_calib";
    const char kYamlFieldDoTimeOnlineCalib[] =
            "do_time_online_calib";
    constexpr char kYamlFieldUseIMU[] =
            "use_imu";
    constexpr char kYamlFieldSigmaGyro[] =
            "sigma_gyro";
    constexpr char kYamlFieldSigmaAcc[] =
            "sigma_acc";
    constexpr char kYamlFieldSigmaBg[] =
            "sigma_bg";
    constexpr char kYamlFieldSigmaBa[] =
            "sigma_ba";
    constexpr char kYamlFieldWinsdowSize[] =
            "window_size";
    constexpr char kYamlFieldMainSensorType[] =
            "main_sensor_type";
    constexpr char kYamlFieldZeroVelocityPixelDiff[] =
            "zero_velocity_pixel_diff";
    constexpr char kYamlFieldMaxFeatureSizeInOpt[] =
            "max_feature_size_in_opt";
    constexpr char kYamlFieldVisualSigmaPixel[] =
            "visual_sigma_pixel";
    constexpr char kYamlFieldDepthSigmaShort[] =
            "depth_sigma_short";
    constexpr char kYamlFieldDepthSigmaMid[] =
            "depth_sigma_mid";
    constexpr char kYamlFieldDepthSigmaFar[] =
            "depth_sigma_far";
    constexpr char kYamlFieldScanSigma[] =
            "scan_sigma";
    constexpr char kYamlFieldRangeSizePerSubmap[] =
            "range_size_per_submap";
    constexpr char kYamlFieldMaxSubmapSize[] =
            "max_submap_size";
    constexpr char kYamlFieldMinMonitorScore[] =
            "min_monitor_score";
    constexpr char kYamlFieldRelocAcceptScore[] =
            "reloc_accept_score";
    constexpr char kYamlFieldRealtimeUpdateMap[] =
            "realtime_update_map";
    constexpr char kYamlFieldTuningMode[] =
            "tuning_mode";
    constexpr char kYamlFieldDoOutlierRejection[] =
            "do_outlier_rejection";
    constexpr char kYamlFieldUseDepth[] =
            "use_depth";
    constexpr char kYamlFieldFixDepth[] =
            "fix_depth";
    constexpr char kYamlFieldOutlierRejectionScale[] =
            "outlier_rejection_scale";
    constexpr char kYamlFieldUseScanMatching[] =
            "use_scan_matching";
    constexpr char kYamlFieldUseScanPointCloud[] =
            "use_scan_pointcloud";
    constexpr char kYamlFieldResolution[] =
            "resolution";
    constexpr char kYamlFieldFeatureDetectorType[] = 
            "feature_detector_type";
    constexpr char kYamlFieldUseGrids[] =
            "use_grids";
    constexpr char kYamlFieldNumFeatureToDetect[] =
            "num_feature_to_detect";
    constexpr char kYamlFieldNumGridsHorizontal[] =
            "detection_grids_horizontal";
    constexpr char kYamlFieldNumGridsVertical[] =
            "detection_grids_vertical";
    constexpr char kYamlFieldNonMaxSuppressionRadius[] =
            "detection_non_max_suppression_radius";
    constexpr char kYamlFieldFreakOrientationNormalized[] =
            "freak_orientation_normalized";
    constexpr char kYamlFieldFreakScaleNormalized[] =
            "freak_scale_normalized";
    constexpr char kYamlFieldFreakPatternScale[] =
            "freak_pattern_scale";
    constexpr char kYamlFieldFreakNumOctaves[] =
            "freak_num_octaves";
    constexpr char kYamlFieldFastScaleFactor[] =
            "fast_scale_factor";
    constexpr char kYamlFieldFastPyramidLevels[] =
            "fast_pyramid_levels";
    constexpr char kYamlFieldFastPatchSize[] =
            "fast_patch_size";
    constexpr char kYamlFieldFastScoreLowerBound[] =
            "fast_score_lower_bound";
    constexpr char kYamlFieldFastCircleThreshold[] =
            "fast_circle_threshold";
    constexpr char kYamlFieldGfttQualityLevel[] = 
            "gftt_quality_level";
    constexpr char kYamlFieldGfttMinDistance[] = 
            "gftt_min_distance";
    constexpr char kYamlFieldGfttBlockSize[] = 
            "gftt_block_size";
    constexpr char kYamlFieldGfttUseHarrisDetector[] =
            "gftt_use_harris_detector";
    constexpr char kYamlFieldGfttK[] =
            "gftt_k";
    const char kYamlFieldLkCandidatesRatio[] =
            "lk_candidates_ratio";
    const char kYamlFieldLkMaxStatusTrackLength[] =
            "lk_max_status_track_length";
    const char kYamlFieldLkTrackDetectedThreshold[] =
            "lk_track_detected_threshold";
    const char kYamlFieldLkWindowSize[] =
            "lk_window_size";
    const char kYamlFieldLkMaxPyramidLevels[] =
            "lk_max_pyramid_levels";
    const char kYamlFieldLkMinEigenvalueThreshold[] =
            "lk_min_eigenvalue_threshold";
    const char kYamlFieldLkSmallSearchDistancePx[] =
            "lk_small_search_distance_px";
    const char kYamlFieldLkLargeSearchDistancePx[] =
            "lk_large_search_distance_px";
    const char kYamlFieldKeypointUncertaintyPx[] =
            "keypoint_uncertainty_px";
    const char kYamlFieldMinDistanceToImageBorderPx[] =
            "min_distance_to_image_border_px";
    const char kYamlFieldRansacOutlierPercentage[] =
            "ransac_outlier_percentage";
    const char kYamlFieldRansacSuccessProbability[] =
            "ransac_success_probability";
    const char kYamlFieldRansacAngleTolerance[] =
            "ransac_angle_tolerance";
    const char kYamlFieldRansacSamplePointSize[] =
            "ransac_sample_point_size";
    constexpr char kYamlFieldLoopFrequency[] =
            "loop_frequency";
    constexpr char kYamlFieldScanLoopInterDistance[] =
            "scan_loop_inter_distance";
    const char kYamlFieldScoringFunction[] =
            "scoring_function";
    const char kYamlFieldMinImageTimeSconds[] =
            "min_image_time_seconds";
    const char kYamlFieldMinVerifyMatchesNum[] =
            "min_verify_matches_num";
    const char kYamlFieldFractionBestScores[] =
            "fraction_best_scores";
    const char kYamlFieldNumNeighbors[] =
            "num_neighbors";
    const char kYamlFieldNumWordsForNnSearch[] =
            "num_words_for_nn_search";
    const char kYamlFieldMinInlierCount[] =
            "min_inlier_count";
    const char kYamlFieldMaxKnnRadius[] =
            "max_knn_radius";
    const char kYamlFieldLocalSearchRadius[] =
            "local_search_radius";
    const char kYamlFieldLocalSearchAngles[] =
            "local_search_angles";
    const char kYamlFieldHardwareThreads[] =
            "hardware_threads";
    const char kYamlFieldPnpNumRansacIters[] =
            "pnp_num_ransac_iters";

    class SlamConfig : public common::ConfigBase {
    public:
        //! Default constructor.
        SlamConfig() = default;

        //! Constructor with config file path.
        explicit SlamConfig(const std::string& file_path,
                            const bool online_mode);

        virtual ~SlamConfig() = default;

        //! Outputters.
        void PrintConfigToGlog() const override;
        /***************************************************************/
        // Wether to deploy robotic applications online.
        bool online = false;
        
        // Path to data.
        std::string data_path = "";

        // Playback rate of the data. Real-time corresponds to 1.0.
        double data_realtime_playback_rate = 1.0;

        // Path to save log files.
        std::string log_path = "../log/";

        // Path to save feature tracking test images.

        // Whether to use time table for efficiency check.
        bool log_time_table = true;

        // Path to a localization map.
        std::string map_path = "../map/";

        // Wether to build map.
        bool mapping = false;

        // The yaml of calib file.
        std::string calib_yaml_config = "../cfg/calib.yaml";

        // The asset file path for loading.
        std::string assets_path = "../assets";

        // SuperPoint onnx file name.
        std::string superpoint_model_file = "../superpoint_240_320_fp16_int8_pruned.rknn";
        
        // The image mask for feature detecting.
        std::string mask_path = "../cfg/mask.png";

        // If perform octo mapping.
        bool do_octo_mapping = false;

        // If run vocabulary training mode.

        // If run feature tracking testing mode.


        // If remove small obstacles when create map show.
        bool do_obstacle_removal = true;
        
        // The contour length of small obstacles
        int contour_length = 4;

        // If do extrinsics online calibration.
        bool do_ex_online_calib = false;

        // If do time drift online calibration.
        bool do_time_online_calib = false;

        // Window size in sliding window optimization.
        int window_size = 8;

        // The main sensor type.
        std::string main_sensor_type = "scan_visual";

        // The map tuning mode.
        std::string tuning_mode = "off";

        /***************************************************************/
        // IMU noise system.

        // If use imu measurements for fusion.
        bool use_imu = false;

        // Gyroscope white noise (rad/s/sqrt(hz))
        double sigma_gyro = 0.004;

        // Accelerometer white noise (m/s^2/sqrt(hz))
        double sigma_acc = 0.08;

        // Gyroscope random walk (rad/s^2/sqrt(hz))
        double sigma_bg = 2.0e-6;

        // Accelerometer random walk (m/s^3/sqrt(hz))
        double sigma_ba = 4.0e-5;

        /***************************************************************/
        // Scan paramenters:

        // If perform fast correlative scan mathcing in odometry.
        bool use_scan_matching = true;

        // If use scan pointcloud.
        bool use_scan_pointcloud = false;

        // The resolution of occupancy map.
        double resolution = 0.05;

        // The scan noise sigma.
        double scan_sigma = 0.01;

        // The max range data size in a submap.
        int range_size_per_submap = 40;

        // The max submap size.
        int max_submap_size = 2;

        // The min score in scan reloc for monitor count.
        double min_monitor_score = 0.4;

        // The accpet score in reloc init.
        double reloc_accept_score = 0.6;

        // If update global map in real time.
        bool realtime_update_map = false;

        /***************************************************************/
        // Common visual parameters:

        // Whether use GFTT or FAST dector
        std::string feature_detector_type = "fast";

        // If use grid in feature detection.
        bool use_grids = true;

        // The zero velocity checking threshold of feature tracking diff.
        double zero_velocity_pixel_diff = 0.01;

        // Max feature size in per optimization.
        int max_feature_size_in_opt = 50;

        // The visual cost noise (pixel).
        double visual_sigma_pixel = 0.8;

        // The depth cost noise params (meter);
        Eigen::Vector3d depth_noise_params = Eigen::Vector3d::Zero();        

        // If perform outlier rejection.
        bool do_outlier_rejection = false;

        // If use RGB-D depth measurements for fusion.
        bool use_depth = true;

        // If fix depth estimate in optimization.
        bool fix_depth = true;

        // Outlier rejection scale threshold of pixel.
        double outlier_rejection_scale = 6.0;

        // Number of feature size to detecting in per-frame.
        int num_feature_to_detect = 200;

        // Number of grids in horizontal direction for feature detection.
        int num_grids_horizontal = 4;

        // Number of grids in vertical direction for feature detection.
        int num_grids_vertical = 3;

        // Ratio to detect more features so that the total number
        // will be closer to desired max after eliminating weak features.
        double feature_num_amplify_ratio = 2.5;

        // Min distance between any two of the detected corners, in pixels.
        int non_max_suppression_radius = 10;

        /***************************************************************/

        /***************************************************************/
        // FAST parameters:

        // Pyramid decimation ratio, greater than 1. scaleFactor==2 means
        // the classical pyramid, where each next level has 4x less pixels
        // than the previous, but such a big scale factor will degrade
        // feature matching scores dramatically. On the other hand, too
        // close to 1 scale factor will mean that to cover certain scale
        // range you will need more pyramid levels and so the speed will suffer.
        double fast_scale_factor = 1.2;

        // The number of pyramid levels. Higher number of pyramids will
        // increase scale invariance properties but will also lead
        // accumulations of keypoints in hotspots. Feature detection can
        // slow down considerably with higher numbers of pyramid levels.
        int fast_pyramid_levels = 1;

        // Size of the patch used by the oriented BRIEF descriptor.
        // On smaller pyramid layers the perceived image area covered by
        // a feature will be larger.
        int fast_patch_size = 10;

        // Keypoints with a score lower than this value will get removed.
        // This can be useful to remove keypoints of low quality or keypoints
        // that are associated with image noise.
        double fast_score_lower_bound = 1e-7;

        // Threshold on difference between intensity of the central pixel and
        // pixels of a circle around this pixel.
        int fast_circle_threshold = 10;
        /***************************************************************/

        /***************************************************************/
        // GFTT parameters

        // Quality level of corners found by GFTT. Default set to 0.01. Changes from
        // 0 to 1. This parameter determines the quality of chosen corners. The higher
        // this parameter, the higher quality corners you get.
        double gftt_quality_level = 0.01;

        // Minimal distance between any two corner found by GFTT. Default set to 1.
        int gftt_min_distance = 1;

        // The size of the neighborhood to use when calculating the corner response function.
        // Normally set to 3 or other odd numbers less than 3.
        int gftt_block_size = 3;

        // Whether to use harris corner detector instead. 
        // If false, use Shi-Tomas corner detector.
        bool gftt_use_harris_detector = false;
        
        // The free parameter of harris detector. Only works when using harris detector.
        double gftt_k = 0.04;
        /***************************************************************/

        /***************************************************************/
        // FREAK parameters

        // Enable orientation normalization for FREAK feature detection.
        bool freak_orientation_normalized = false;

        // Enable scale normalization for FREAK feature detection.
        bool freak_scale_normalized = true;

        // Scaling of the description pattern for FREAK feature detection.
        float freak_pattern_scale = 22.0;

        // Number of octaves covered by the detected keypoints.
        int freak_num_octaves = 3;
        /***************************************************************/

        /***************************************************************/
        // Tracking parameters

        // This ratio defines the number of
        // unmatched (from frame k to (k+1)) keypoints that will be tracked with
        // the lk tracker to the next frame. If we detect N keypoints in frame (k+1),
        // we track at most 'N times this ratio' keypoints to frame (k+1) with the
        // lk tracker. A value of 0 means that pure tracking by matching descriptors
        // will be used.
        double lk_candidates_ratio = 0.4;

        // Status track length is the
        // track length since the status of the keypoint has changed (e.g. from lk
        // tracked to detected or the reverse). The lk tracker will not track
        // keypoints with longer status track length than this value.
        int lk_max_status_track_length = 10;

        // Threshold that defines
        // how many times a detected feature has to be matched to be deemed
        // worthy to be tracked by the LK-tracker. A value of 1 means that it has
        // to be at least detected twice and matched once.
        int lk_track_detected_threshold = 1;

        // Size of the search window at each pyramid level.
        int lk_window_size = 21;

        // If set to 0, pyramids are not
        // used (single level), if set to 1, two levels are used, and so on.
        // If pyramids are passed to the input then the algorithm will use as many
        // levels as possible but not more than this threshold.

        int lk_max_pyramid_levels = 1;
        // The algorithm
        // calculates the minimum eigenvalue of a 2x2 normal matrix of optical flow
        // equations, divided by number of pixels in a window. If this value is less
        // than this threshold, the corresponding feature is filtered out and its
        // flow is not processed, so it allows to remove bad points and get a
        // performance boost.
        double lk_min_eigenvalue_threshold = 0.001;

        // Small search rectangle size for keypoint matches.
        int lk_small_search_distance_px = 10;

        // Large search rectangle size for keypoint matches.
        // Only used if small search was unsuccessful.
        int lk_large_search_distance_px = 20;

        // Keypoint uncertainty.
        double keypoint_uncertainty_px = 0.8;

        // Enforce a minimal distance to the image border for feature tracking.
        // Hence, features can be detected close to the image border but
        // might not be tracked if the predicted location of the keypoint in the
        // next frame is closer to the image border than this value.
        int min_distance_to_image_border_px = 30;

        // Ransac outlier percentage.
        double ransac_outlier_percentage = 0.5;

        // Expected RANSAC success probability.
        double ransac_success_probability = 0.995;

        // RANSAC angle tolerance, should be 0 to 90 degrees.
        double ransac_angle_tolerance = 45.0;

        // N-Point-RANSAC sample point size.
        double ransac_sample_point_size = 2.;
        /***************************************************************/

        /***************************************************************/
        // Loop closure frequency.
        double loop_frequency = 2.0;

        // Scan loop closure international distance between submap.
        double scan_loop_inter_distance = 3.0;

        // Type of scoring function to be used for scoring keyframes.
        std::string scoring_function = "accumulation";

        // Minimum time between matching images to allow a loop closure.
        double min_image_time_seconds = 10.0;

        // The minimum number of matches needed to verify geometry.
        int min_verify_matches_num = 10;

        // Fraction of best scoring keyframes/vertices
        // that are considered for covisibility filtering.
        float fraction_best_scores = 0.25;

        // Number of neighbors to retrieve for loop-closure. -1 auto.
        int num_neighbors = -1;

        // Number of nearest words to retrieve in the inverted index.
        int num_words_for_nn_search = 10;

        // Minimum inlier count for loop closure.
        int min_inlier_count = 10;

        // The max KNN search radius in feature Kd-tree loop closure.
        float max_knn_radius = 0.3;

        // KNN local search radius in pose searching.
        double local_search_radius = 2.0;

        // KNN local search angles in pose searching.
        double local_search_angles = 90.0;

        // Number of hardware threads to announce.
        size_t hardware_threads = 0u;
        
        // Max iters of pnp ransac.
        int pnp_num_ransac_iters = 50;
        /***************************************************************/

    private:
        // The depth noise 1 meter away from camera.
        double depth_sigma_short = 0.01;

        // The depth noise 5 meter away from camera.
        double depth_sigma_mid = 0.05;
        
        // The depth noise 10 meter away from camera.
        double depth_sigma_far = 0.20;

        //! Load configuration from config file.
        void LoadConfigFromFile() override;

        //! Yaml node to load config.
        YAML::Node config_node_;
    };

    typedef std::shared_ptr<SlamConfig> SlamConfigPtr;
}

#endif  // MVINS_LASER_CONFIG_H_
