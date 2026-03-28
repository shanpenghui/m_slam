#include "summary_map/map_loader.h"

#include "file_common/file_system_tools.h"

namespace loop_closure {
MapLoader::MapLoader(const std::string &load_path) {
    int ret = sqlite3_open(load_path.c_str(), &database_);
    if (ret) {
        LOG(FATAL) << "Can't open database: " << sqlite3_errmsg(database_);
    } else {
        LOG(INFO) << "Opened database successfully";
    }
}

MapLoader::~MapLoader() {
    sqlite3_close(database_);
}

void MapLoader::ReadObservers(std::vector<int>* vertex_ids_ptr,
                              Eigen::Matrix<double, 7, Eigen::Dynamic>* T_OtoMs_ptr) {
    std::vector<int>& vertex_ids = *CHECK_NOTNULL(vertex_ids_ptr);
    Eigen::Matrix<double, 7, Eigen::Dynamic>& T_OtoMs = *CHECK_NOTNULL(T_OtoMs_ptr);

    const std::string sql_query_observer =
            "SELECT "
            + summary_map::column_name_observer(
                summary_map::table_observer::ID) + ","
            + summary_map::column_name_observer(
                summary_map::table_observer::POSE) +
            " FROM "
            + summary_map::table_name(
                summary_map::database_tables::OBSERVER);

    sqlite3_stmt* pstmt_observer = nullptr;
    CHECK(common::Sqlite3_prepare_v2(sql_query_observer,
                             &pstmt_observer,
                             database_));
    CHECK(common::Sqlite3_reset(pstmt_observer));
    common::EigenVector7dVec poses;
    // 0--ID
    // 1--pose
    while (sqlite3_step(pstmt_observer) == SQLITE_ROW) {
        const int64_t id = sqlite3_column_int64(pstmt_observer, 0);
        const void* blob_pose = sqlite3_column_blob(pstmt_observer, 1);
        const int size_pose = sqlite3_column_bytes(pstmt_observer, 1);
        CHECK_EQ(size_pose, 56);
        Eigen::Matrix<double, 7 ,1> pose(reinterpret_cast<const double*>(blob_pose));

        vertex_ids.push_back(static_cast<int>(id));
        poses.push_back(pose);
    }
    CHECK(common::Sqlite3_finalize(&pstmt_observer));
    T_OtoMs = Eigen::Map<Eigen::Matrix<double, 7, Eigen::Dynamic, Eigen::ColMajor>>(
                poses[0].data(), 7, poses.size());
}

void MapLoader::ReadLandmarks(std::vector<int>* track_ids_ptr,
                                Eigen::Matrix3Xd* p_LinMs_ptr) {
    std::vector<int>& track_ids = *CHECK_NOTNULL(track_ids_ptr);
    Eigen::Matrix3Xd& p_LinMs = *CHECK_NOTNULL(p_LinMs_ptr);

    const std::string sql_query_landmark =
            "SELECT "
            + summary_map::column_name_landmark(
                summary_map::table_landmark::ID) + ","
            + summary_map::column_name_landmark(
                summary_map::table_landmark::POSITION) +
            " FROM "
            + summary_map::table_name(
                summary_map::database_tables::LANDMARK);

    sqlite3_stmt* pstmt_landmark = nullptr;
    CHECK(common::Sqlite3_prepare_v2(sql_query_landmark,
                             &pstmt_landmark,
                             database_));
    CHECK(common::Sqlite3_reset(pstmt_landmark));
    common::EigenVector3dVec positions;
    // 0--ID
    // 1--position
    while (sqlite3_step(pstmt_landmark) == SQLITE_ROW) {
        const int64_t id = sqlite3_column_int64(pstmt_landmark, 0);
        const void* blob_position = sqlite3_column_blob(pstmt_landmark, 1);
        const int size_position = sqlite3_column_bytes(pstmt_landmark, 1);
        CHECK_EQ(size_position, 24);
        Eigen::Vector3d position(reinterpret_cast<const double*>(blob_position));

        track_ids.push_back(static_cast<int>(id));
        positions.push_back(position);
    }
    CHECK(common::Sqlite3_finalize(&pstmt_landmark));
    p_LinMs = Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::ColMajor>>(
                positions[0].data(), 3, positions.size());
}

void MapLoader::ReadObservations(std::vector<int>* obs_vertex_ids_ptr,
                                  std::vector<int>* obs_cam_ids_ptr,
                                  std::vector<int>* obs_track_ids_ptr,
                                  Eigen::Matrix2Xd* key_points_ptr,
                                  common::DescriptorsMatF32* descriptors_ptr) {
    std::vector<int>& obs_vertex_ids = *CHECK_NOTNULL(obs_vertex_ids_ptr);
    std::vector<int>& obs_cam_ids = *CHECK_NOTNULL(obs_cam_ids_ptr);
    std::vector<int>& obs_track_ids = *CHECK_NOTNULL(obs_track_ids_ptr);
    Eigen::Matrix2Xd& key_points = *CHECK_NOTNULL(key_points_ptr);
    common::DescriptorsMatF32& descriptors = *CHECK_NOTNULL(descriptors_ptr);

    const std::string sql_query_observation =
            "SELECT "
            + summary_map::column_name_observation(
                summary_map::table_observation::OBSERVER_ID) + ","
            + summary_map::column_name_observation(
                summary_map::table_observation::CAMERA_ID) + ","
            + summary_map::column_name_observation(
                summary_map::table_observation::LANDMARK_ID) + ","
            + summary_map::column_name_observation(
                summary_map::table_observation::KEY_POINT) + ","
            + summary_map::column_name_observation(
                summary_map::table_observation::DESCRIPTORS) +
            " FROM "
            + summary_map::table_name(
                summary_map::database_tables::OBSERVATION);

    sqlite3_stmt* pstmt_observation = nullptr;
    CHECK(common::Sqlite3_prepare_v2(sql_query_observation,
                                     &pstmt_observation,
                                     database_));
    CHECK(common::Sqlite3_reset(pstmt_observation));

    common::EigenVector2dVec tmp_keypoints;
    std::vector<common::DescriptorsF32> tmp_descriptors;
    // 0--vertex_id
    // 1--camera_id
    // 2--track_id
    // 3--key_point
    // 4--descriptors
    while (sqlite3_step(pstmt_observation) == SQLITE_ROW) {
        const int64_t vertex_id = sqlite3_column_int64(pstmt_observation, 0);
        const int64_t cam_id = sqlite3_column_int64(pstmt_observation, 1);
        const int64_t track_id = sqlite3_column_int64(pstmt_observation, 2);
        const void* blob_keypoint = sqlite3_column_blob(pstmt_observation, 3);
        const int size_keypoint = sqlite3_column_bytes(pstmt_observation, 3);
        CHECK_EQ(size_keypoint, 8 * 2);
        const void* blob_descriptors = sqlite3_column_blob(pstmt_observation, 4);
        const int size_descriptors = sqlite3_column_bytes(pstmt_observation, 4);
        CHECK_EQ(size_descriptors, 4 * loop_closure::kDescriptorDim);

        obs_vertex_ids.push_back(static_cast<int>(vertex_id));
        obs_cam_ids.push_back(static_cast<int>(cam_id));
        obs_track_ids.push_back(static_cast<int>(track_id));
        tmp_keypoints.push_back(Eigen::Map<Eigen::Vector2d>(
                                            const_cast<double*>(reinterpret_cast<const double*>(blob_keypoint))));
        tmp_descriptors.push_back(Eigen::Map<common::DescriptorsF32>(
                                            const_cast<float*>(reinterpret_cast<const float*>(blob_descriptors)),
                                            loop_closure::kDescriptorDim,
                                            1));
    }
    CHECK(common::Sqlite3_finalize(&pstmt_observation));
    descriptors.resize(loop_closure::kDescriptorDim, tmp_descriptors.size());
    for (size_t i = 0u; i < tmp_descriptors.size(); ++i) {
        descriptors.col(i) = tmp_descriptors[i];
    }
    key_points.resize(Eigen::NoChange, tmp_keypoints.size());
    for (size_t i = 0u; i < tmp_keypoints.size(); ++i) {
        key_points.col(i) = tmp_keypoints[i];
    }
}

void MapLoader::LoadMap(SummaryMap *summary_map_ptr) {
    SummaryMap& summary_map = *CHECK_NOTNULL(summary_map_ptr);
    std::vector<int> vertex_ids;
    Eigen::Matrix<double, 7, Eigen::Dynamic> T_OtoMs;
    ReadObservers(&vertex_ids, &T_OtoMs);
    CHECK_EQ(vertex_ids.size(), static_cast<size_t>(T_OtoMs.cols()));
    summary_map.SetObservers(vertex_ids, T_OtoMs);

    std::vector<int> track_ids;
    Eigen::Matrix3Xd p_LinMs;
    ReadLandmarks(&track_ids, &p_LinMs);
    CHECK_EQ(track_ids.size(), static_cast<size_t>(p_LinMs.cols()));
    summary_map.SetLandmarks(track_ids, p_LinMs);

    std::vector<int> obs_vertex_ids;
    std::vector<int> obs_cam_ids;
    std::vector<int> obs_landmark_ids;
    Eigen::Matrix2Xd key_points;
    common::DescriptorsMatF32 descriptors;
    ReadObservations(&obs_vertex_ids,
                      &obs_cam_ids,
                      &obs_landmark_ids,
                      &key_points,
                      &descriptors);
    summary_map.SetObservations(obs_vertex_ids,
                                obs_cam_ids,
                                obs_landmark_ids,
                                key_points,
                                descriptors);

    LOG(INFO) << "Visual map loading success.";
}
}
