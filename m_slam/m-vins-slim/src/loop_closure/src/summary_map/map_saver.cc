#include "summary_map/map_saver.h"

#include "data_common/constants.h"
#include "file_common/file_system_tools.h"

namespace loop_closure {
MapSaver::MapSaver(const std::string &save_path)
    : save_path_(save_path) {
    const std::string complete_map_file_name =
        common::ConcatenateFilePathFrom(
            common::getRealPath(save_path_), common::kVisualMapFileName);
    const bool exist_map = common::fileExists(complete_map_file_name);

    if (exist_map) {
        remove(complete_map_file_name.c_str());
        LOG(INFO) << "Old visual map file has been removed";
    }

    int ret = sqlite3_open(complete_map_file_name.c_str(), &database);
    if (ret) {
        LOG(FATAL) << "Cannot create database: " << sqlite3_errmsg(database);
    } else {
        LOG(INFO) << "Opened database successfully";
    }
}

MapSaver::~MapSaver() {
    sqlite3_close(database);
}

void MapSaver::CreateTable() {
    // Drop exist table observer.
    std::string drop_table_observer =
        "DROP TABLE IF EXISTS "
        + summary_map::table_name(
            summary_map::database_tables::OBSERVER);
    CHECK(common::Sqlite3_exec(drop_table_observer, database));

    // Create table observer.
    std::string create_table_observer = "CREATE TABLE "
        + summary_map::table_name(
            summary_map::database_tables::OBSERVER) +
        "("
        + summary_map::column_name_observer(
            summary_map::table_observer::ID) +
        " INTEGER PRIMARY KEY     NOT NULL,"
        + summary_map::column_name_observer(
            summary_map::table_observer::POSE) +
        " BLOB    NOT NULL);";

    CHECK(common::Sqlite3_exec(create_table_observer, database));

    // Drop exist table landmark.
    std::string drop_table_landmark =
        "DROP TABLE IF EXISTS "
        + summary_map::table_name(
            summary_map::database_tables::LANDMARK);
    CHECK(common::Sqlite3_exec(drop_table_landmark, database));

    // Create table landmark.
    std::string create_table_landmark = "CREATE TABLE "
        + summary_map::table_name(
            summary_map::database_tables::LANDMARK) +
        "("
        + summary_map::column_name_landmark(
            summary_map::table_landmark::ID) +
        " INTEGER PRIMARY KEY     NOT NULL,"
         + summary_map::column_name_landmark(
        summary_map::table_landmark::POSITION) +
        " BLOB    NOT NULL);";
    CHECK(common::Sqlite3_exec(create_table_landmark, database));

    // Drop exist table observation.
    std::string drop_table_observation =
        "DROP TABLE IF EXISTS "
        + summary_map::table_name(
            summary_map::database_tables::OBSERVATION);
    CHECK(common::Sqlite3_exec(drop_table_observation, database));

    // Create table observation.
    std::string create_table_observation = "CREATE TABLE "
        + summary_map::table_name(
            summary_map::database_tables::OBSERVATION) +
        "("
        + summary_map::column_name_observation(
            summary_map::table_observation::ID) +
        " INTEGER PRIMARY KEY    NOT NULL,"
        + summary_map::column_name_observation(
            summary_map::table_observation::OBSERVER_ID) +
        " INTEGER     NOT NULL,"
        + summary_map::column_name_observation(
            summary_map::table_observation::CAMERA_ID) +
        " INTEGER     NOT NULL,"
        + summary_map::column_name_observation(
            summary_map::table_observation::LANDMARK_ID) +
        " INTEGER     NOT NULL,"
        + summary_map::column_name_observation(
            summary_map::table_observation::KEY_POINT) +
        " BLOB        NOT NULL,"        
        + summary_map::column_name_observation(
            summary_map::table_observation::DESCRIPTORS) +
        " BLOB);";

    CHECK(common::Sqlite3_exec(create_table_observation, database));
}

void MapSaver::SaveObservers(const SummaryMap& summary_map) {
    CHECK(common::Sqlite3_exec("BEGIN;", database));
    sqlite3_stmt* stmt = nullptr;
    const std::string insert_sql = "INSERT INTO "
        + summary_map::table_name(
            summary_map::database_tables::OBSERVER)
        + " VALUES(?,?)";

    CHECK(common::Sqlite3_prepare_v2(insert_sql, &stmt, database));

    const std::unordered_map<int, size_t>& vertex_id_to_idx =
            summary_map.GetMapVertexId();
    for (const auto& iter : vertex_id_to_idx) {
        const int vertex_id = iter.first;
        const Eigen::Matrix<double, 7, 1> T_OtoM = summary_map.GetObserverPose(vertex_id);

        void* blob_pose = nullptr;
        blob_pose = reinterpret_cast<char*>(malloc(sizeof(double) * 7));
        memcpy(blob_pose, reinterpret_cast<const char*>(T_OtoM.data()),
               sizeof(double) * 7);
        CHECK(common::Sqlite3_reset(stmt));
        CHECK(common::Sqlite3_bind_int64(static_cast<int64_t>(vertex_id),
                                 1,
                                 stmt));
        CHECK(common::Sqlite3_bind_blob(blob_pose,
                                2,
                                sizeof(double) * 7,
                                stmt));
        CHECK(common::Sqlite3_step(stmt));

        free(blob_pose);
    }
    CHECK(common::Sqlite3_finalize(&stmt));
    CHECK(common::Sqlite3_exec("COMMIT;", database));
}

void MapSaver::SaveLandmarks(const SummaryMap& summary_map) {
    CHECK(common::Sqlite3_exec("BEGIN;", database));
    static sqlite3_stmt* stmt = nullptr;
    const std::string insert_sql = "INSERT INTO "
        + summary_map::table_name(
            summary_map::database_tables::LANDMARK) +
        " VALUES(?,?)";

    CHECK(common::Sqlite3_prepare_v2(insert_sql, &stmt, database));

    const std::unordered_map<int, size_t>& track_id_to_idx =
            summary_map.GetMapTrackId();
    for (const auto& iter : track_id_to_idx) {
        const int track_id = iter.first;
        const Eigen::Vector3d p_LinM = summary_map.GetLandmarkPosition(track_id);

        void* blob_position = nullptr;
        blob_position = reinterpret_cast<char*>(malloc(sizeof(double) * 3));
        memcpy(blob_position, reinterpret_cast<const char*>(p_LinM.data()),
               sizeof(double) * 3);
        CHECK(common::Sqlite3_reset(stmt));
        CHECK(common::Sqlite3_bind_int64(static_cast<int64_t>(track_id),
                                 1,
                                 stmt));
        CHECK(common::Sqlite3_bind_blob(blob_position,
                                2,
                                sizeof(double) * 3,
                                stmt));
        CHECK(common::Sqlite3_step(stmt));

        free(blob_position);
    }
    CHECK(common::Sqlite3_finalize(&stmt));
    CHECK(common::Sqlite3_exec("COMMIT;", database));
}

void MapSaver::SaveObservations(const SummaryMap& summary_map) {
    CHECK(common::Sqlite3_exec("BEGIN;", database));
    static sqlite3_stmt* stmt = nullptr;
    const std::string insert_sql = "INSERT INTO "
        + summary_map::table_name(
            summary_map::database_tables::OBSERVATION) +
        " VALUES(?,?,?,?,?,?)";
    CHECK(common::Sqlite3_prepare_v2(insert_sql, &stmt, database));

    const common::ObservationDeq& observation_all =
            summary_map.GetObservationsAll();

    for (size_t idx = 0u; idx < observation_all.size(); ++idx) {
        const common::Observation& obs = observation_all[idx];
        void* blob_keypoint = nullptr;
        common::VectorMemcpy<double>(obs.key_point, &blob_keypoint);
        void* blob_descriptors = nullptr;
        common::VectorMemcpy<float>(obs.projected_descriptors, &blob_descriptors);
        CHECK(common::Sqlite3_reset(stmt));
        CHECK(common::Sqlite3_bind_int64(static_cast<int64_t>(idx),
                                 1,
                                 stmt));
        CHECK(common::Sqlite3_bind_int64(static_cast<int64_t>(obs.keyframe_id),
                                 2,
                                 stmt));
        CHECK(common::Sqlite3_bind_int64(static_cast<int64_t>(obs.camera_idx),
                                 3,
                                 stmt));
        CHECK(common::Sqlite3_bind_int64(static_cast<int64_t>(obs.track_id),
                                 4,
                                 stmt));
        CHECK(common::Sqlite3_bind_blob(blob_keypoint,
                                 5,
                                 sizeof(double) * obs.key_point.rows(),
                                 stmt));
        CHECK(common::Sqlite3_bind_blob(blob_descriptors,
                                 6,
                                 sizeof(float) * obs.projected_descriptors.rows(),
                                 stmt));
        CHECK(common::Sqlite3_step(stmt));

        free(blob_descriptors);
    }
    CHECK(common::Sqlite3_finalize(&stmt));
    CHECK(common::Sqlite3_exec("COMMIT;", database));
}

void MapSaver::SaveMap(const SummaryMap& summary_map) {
    // Clean and create table.
    CreateTable();

    // Save observers.
    SaveObservers(summary_map);

    // Save landmarks.
    SaveLandmarks(summary_map);

    // Save observations.
    SaveObservations(summary_map);

    LOG(INFO) << "Map saving success.";
}

}
