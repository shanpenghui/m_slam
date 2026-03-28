#ifndef MVINS_LOOP_CLOSURE_DB_DEFINE_H_
#define MVINS_LOOP_CLOSURE_DB_DEFINE_H_

#include <glog/logging.h>
#include <string>

namespace summary_map {

enum class database_tables {
    OBSERVER,  // The observers information in the global frame.
    LANDMARK,  // The point landmarks information in the global frame.
    OBSERVATION  // Relationship information between observers and landmarks.
};

// There define the column's name in table observer.
enum class table_observer {
    ID,
    POSE
};

enum class table_landmark {
    ID,
    POSITION
};

enum class table_observation {
    ID,
    OBSERVER_ID,
    CAMERA_ID,
    LANDMARK_ID,
    KEY_POINT,
    DESCRIPTORS
};

inline std::string table_name(const database_tables& value) {
    switch (value) {
        case database_tables::OBSERVER : return "OBSERVER";
        case database_tables::LANDMARK : return "LANDMARK";
        case database_tables::OBSERVATION : return "OBSERVATION";
        default :
            LOG(FATAL) << "database have no this table name.";
            return "";
    }
}

inline std::string column_name_observer(const table_observer& value) {
    switch (value) {
        case table_observer::ID : return "ID";
        case table_observer::POSE : return "POSE";
        default :
            LOG(FATAL) << "table observer have no this column name.";
            return "";
    }
}

inline std::string column_name_landmark(const table_landmark& value) {
    switch (value) {
        case table_landmark::ID : return "ID";
        case table_landmark::POSITION : return "POSITION";
        default :
            LOG(FATAL) << "table landmark have no this column name.";
            return "";
    }
}

inline std::string column_name_observation(
    const table_observation& value) {
    switch (value) {
        case table_observation::ID : return "ID";
        case table_observation::OBSERVER_ID : return "OBSERVER_ID";
        case table_observation::CAMERA_ID : return "CAMERA_ID";
        case table_observation::LANDMARK_ID : return "LANDMARK_ID";
        case table_observation::KEY_POINT : return "KEY_POINT";
        case table_observation::DESCRIPTORS : return "DESCRIPTORS";
        default :
            LOG(FATAL) << "table observation have no this column name.";
            return "";
    }
}

}

#endif
