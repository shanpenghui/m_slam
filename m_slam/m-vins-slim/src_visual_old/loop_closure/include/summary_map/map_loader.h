#ifndef MVINS_LOOP_CLOSURE_MAP_LOADER_H_
#define MVINS_LOOP_CLOSURE_MAP_LOADER_H_

#include "summary_map/db_define.h"
#include "sqlite_common/sqlite_util.h"

#include "summary_map/summary_map.h"

namespace loop_closure {
class MapLoader {
public:
    MapLoader(const std::string& load_path);
    ~MapLoader();
    void LoadMap(SummaryMap* summary_map_ptr);
private:
    void ReadObservers(std::vector<int>* vertex_ids_ptr,
                       Eigen::Matrix<double, 7, Eigen::Dynamic>* T_OtoMs_ptr);
    void ReadLandmarks(std::vector<int>* track_ids_ptr,
                         Eigen::Matrix3Xd* p_LinMs_ptr);
    void ReadObservations(std::vector<int>* obs_vertex_ids_ptr,
                           std::vector<int>* obs_cam_ids_ptr,
                           std::vector<int>* obs_track_ids_ptr,
                           Eigen::Matrix2Xd* key_points_ptr,
                           common::DescriptorsMatF32* descriptors_ptr);
    sqlite3* database_;
};
}

#endif
