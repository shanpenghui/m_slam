#ifndef MVINS_LOOP_CLOSURE_MAP_SAVER_H_
#define MVINS_LOOP_CLOSURE_MAP_SAVER_H_

#include "summary_map/db_define.h"
#include "sqlite_common/sqlite_util.h"
#include "summary_map/summary_map.h"

namespace loop_closure {
class MapSaver {
public:
    MapSaver(const std::string& save_path);
    ~MapSaver();
    void SaveMap(const SummaryMap& summary_map);
private:
    void CreateTable();
    void SaveObservers(const SummaryMap& summary_map);
    void SaveLandmarks(const SummaryMap& summary_map);
    void SaveObservations(const SummaryMap& summary_map);

    const std::string save_path_;
    sqlite3* database;

};
}

#endif
