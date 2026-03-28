#include "interface/interface_manager.h"

namespace mvins
{
bool InterfaceManager::SaveMaskOperationInMaskDatabase(const std::vector<float>& params) {
    if (params.size() % 2 != 0 || params.size() < 6) {
        LOG(ERROR) << "ERROR: Wrong params size, skip mask operation this time.";
        return false;
    }
    
    MaskOperation mask_operation = ParamToMaskOperation(params[0]);
    
    if (params.size() == 6) {
        const size_t line_index = static_cast<size_t>(params[1]);
        const Line mask_line(params[2], params[3], params[4], params[5]);
        switch (mask_operation) {
            case MaskOperation::ADD: {
                LOG(WARNING) << "Add mask line: " << mask_line.transpose();
                AddMaskLine(line_index, mask_line);
                break;
            }
            case MaskOperation::MODIFY: {
                LOG(WARNING) << "Modify mask line: " << mask_line.transpose();
                ModifyMaskLine(line_index, mask_line);
                break;
            }
            case MaskOperation::DELETE: {
                LOG(WARNING) << "Delete mask line: " << mask_line.transpose();
                DeleteMaskLine(line_index);
                break;
            }
            default: {
                LOG(ERROR) << "ERROR: Unknown mask line operation type.";
                return false;
            }
        }
    } else {
        const size_t polygon_index = static_cast<size_t>(params[1]);
        std::vector<float> polygon_points(params.begin() + 2, params.end());
        const int num_of_points = params.size() / 2 - 1;
        Polygon polygon(2, num_of_points);
        Eigen::Map<Polygon> vector_to_matrix_map(polygon_points.data(), 2, num_of_points);
        polygon = vector_to_matrix_map;
        switch (mask_operation) {
            case MaskOperation::ADD: {
                LOG(WARNING) << "Add mask polygon: \n" << polygon;
                AddMaskPolygon(polygon_index, num_of_points, polygon);
                break;
            }
            case MaskOperation::MODIFY: {
                LOG(WARNING) << "Modify mask polygon: \n" << polygon;
                ModifyMaskPolygon(polygon_index, num_of_points, polygon);
                break;
            }
            case MaskOperation::DELETE: {
                LOG(WARNING) << "Delete mask polygon: \n" << polygon;
                DeleteMaskPolygon(polygon_index);
                break;
            }
            default: {
                LOG(ERROR) << "ERROR: Unknown mask polygon operation type.";
                return false;
            }
        }
    }
    return true;
}

InterfaceManager::MaskOperation InterfaceManager::ParamToMaskOperation(const float& param) {
    if (param == 0) {
        return MaskOperation::ADD;
    } else if (param == 1) {
        return MaskOperation::MODIFY;
    } else if (param == 2) {
        return MaskOperation::DELETE;
    } else {
        return MaskOperation::ERROR;
    }
}

bool InterfaceManager::IsMaskDatabaseOpen() {
    return mask_database_ != nullptr;
}

bool InterfaceManager::IsMaskLineExist(size_t line_index) {
    if (!IsMaskDatabaseOpen()) {
        LOG(ERROR) << "ERROR: Mask database is not open.";
        return true;
    }
    
    sqlite3_stmt* stmt;
    std::string sql_select = "SELECT ID FROM MASK_LINE WHERE ID = "
        + std::to_string(line_index) + ";";
    CHECK(common::Sqlite3_prepare_v2(sql_select, &stmt, mask_database_));
    
    int result = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return result == SQLITE_ROW;
}

bool InterfaceManager::IsMaskPolygonExist(size_t polygon_index) {
    if (!IsMaskDatabaseOpen()) {
        LOG(ERROR) << "ERROR: Mask database is not open.";
        return true;
    }

    sqlite3_stmt* stmt;
    std::string sql_select = "SELECT ID FROM MASK_POLYGON WHERE ID = "
        + std::to_string(polygon_index) + ";";
    CHECK(common::Sqlite3_prepare_v2(sql_select, &stmt, mask_database_));
    
    int result = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return result == SQLITE_ROW;
}

void InterfaceManager::CreateMaskTable() {
    if (!IsMaskDatabaseOpen()) {
        const std::string complete_mask_file_name = 
            common::ConcatenateFilePathFrom(config_->map_path, common::kMaskFileName);
        const int open_failed = sqlite3_open(complete_mask_file_name.c_str(), &mask_database_);
        if (open_failed) {
            LOG(FATAL) << "ERROR: Cannot open mask database: "
                       << sqlite3_errmsg(mask_database_);
        }
    }
    // Drop exist line table
    const std::string drop_table_line = "DROP TABLE IF EXISTS MASK_LINE";
    CHECK(common::Sqlite3_exec(drop_table_line, mask_database_));
    
    // Create line table
    const std::string create_table_line = 
        "CREATE TABLE MASK_LINE(" \
        "ID   INT PRIMARY KEY NOT NULL UNIQUE," \
        "LINE BLOB            NOT NULL);";

    CHECK(common::Sqlite3_exec(create_table_line, mask_database_));

    // Drop exist polygon table
    const std::string drop_table_polygon = "DROP TABLE IF EXISTS MASK_POLYGON";
    CHECK(common::Sqlite3_exec(drop_table_polygon, mask_database_));
    
    // Create polygon table
    const std::string create_table_polygon = 
        "CREATE TABLE MASK_POLYGON(" \
        "ID         INT PRIMARY KEY NOT NULL UNIQUE," \
        "NUM_POINTS INT             NOT NULL," \
        "POLYGON       BLOB         NOT NULL);";

    CHECK(common::Sqlite3_exec(create_table_polygon, mask_database_));
}

void InterfaceManager::AddMaskLine(const size_t line_index, const Line& line) {
    if (IsMaskLineExist(line_index)) {
        LOG(ERROR) << "ERROR: Line " << line_index << " is already exist.";
        return;
    }

    mask_lines_[line_index] = line;
    
    sqlite3_stmt* stmt = nullptr;
    const std::string sql_insert = "INSERT INTO MASK_LINE VALUES(?,?)";  
    CHECK(common::Sqlite3_prepare_v2(sql_insert, &stmt, mask_database_));

    void* blob_line = nullptr;
    blob_line = reinterpret_cast<char*>(malloc(sizeof(Line)));
    memcpy(blob_line, reinterpret_cast<const char*>(line.data()), sizeof(Line));
    
    CHECK(common::Sqlite3_bind_int64(static_cast<int64_t>(line_index),
                                    1,
                                    stmt));
    CHECK(common::Sqlite3_bind_blob(blob_line,
                                    2,
                                    sizeof(Line),
                                    stmt));
    CHECK(common::Sqlite3_step(stmt));

    free(blob_line);
    CHECK(common::Sqlite3_finalize(&stmt));
}

void InterfaceManager::ModifyMaskLine(const size_t line_index, const Line& line) {
    if (!IsMaskLineExist(line_index)) {
        LOG(ERROR) << "ERROR: Line " << line_index << " is not exist.";
        return;
    }

    mask_lines_.at(line_index) = line;
    
    sqlite3_stmt* stmt = nullptr;
    const std::string sql_modify = 
        "UPDATE MASK_LINE SET LINE = ? WHERE ID = " 
        + std::to_string(line_index) + ";";
    CHECK(common::Sqlite3_prepare_v2(sql_modify, &stmt, mask_database_));

    void* blob_line = nullptr;
    blob_line = reinterpret_cast<char*>(malloc(sizeof(Line)));
    memcpy(blob_line, reinterpret_cast<const char*>(line.data()), sizeof(Line));
    
    CHECK(common::Sqlite3_bind_blob(blob_line,
                                    1,
                                    sizeof(Line),
                                    stmt));
    CHECK(common::Sqlite3_step(stmt));
    free(blob_line);
    CHECK(common::Sqlite3_finalize(&stmt));
}

void InterfaceManager::DeleteMaskLine(const size_t line_index) {
    if (!IsMaskLineExist(line_index)) {
        LOG(ERROR) << "ERROR: Line " << line_index << " is not exist.";
        return;
    }

    mask_lines_.erase(line_index);

    sqlite3_stmt* stmt = nullptr;
    const std::string sql_delete = "DELETE FROM MASK_LINE WHERE ID = "
        + std::to_string(line_index) + ";";

    CHECK(common::Sqlite3_prepare_v2(sql_delete, &stmt, mask_database_));
    CHECK(common::Sqlite3_step(stmt));
    CHECK(common::Sqlite3_finalize(&stmt));
}

void InterfaceManager::AddMaskPolygon(const size_t polygon_index, const int num_of_points, const Polygon& polygon) {
    if (IsMaskPolygonExist(polygon_index)) {
        LOG(ERROR) << "ERROR: Polygon " << polygon_index << " is already exist.";
        return;
    }

    mask_polygons_[polygon_index] = polygon;

    sqlite3_stmt* stmt = nullptr;
    const std::string sql_insert = "INSERT INTO MASK_POLYGON VALUES(?,?,?)";  
    CHECK(common::Sqlite3_prepare_v2(sql_insert, &stmt, mask_database_));

    void* blob_polygon = nullptr;
    blob_polygon = reinterpret_cast<char*>(malloc(sizeof(float) * 2 * num_of_points));
    memcpy(blob_polygon, reinterpret_cast<const char*>(polygon.data()), sizeof(float) * 2 * num_of_points);
    
    CHECK(common::Sqlite3_bind_int64(static_cast<int64_t>(polygon_index),
                                    1,
                                    stmt));
    CHECK(common::Sqlite3_bind_int64(static_cast<int64_t>(num_of_points),
                                    2,
                                    stmt));
    CHECK(common::Sqlite3_bind_blob(blob_polygon,
                                    3,
                                    sizeof(float) * 2 * num_of_points,
                                    stmt));
    CHECK(common::Sqlite3_step(stmt));

    free(blob_polygon);
    CHECK(common::Sqlite3_finalize(&stmt));  
}

void InterfaceManager::ModifyMaskPolygon(const size_t polygon_index, const int num_of_points, const Polygon& polygon) {
    if (!IsMaskPolygonExist(polygon_index)) {
        LOG(ERROR) << "ERROR: Polygon " << polygon_index << " is not exist.";
        return;
    }

    mask_polygons_.at(polygon_index) = polygon;

    sqlite3_stmt* stmt = nullptr;
    const std::string sql_modify = 
        "UPDATE MASK_POLYGON SET NUM_POINTS = ?, POLYGON = ? WHERE ID = " 
        + std::to_string(polygon_index) + ";";
    CHECK(common::Sqlite3_prepare_v2(sql_modify, &stmt, mask_database_));

    void* blob_polygon = nullptr;
    blob_polygon = reinterpret_cast<char*>(malloc(sizeof(float) * 2 * num_of_points));
    memcpy(blob_polygon, reinterpret_cast<const char*>(polygon.data()), sizeof(float) * 2 * num_of_points);
    
    CHECK(common::Sqlite3_bind_int64(static_cast<int64_t>(num_of_points),
                                    1,
                                    stmt));
    CHECK(common::Sqlite3_bind_blob(blob_polygon,
                                    2,
                                    sizeof(float) * 2 * num_of_points,
                                    stmt));
    CHECK(common::Sqlite3_step(stmt));
    free(blob_polygon);
    CHECK(common::Sqlite3_finalize(&stmt));
}

void InterfaceManager::DeleteMaskPolygon(const size_t polygon_index) {
    if (!IsMaskPolygonExist(polygon_index)) {
        LOG(ERROR) << "ERROR: Polygon " << polygon_index << " is not exist.";
        return;
    }

    mask_polygons_.erase(polygon_index);

    sqlite3_stmt* stmt = nullptr;
    const std::string sql_delete = "DELETE FROM MASK_POLYGON WHERE ID = "
        + std::to_string(polygon_index) + ";";

    CHECK(common::Sqlite3_prepare_v2(sql_delete, &stmt, mask_database_));
    CHECK(common::Sqlite3_step(stmt));
    CHECK(common::Sqlite3_finalize(&stmt));
}

void InterfaceManager::LoadMaskTable() {
    LoadMaskLine();
    LoadMaskPolygon();
}

void InterfaceManager::LoadMaskLine() {
    const std::string sql_query_line = "SELECT * FROM MASK_LINE;";
    sqlite3_stmt* stmt_line = nullptr;
    CHECK(common::Sqlite3_prepare_v2(
                            sql_query_line,
                            &stmt_line,
                            mask_database_));
    CHECK(common::Sqlite3_reset(stmt_line));
    while (sqlite3_step(stmt_line) == SQLITE_ROW) {
        const int id = sqlite3_column_int64(stmt_line, 0);
        const void* blob_line = sqlite3_column_blob(stmt_line, 1);
        const int line_size = sqlite3_column_bytes(stmt_line, 1);
        CHECK_EQ(line_size, 16);
        Line line(reinterpret_cast<const float*>(blob_line));
        mask_lines_[id] = line;
    }

    CHECK(common::Sqlite3_finalize(&stmt_line));
}

void InterfaceManager::LoadMaskPolygon() {
    const std::string sql_query_polygon = 
        "SELECT * FROM MASK_POLYGON;";

    sqlite3_stmt* stmt_polygon = nullptr;
    CHECK(common::Sqlite3_prepare_v2(
                            sql_query_polygon,
                            &stmt_polygon,
                            mask_database_));
    CHECK(common::Sqlite3_reset(stmt_polygon));
    while (sqlite3_step(stmt_polygon) == SQLITE_ROW) {
        const int id = sqlite3_column_int64(stmt_polygon, 0);
        const int num_of_points = sqlite3_column_int64(stmt_polygon, 1);
        const void* blob_polygon = sqlite3_column_blob(stmt_polygon, 2);
        const int polygon_size = sqlite3_column_bytes(stmt_polygon, 2);
        CHECK_EQ(polygon_size, sizeof(float) * 2 * num_of_points);
        Polygon polygon = Eigen::Map<const Polygon>(reinterpret_cast<const float*>(blob_polygon), 2, num_of_points);
        mask_polygons_[id] = polygon;   
    }
    CHECK(common::Sqlite3_finalize(&stmt_polygon));
}
} 
