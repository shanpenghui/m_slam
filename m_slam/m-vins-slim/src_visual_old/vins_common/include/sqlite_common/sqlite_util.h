// This file implements a series of operations on Sqlite3
// and defines a memory copy template function
// when the content to be copied is a std::vector.
#ifndef MVINS_LOOP_CLOSURE_SQLITE_UTIL_H_
#define MVINS_LOOP_CLOSURE_SQLITE_UTIL_H_

#include <Eigen/Core>
#include <glog/logging.h>
#include <sqlite3.h>
#include <string>

namespace common {

// Default sqlite3 query callback.
inline int Callback(void* NotUsed,
             int argc,
             char** argv,
             char** azColName) {
    for (int i = 0; i < argc; i++) {
        printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
    }
    printf("\n");
    return 0;
}

// Convert error code to error description.
inline std::string ConvertErrorCodeToDescription(const int error_code) {
    switch (error_code) {
        case 0 : return "Successful result";
        case 1 : return "Generic error";
        case 2 : return "Access permission denied";
        case 3 : return "Callback routine requested an abort";
        case 4 : return "The database file is locked";
        case 5 : return "A table in the database is locked";
        case 6 : return "A malloc() failed";
        case 7 : return "Attempt to write a readonly database";
        case 8 : return "Operation terminated by sqlite3_interrupt()";
        case 9 : return "Internal logic error in SQLite";
        case 10 : return "Some kind of disk I/O error occurred";
        case 11 : return "The database disk image is malformed";
        case 12 : return "Unknown opcode in sqlite3_file_control()";
        case 13 : return "Insertion failed because database is full";
        case 14 : return "Unable to open the database file";
        case 15 : return "Database lock protocol error";
        case 16 : return "Internal use only";
        case 17 : return "The database schema changed";
        case 18 : return "String or BLOB exceeds size limit";
        case 19 : return "Abort due to constraint violation";
        case 20 : return "Data type mismatch";
        case 21 : return "Library used incorrectly";
        case 22 : return "Uses OS features not supported on host";
        case 23 : return "Authorization denied";
        case 24 : return "Not used";
        case 25 : return "2nd parameter to sqlite3_bind out of range";
        case 26 : return "File opened that is not a database file";
        case 27 : return "Notifications from sqlite3_log()";
        case 28 : return "Warnings from sqlite3_log()";
        case 100 : return "sqlite3_step() has another row ready";
        case 101 : return "sqlite3_step() has finished executing";
        default:
            return "Unknown";
    }
}

// Memory copy, contents into buffer.
template <typename Scalar>
inline void VectorMemcpy(const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& contents,
                         void** buffer_ptr) {
    void*& buffer = *CHECK_NOTNULL(buffer_ptr);
    if (contents.cols() == 0) {
        return;
    }
    buffer = reinterpret_cast<char*>(malloc(sizeof(Scalar) * contents.cols()));
    memcpy(buffer, reinterpret_cast<const char*>(&contents(0)),
             sizeof(Scalar) * contents.cols());
}

// Execute SQL directly.
inline bool Sqlite3_exec(const std::string& sql,
                         sqlite3* database) {
    char* zErrMsg = nullptr;
    const int rc = sqlite3_exec(database,
                                sql.c_str(),
                                nullptr,
                                nullptr,
                                &zErrMsg);
    if (rc != SQLITE_OK) {
        LOG(ERROR) << "SQL: " << sql
                   << "error: " << zErrMsg;
        sqlite3_free(zErrMsg);
        return false;
    } else {
        LOG(INFO) << "SQL: " << sql << " execute successfully.";
        return true;
    }
}

// Execute SQL prepare.
inline bool Sqlite3_prepare_v2(const std::string& sql,
                               sqlite3_stmt** stmt_ptr,
                               sqlite3* database) {
    sqlite3_stmt*& stmt = *CHECK_NOTNULL(stmt_ptr);
    const int rc = sqlite3_prepare_v2(database,
                                      sql.c_str(),
                                      -1,
                                      &stmt,
                                      nullptr);
    if (rc != SQLITE_OK) {
        LOG(ERROR) << "prepare SQL:" << sql << " failure! error descriptor: "
                   << ConvertErrorCodeToDescription(rc);
        return false;
    } else {
        return true;
    }

}

// Execute SQL reset.
inline bool Sqlite3_reset(sqlite3_stmt* stmt) {
    const int rc = sqlite3_reset(stmt);
    if (rc != SQLITE_OK) {
        LOG(ERROR) << "reset stmt operation failure! error descriptor: "
                   << ConvertErrorCodeToDescription(rc);
        return false;
    } else {
        return true;
    }
}

// Execute SQL step
inline bool Sqlite3_step(sqlite3_stmt* stmt) {
    const int rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        LOG(ERROR) << "step stmt operation failure! error descriptor: "
                   << ConvertErrorCodeToDescription(rc);
        return false;
    } else {
        return true;
    }
}

// Execute SQL finalize.
inline bool Sqlite3_finalize(sqlite3_stmt** stmt_ptr) {
    sqlite3_stmt*& stmt = *CHECK_NOTNULL(stmt_ptr);
    const int rc = sqlite3_finalize(stmt);
    if (rc != SQLITE_OK) {
        LOG(ERROR) << "finalize stmt operation failure! error descriptor: "
                   << ConvertErrorCodeToDescription(rc);
        return false;
    } else {
        return true;
    }
}

// Bind int64 data into prepared statements.
inline bool Sqlite3_bind_int64(const int64_t value,
                               const int number,
                               sqlite3_stmt* stmt) {
    const int rc = sqlite3_bind_int64(
        stmt,
        number,
        value);
    if (rc != SQLITE_OK) {
        LOG(ERROR) << "bind int64 operation failure! error descriptor: "
                   << ConvertErrorCodeToDescription(rc);
        return false;
    } else {
        return true;
    }
}

// Bind int32 data into prepared statements.
inline bool Sqlite3_bind_int32(const int value,
                               const int number,
                               sqlite3_stmt* stmt) {
    const int rc = sqlite3_bind_int(
        stmt,
        number,
        value);
    if (rc != SQLITE_OK) {
        LOG(ERROR) << "bind int32 operation failure! error descriptor: "
                   << ConvertErrorCodeToDescription(rc);
        return false;
    } else {
        return true;
    }
}

// Bind double data into prepared statements.
inline bool Sqlite3_bind_double(const double value,
                                const int number,
                                sqlite3_stmt* stmt) {
    const int rc = sqlite3_bind_double(
        stmt,
        number,
        value);
    if (rc != SQLITE_OK) {
        LOG(ERROR) << "bind double operation failure! error descriptor: "
                   << ConvertErrorCodeToDescription(rc);
        return false;
    } else {
        return true;
    }
}

// Bind blob data into prepared statements.
inline bool Sqlite3_bind_blob(const void* value,
                              const int number,
                              const int size,
                              sqlite3_stmt* stmt) {
    const int rc = sqlite3_bind_blob(
        stmt,
        number,
        value,
        size,
        nullptr);
    if (rc != SQLITE_OK) {
        LOG(ERROR) << "bind blob operation failure! error descriptor: "
                   << ConvertErrorCodeToDescription(rc);
        return false;
    } else {
        return true;
    }
}

}
#endif
