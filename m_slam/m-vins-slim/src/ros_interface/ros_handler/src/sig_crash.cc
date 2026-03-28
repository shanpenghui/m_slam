#include "ros_handler/sig_crash.h"

const int MAX_STACK_FRAMES = 128;
std::string crash_file_path = "./crash.txt";

void SetCrashFilePath(const std::string& file_path) {
    const std::string crash_file_name = "crash.txt";
    crash_file_path = common::ConcatenateFilePathFrom(file_path, crash_file_name);
}

void InterruptCallBack(int sig) {
    if (sig == SIGINT) {
        received_end_signal = true;
    }
}

void SigCrash(int sig) {
    FILE* fd = NULL;
    struct stat buf;
    stat(crash_file_path.c_str(), &buf);
    if (buf.st_size > 1 * 1000 * 1000) {// 超过 1 兆则清空内容
        fd = fopen(crash_file_path.c_str(), "w");
    } else {
        fd = fopen(crash_file_path.c_str(), "at");
    }
        
    try {
        char szLine[1024];
        time_t t = time(NULL);
        tm* now = localtime(&t);
        sprintf(szLine,
                "------------------------------------------------------------------------------------------\n"
                "[%04d-%02d-%02d %02d:%02d:%02d][crash signal number:%d]\n",
                now->tm_year + 1900, now->tm_mon + 1, now->tm_mday, now->tm_hour,
                now->tm_min, now->tm_sec, sig);
        if (fd) {
            fwrite(szLine, 1, strlen(szLine), fd);
        }

        void* array[MAX_STACK_FRAMES];
        int size = 0;
        char** strings = NULL;
        signal(sig, SIG_DFL);
        size = backtrace(array, MAX_STACK_FRAMES);
        strings = (char**)backtrace_symbols(array, size);
        LOG(ERROR) << "ERROR: Crashed. \n";

        for (int i = 0; i < size; ++i) {
            char szLine[1024];
            sprintf(szLine, "%d %s\n", i, strings[i]);
            std::string error_information(szLine);
            LOG(WARNING) << error_information;
            if (fd) { 
               fwrite(szLine, 1, strlen(szLine), fd);
            }
            fwrite(szLine, 1, strlen(szLine), stderr);
        }
        if (sig == SIGSEGV) {
            LOG(ERROR) << "Segmentation fault";
        } else if (sig == SIGABRT) {
            LOG(ERROR) << "Aborted";
        }

        free(strings);
    } catch (...) {
        //
    }

    fflush(fd);
    fclose(fd);
    fd = NULL;
    signal(sig, SIG_DFL);
}
