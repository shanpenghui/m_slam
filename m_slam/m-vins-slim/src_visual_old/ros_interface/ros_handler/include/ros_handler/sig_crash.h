#ifndef __SIG_CRASH_H__
#define __SIG_CRASH_H__

#include <csignal>
#include <errno.h>
#include <execinfo.h>
#include <functional>
#include "log_common/logging_tools.h"


bool received_end_signal = false;
void InterruptCallBack(int sig); 
void SigCrash(int sig);
void SetCrashFilePath(const std::string& crash_file_path);


#endif