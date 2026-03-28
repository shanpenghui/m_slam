/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef COMMON_TIME_H_
#define COMMON_TIME_H_

#include <chrono>
#include <ostream>
#include <ratio>
#include <time.h>

#include "data_common/constants.h"

namespace common {

constexpr double kMillisecondsToSeconds = 1.e-3;
constexpr size_t kSecondsToMilliSeconds = 1e3;
constexpr double kNanosecondsToSeconds = 1.e-9;
constexpr size_t kSecondsToNanoSeconds = 1e9;

inline double NanoSecondsToSeconds(uint64_t time) {
    return static_cast<double>(time) * kNanosecondsToSeconds;
}

inline int64_t SecondsToNanoSeconds(double time) {
    return static_cast<int64_t>(time * kSecondsToNanoSeconds);
}

inline double MilliSecondsToSeconds(double time) {
    return time * kMillisecondsToSeconds;
}

inline double SecondsToMilliSeconds(double time) {
    return time * kSecondsToMilliSeconds;
}

class TicToc {
 public:
    TicToc() {
        tic();
    }

    void tic() {
        start = std::chrono::system_clock::now();
    }

    // return duration in milliseconds.
    double toc() {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return SecondsToMilliSeconds(elapsed_seconds.count());
    }

 private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

constexpr int64 kUtsEpochOffsetFromUnixEpochInSeconds =
    (719162ll * 24ll * 60ll * 60ll);

struct UniversalTimeScaleClock {
  using rep = int64;
  using period = std::ratio<1, 10000000>;
  using duration = std::chrono::duration<rep, period>;
  using time_point = std::chrono::time_point<UniversalTimeScaleClock>;
  static constexpr bool is_steady = true;
};

// Represents Universal Time Scale durations and timestamps which are 64-bit
// integers representing the 100 nanosecond ticks since the Epoch which is
// January 1, 1 at the start of day in UTC.
using Duration = UniversalTimeScaleClock::duration;
using Time = UniversalTimeScaleClock::time_point;

// Convenience functions to create common::Durations.
Duration FromSeconds(double seconds);
Duration FromMilliseconds(int64 milliseconds);

// Returns the given duration in seconds.
double ToSeconds(Duration duration);

// Creates a time from a Universal Time Scale.
Time FromUniversal(int64 ticks);

// Outputs the Universal Time Scale timestamp for a given Time.
int64 ToUniversal(Time time);

// For logging and unit tests, outputs the timestamp integer.
std::ostream& operator<<(std::ostream& os, Time time);

Time FromRos(const timespec& time);

long long getCurrentTimestamp();
}  // namespace common

#endif  // COMMON_TIME_H_
