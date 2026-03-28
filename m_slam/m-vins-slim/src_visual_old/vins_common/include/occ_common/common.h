#ifndef OCCUPANCY_GRID_COMMON_H_
#define OCCUPANCY_GRID_COMMON_H_

#include <cinttypes>
#include <cmath>
#include <memory>
#include <string>

#include <glog/logging.h>

#include "data_common/constants.h"
#include "math_common/math.h"

namespace common {

constexpr float kMinProbability = 0.1f;
constexpr float kMaxProbability = 1.f - kMinProbability;
constexpr float kMinCorrespondenceCost = 1.f - kMaxProbability;
constexpr float kMaxCorrespondenceCost = 1.f - kMinProbability;

// Converts the given probability to log odds.
inline float Logit(float probability) {
    return std::log(probability / (1.f - probability));
}

const float kMaxLogOdds = Logit(kMaxProbability);
const float kMinLogOdds = Logit(kMinProbability);

// Converts a probability to a log odds integer. 0 means unknown, [kMinLogOdds,
// kMaxLogOdds] is mapped to [1, 255].
inline uint8 ProbabilityToLogOddsInteger(const float probability) {
    const int value = RoundToInt((Logit(probability) - kMinLogOdds) *
                254.f / (kMaxLogOdds - kMinLogOdds)) + 1;
    CHECK_LE(1, value);
    CHECK_GE(255, value);
    return value;
}

}
#endif
