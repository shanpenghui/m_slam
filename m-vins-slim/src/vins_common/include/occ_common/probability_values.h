#ifndef OCCUPANCY_GRID_PROBABILITY_VALUES_H_
#define OCCUPANCY_GRID_PROBABILITY_VALUES_H_

#include <cmath>
#include <vector>

#include "glog/logging.h"

#include "occ_common/common.h"

namespace common {

constexpr int kValueCount = 32768;
constexpr uint16_t kUnknownProbabilityValue = 0;
constexpr uint16_t kUnknownCorrespondenceValue = kUnknownProbabilityValue;
constexpr uint16_t kUpdateMarker = 1u << 15;

extern const std::vector<float>* const kValueToProbability;
extern const std::vector<float>* const kValueToCorrespondenceCost;

inline uint16_t BoundedFloatToValue(const float float_value,
                                    const float lower_bound,
                                    const float upper_bound) {
    const int value =
            RoundToInt(
                (Clamp(float_value, lower_bound, upper_bound) - lower_bound) *
                (32766.f / (upper_bound - lower_bound))) +
            1;
    // DCHECK for performance.
    DCHECK_GE(value, 1);
    DCHECK_LE(value, 32767);
    return value;
}

// 0 is unknown, [1, 32767] maps to [lower_bound, upper_bound].
inline float SlowValueToBoundedFloat(
        const uint16 value,
        const uint16 unknown_value,
        const float unknown_result,
        const float lower_bound,
        const float upper_bound) {
    CHECK_LT(value, kValueCount);
    if (value == unknown_value) return unknown_result;
    const float kScale = (upper_bound - lower_bound) / (kValueCount - 2.f);
    return value * kScale + (lower_bound - kScale);
}

inline float Odds(float probability) {
    return probability / (1.f - probability);
}

inline float ProbabilityFromOdds(const float odds) {
    return odds / (odds + 1.f);
}

inline float ProbabilityToCorrespondenceCost(const float probability) {
    return 1.f - probability;
}

inline float CorrespondenceCostToProbability(const float correspondence_cost) {
    return 1.f - correspondence_cost;
}

// Converts a correspondence_cost to a uint16 in the [1, 32767] range.
inline uint16_t CorrespondenceCostToValue(const float correspondence_cost) {
    return BoundedFloatToValue(correspondence_cost, kMinCorrespondenceCost,
                               kMaxCorrespondenceCost);
}

// Converts a probability to a uint16 in the [1, 32767] range.
inline uint16_t ProbabilityToValue(const float probability) {
    return BoundedFloatToValue(probability, kMinProbability, kMaxProbability);
}

// Converts a uint16 (which may or may not have the update marker set) to a
// probability in the range [kMinProbability, kMaxProbability].
inline float ValueToProbability(const uint16_t value) {
    return (*kValueToProbability)[value];
}

// Converts a uint16 (which may or may not have the update marker set) to a
// correspondence cost in the range [kMinCorrespondenceCost,
// kMaxCorrespondenceCost].
inline float ValueToCorrespondenceCost(const uint16_t value) {
    return (*kValueToCorrespondenceCost)[value];
}

std::unique_ptr<std::vector<float>> PrecomputeValueToBoundedFloat(
        const uint16 unknown_value, const float unknown_result,
        const float lower_bound, const float upper_bound);
std::vector<uint16_t> ComputeLookupTableToApplyOdds(float odds);
std::vector<uint16_t> ComputeLookupTableToApplyCorrespondenceCostOdds(
        float odds);

}
#endif
