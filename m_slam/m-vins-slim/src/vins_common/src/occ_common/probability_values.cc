#include "occ_common/probability_values.h"

#include <limits>
#include <utility>

namespace common {

const std::vector<float>* const kValueToProbability =
        PrecomputeValueToBoundedFloat(
            kUnknownProbabilityValue,
            kMinProbability,
            kMinProbability,
            kMaxProbability).release();

const std::vector<float>* const kValueToCorrespondenceCost =
        PrecomputeValueToBoundedFloat(
            kUnknownCorrespondenceValue,
            kMaxCorrespondenceCost,
            kMinCorrespondenceCost,
            kMaxCorrespondenceCost).release();

std::unique_ptr<std::vector<float>> PrecomputeValueToBoundedFloat(
        const uint16 unknown_value, const float unknown_result,
        const float lower_bound, const float upper_bound) {
    auto result = std::make_unique<std::vector<float>>();
    // Repeat two times, so that both values with and without the update marker
    // can be converted to a probability.
    constexpr int kRepetitionCount = 2;
    result->reserve(kRepetitionCount * kValueCount);
    for (int repeat = 0; repeat != kRepetitionCount; ++repeat) {
        for (int value = 0; value != kValueCount; ++value) {
            result->push_back(SlowValueToBoundedFloat(
                                  value, unknown_value,
                                  unknown_result, lower_bound, upper_bound));
        }
    }
    CHECK_EQ(std::numeric_limits<uint16>::max() + 1u, result->size());
    return result;
}

std::vector<uint16> ComputeLookupTableToApplyOdds(const float odds) {
    std::vector<uint16> result;
    result.reserve(kValueCount);
    result.push_back(ProbabilityToValue(ProbabilityFromOdds(odds)) +
                     kUpdateMarker);
    for (int cell = 1; cell != kValueCount; ++cell) {
        result.push_back(ProbabilityToValue(
                             ProbabilityFromOdds(
                                 odds * Odds((*kValueToProbability)[cell]))) +
                         kUpdateMarker);
    }
    return result;
}

std::vector<uint16> ComputeLookupTableToApplyCorrespondenceCostOdds(
        float odds) {
    std::vector<uint16> result;
    result.reserve(kValueCount);
    result.push_back(CorrespondenceCostToValue(
                         ProbabilityToCorrespondenceCost(
                             ProbabilityFromOdds(odds))) +
                     kUpdateMarker);

    for (int cell = 1; cell != kValueCount; ++cell) {
        result.push_back(
                    CorrespondenceCostToValue(
                        ProbabilityToCorrespondenceCost(
                            ProbabilityFromOdds(
                                odds * Odds(
                                    CorrespondenceCostToProbability(
                                        (*kValueToCorrespondenceCost)[cell])))))
                    + kUpdateMarker);
    }
    return result;
}

}
