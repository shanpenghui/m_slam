#include <glog/logging.h>

#include "sensor_propagator/sensor_propagator.h"

namespace vins_core {

template <typename DataType>
bool SensorPropagator::VerifyDataCoverage(
        const uint64_t timestamp_ns,
        const std::deque<DataType>& meas_buffer) {
    if (meas_buffer.empty()) {
        VLOG(2) << "No " << typeid(DataType).name()
                << " measurements for " << timestamp_ns;
        return false;
    }

    if (meas_buffer.front().timestamp_ns > timestamp_ns) {
        VLOG(2) << "First sensor data is newer than the queried timestamp: "
                << meas_buffer.front().timestamp_ns << ", " << timestamp_ns;
        return false;
    }

    if (meas_buffer.size() < 2u || meas_buffer.back().timestamp_ns < timestamp_ns) {
        VLOG(2) << "Queried time: " << std::setprecision(18)
                << common::NanoSecondsToSeconds(timestamp_ns) << ", latest sensor time: "
                << common::NanoSecondsToSeconds(meas_buffer.back().timestamp_ns);
        return false;
    }

    return true;
}

template <typename DataType>
void SensorPropagator::RemovePropagateData(
        const uint64_t timestamp_ns,
        std::deque<DataType>* meas_buffer_ptr) {
    std::deque<DataType>& meas_buffer = *CHECK_NOTNULL(meas_buffer_ptr);

    if (!VerifyDataCoverage(timestamp_ns, meas_buffer)) {
        return;
    }

    // Make sure the second one's timestamp is GE query timestamp_ns.
    while (meas_buffer.size() >= 2u) {
        if (meas_buffer[1].timestamp_ns < timestamp_ns) {
            meas_buffer.pop_front();
        } else {
            break;
        }
    }

    CHECK_GE(meas_buffer.size(), 2u);

    // The second one's timestamp is greater or equal to the queried timestamp_ns.
    if (meas_buffer[1].timestamp_ns == timestamp_ns) {
        // Just erase the first one.
        meas_buffer.pop_front();
    } else {
        DataType temp = InterpolateData(meas_buffer[0],
                                          meas_buffer[1],
                                          timestamp_ns);
        meas_buffer.pop_front();
        // Insert the interpolated data.
        meas_buffer.push_front(temp);
    }
}

template <typename DataType>
bool SensorPropagator::GetPropagateData(
        const uint64_t timestamp_ns,
        std::deque<DataType>* meas_buffer_ptr,
        std::vector<DataType>* prop_data_ptr) {
    std::deque<DataType>& meas_buffer = *CHECK_NOTNULL(meas_buffer_ptr);
    std::vector<DataType>& prop_data = *CHECK_NOTNULL(prop_data_ptr);

    if (!VerifyDataCoverage(timestamp_ns, meas_buffer)) {
        return false;
    }

    return SelectSensorReadings(timestamp_ns,
                                &meas_buffer,
                                &prop_data);
}

template <typename DataType>
bool SensorPropagator::SelectSensorReadings(
        const uint64_t timestamp_ns,
        std::deque<DataType>* meas_buffer_ptr,
        std::vector<DataType>* prop_data_ptr) {
    std::deque<DataType>& meas_buffer = *CHECK_NOTNULL(meas_buffer_ptr);
    std::vector<DataType>& prop_data = *CHECK_NOTNULL(prop_data_ptr);

    const size_t buffer_size = meas_buffer.size();
    size_t idx = 0u;
    for (; idx < buffer_size; ++idx) {
        if (meas_buffer[idx].timestamp_ns >= timestamp_ns) {
            break;
        }
    }

    if (idx == 0) {
        return false;
    }

    prop_data.resize(idx + 1);

    for (size_t i = 0u; i < idx; ++i) {
        prop_data[i] = meas_buffer.front();
        meas_buffer.pop_front();
    }

    CHECK_GE(meas_buffer.front().timestamp_ns, timestamp_ns);
    CHECK_GE(timestamp_ns, prop_data.back().timestamp_ns);

    DataType temp;
    temp = InterpolateData(prop_data[idx - 1],
                            meas_buffer.front(),
                            timestamp_ns);
    // If the first measurement's timestamp is close to timestamp_ns
    // it should be discarded.
    if ((meas_buffer.front().timestamp_ns - timestamp_ns) < 1e6) {
        meas_buffer.pop_front();
    }

    meas_buffer.push_front(temp);
    prop_data[idx] = temp;

    return true;
}

template <typename DataType>
DataType SensorPropagator::InterpolateData(
        const DataType& meas_1,
        const DataType& meas_2,
        uint64_t timestamp_ns) {
    // time-distance lambda
    double lambda = (common::NanoSecondsToSeconds(timestamp_ns) -
                       common::NanoSecondsToSeconds(meas_1.timestamp_ns)) /
                      (common::NanoSecondsToSeconds(meas_2.timestamp_ns) -
                       common::NanoSecondsToSeconds(meas_1.timestamp_ns));
    // CHECK_GE(lambda, 0.0);
    // CHECK_LE(lambda, 1.0);
    // // interpolate between the two times
    // DataType data = common::Interpolate(timestamp_ns, lambda, meas_1, meas_2);
    if (lambda  > 1.0) { 
        auto meas = meas_2;
        meas.timestamp_ns = timestamp_ns;
        return meas;
    } else if (lambda < 0.0) {
        auto meas = meas_1;
        meas.timestamp_ns = timestamp_ns;
        return meas;
    } else {
        return common::Interpolate(timestamp_ns, lambda, meas_1, meas_2);
    }
}

template
bool SensorPropagator::VerifyDataCoverage<common::ImuData>(
        const uint64_t timestamp_ns,
        const std::deque<common::ImuData>& meas_buffer);

template
bool SensorPropagator::VerifyDataCoverage<common::OdomData>(
        const uint64_t timestamp_ns,
        const std::deque<common::OdomData>& meas_buffer);

template
void SensorPropagator::RemovePropagateData<common::ImuData>(
        const uint64_t timestamp_ns,
        std::deque<common::ImuData>* meas_buffer_ptr);

template
void SensorPropagator::RemovePropagateData<common::OdomData>(
        const uint64_t timestamp_ns,
        std::deque<common::OdomData>* meas_buffer_ptr);

template
bool SensorPropagator::GetPropagateData<common::ImuData>(
        const uint64_t time_ns,
        std::deque<common::ImuData>* meas_buffer_ptr,
        std::vector<common::ImuData>* prop_data_ptr);

template
bool SensorPropagator::GetPropagateData<common::OdomData>(
        const uint64_t time_ns,
        std::deque<common::OdomData>* meas_buffer_ptr,
        std::vector<common::OdomData>* prop_data_ptr);

template
bool SensorPropagator::SelectSensorReadings<common::ImuData>(
        const uint64_t time_ns,
        std::deque<common::ImuData>* meas_buffer_ptr,
        std::vector<common::ImuData>* prop_data_ptr);

template
bool SensorPropagator::SelectSensorReadings<common::OdomData>(
        const uint64_t time_ns,
        std::deque<common::OdomData>* meas_buffer_ptr,
        std::vector<common::OdomData>* prop_data_ptr);

template
common::ImuData SensorPropagator::InterpolateData<common::ImuData>(
        const common::ImuData& meas_1,
        const common::ImuData& meas_2,
        uint64_t timestamp_ns);

template
common::OdomData SensorPropagator::InterpolateData<common::OdomData>(
        const common::OdomData& meas_1,
        const common::OdomData& meas_2,
        uint64_t timestamp_ns);
}
