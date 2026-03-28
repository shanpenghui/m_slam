#ifndef MVINS_SENSOR_PROPAGATOR_H_
#define MVINS_SENSOR_PROPAGATOR_H_

#include <deque>
#include <memory>

#include "cfg_common/slam_config.h"
#include "data_common/sensor_structures.h"
#include "data_common/state_structures.h"
#include "time_common/time.h"

namespace vins_core {

class SensorPropagator {
public:
    SensorPropagator() = default;
    template <typename DataType>
    void RemovePropagateData(const uint64_t timestamp_ns,
                             std::deque<DataType>* meas_buffer_ptr);
    template <typename DataType>
    bool GetPropagateData(const uint64_t time_ns,
                          std::deque<DataType>* meas_buffer_ptr,
                          std::vector<DataType>* prop_data_ptr);

    template <typename DataType>
    bool SelectSensorReadings(const uint64_t timestamp_ns,
                              std::deque<DataType>* meas_buffer_ptr,
                              std::vector<DataType>* prop_data_ptr);

protected:
    template <typename DataType>
    bool VerifyDataCoverage(const uint64_t time_ns,
                            const std::deque<DataType>& meas_buffer);
    template <typename DataType>
    DataType InterpolateData(const DataType& meas_1,
                             const DataType& meas_2,
                             uint64_t timestamp_ns);
};

}

#endif
