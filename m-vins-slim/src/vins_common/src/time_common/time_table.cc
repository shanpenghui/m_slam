#include "time_common/time_table.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <glog/logging.h>

namespace common {

std::unique_ptr<TimeTable> TimeTable::instance_ = nullptr;

#define ADD_EVENT(event) #event,
static const char *kEventPrinter[] = {
    MVINS_EVENTS
};
#undef ADD_EVENT

void TimeTable::Reset() {
    instance_.reset(new TimeTable);
}

void TimeTable::Tic(Event event) {
    if (!instance_)
        return;

    instance_->live_time_table_[event] = std::chrono::system_clock::now();
}

double TimeTable::Toc(Event event) {
    if (!instance_)
        return 0.0;

    CHECK(instance_->live_time_table_.find(event) !=
        instance_->live_time_table_.end())
        << "Tic of Event " << event << " not called!";
    const Duration duration = std::chrono::system_clock::now()
        - instance_->live_time_table_[event];
    const double duration_double = static_cast<double>(duration.count());
    AddEventImpl(duration_double, &instance_->time_table_[event]);

    return duration_double;
}

void TimeTable::Count(Event event, const size_t count) {
    if (!instance_)
        return;

    AddEventImpl(count, &instance_->count_table_[event]);
}

void TimeTable::PrintAll() {
    if (!instance_)
        return;

    instance_->PrintAllImpl(instance_->time_table_, "TimeTable (ms)");
    instance_->PrintAllImpl(instance_->count_table_, "CountTable");
}

template <typename T>
void TimeTable::AddEventImpl(
    const T metric,
    TimeTableItem<T>* tab_ptr) {
    ++tab_ptr->count;

    if (tab_ptr->count == 1u) {
        tab_ptr->mean = metric;
        tab_ptr->variance = 0.0;
    } else {
        // Recursive update.
        const double old_mean = tab_ptr->mean;
        tab_ptr->mean += (metric - old_mean) / tab_ptr->count;
        const double old_var = tab_ptr->variance;
        const double old_mean_square = old_mean * old_mean;
        tab_ptr->variance += (old_mean_square - tab_ptr->mean * tab_ptr->mean
                         + (metric * metric - old_var - old_mean_square)
                         / tab_ptr->count);

    }

    tab_ptr->min = std::min(metric, tab_ptr->min);
    tab_ptr->max = std::max(metric, tab_ptr->max);
}

template <typename T>
void TimeTable::PrintAllImpl(
    const std::map<Event, TimeTableItem<T>>& table,
    const std::string& table_name) {
    const std::streamsize precision = std::cout.precision();
    std::cout.precision(3);

    if (!table.empty()) {
        LOG(INFO) << "Printing " << table_name
                  << ": Ave | Stdev | Min | Max | Count";
        for (const auto& item : table) {
            if (item.second.count != 0u) {
                LOG(INFO) << kEventPrinter[item.first] << ": "
                    << item.second.mean << " | "
                    << std::sqrt(item.second.variance) << " | "
                    << item.second.min << " | "
                    << item.second.max << " | "
                    << item.second.count;
            }
        }
    }

    std::cout.precision(precision);
}

SmartTictoc::SmartTictoc(Event event) : event_(event) {
    common::TimeTable::Tic(event);
}

SmartTictoc::~SmartTictoc() {
    common::TimeTable::Toc(event_);
}

}  // namespace common
