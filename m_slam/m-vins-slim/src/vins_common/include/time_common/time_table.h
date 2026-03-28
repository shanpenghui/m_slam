#ifndef TIME_TABLE_H_
#define TIME_TABLE_H_

#include <chrono>
#include <ctime>
#include <map>
#include <memory>

// You need to add events in this header file.
#include "time_common/time_table_events.h"

#define TIME_TIC(name) common::TimeTable::Tic(common::Event::name)
#define TIME_TIC_IF(name, condition) if (condition) TIME_TIC(name)

#define TIME_TOC(name) common::TimeTable::Toc(common::Event::name)
#define TIME_TOC_IF(name, condition) if (condition) TIME_TOC(name)

#define TIME_COUNT(name, count) \
    common::TimeTable::Count(common::Event::name, count); \

#define TIME_COUNT_IF(name, count, condition) if (condition) \
    TIME_COUNT(name, count) \

#define TIME_SMART_TIC(name) common::SmartTictoc name(common::Event::name)
#define TIME_SMART_TIC_IF(name, condition) \
    std::unique_ptr<common::SmartTictoc> name((condition) \
        ? new common::SmartTictoc(common::Event::name) \
        : nullptr) \

namespace common {

// List of events, automatically generated. Please add your events
// in "time_table_events.h"
#define ADD_EVENT(event) event,
enum Event {
    MVINS_EVENTS
};
#undef ADD_EVENT

// For storing log information of each type of event.
template<typename T>
struct TimeTableItem {
    // Num. calls.
    size_t count;

    // Total metric.
    double mean, variance;

    // Min, Max metric.
    T min, max;

    TimeTableItem() :
        count(0u),
        mean(0.0),
        variance(0.0),
        min(std::numeric_limits<T>::max()),
        max(static_cast<T>(0.0)) {}
};

// Singleton pattern for watching all time costs.
class TimeTable {
    typedef std::chrono::time_point<std::chrono::system_clock> Clock;
    typedef std::chrono::duration<double, std::milli> Duration;

 public:
    // Enable timetable, often called in GLogHelper::UseTimeTable() func.
    // If not reseted, program will ignore all other static functions.
    static void Reset();

    // Start counting time spent for an event.
    static void Tic(Event event);

    // End counting time spent for an event, also returns times in milliseconds.
    static double Toc(Event event);

    // Count for an event.
    static void Count(Event event, const size_t count);

    // Print time table and count table if exists.
    static void PrintAll();

 private:
    // Add one event.
    template <typename T>
    static void AddEventImpl(
        const T metric,
        TimeTableItem<T>* table);

    // Print results of a table.
    template <typename T>
    static void PrintAllImpl(
        const std::map<Event, TimeTableItem<T>>& table,
        const std::string& table_name);

    // In milliseconds, stores time info for all events.
    std::map<Event, TimeTableItem<double>> time_table_;

    // In milliseconds, stores the tic() of each live event.
    std::map<Event, Clock> live_time_table_;

    // In Num, stores count info for all events.
    std::map<Event, TimeTableItem<size_t>> count_table_;

    // Singleton.
    static std::unique_ptr<TimeTable> instance_;
};

// Smart tictoc for watching time costs. Use instance lifecycle for tic-toc.
class SmartTictoc {
 public:
    // Tic in the constructor.
    explicit SmartTictoc(Event event);

    // Toc in the destructor.
    ~SmartTictoc();

 private:
    const Event event_;
};

}  // namespace common

#endif  // TIME_TABLE_H_
