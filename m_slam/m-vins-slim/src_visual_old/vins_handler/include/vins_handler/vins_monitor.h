

#ifndef M_VINS_VINS_MONITOR_H_
#define M_VINS_VINS_MONITOR_H_

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>
#include <Eigen/Core>

namespace vins_handler {
enum UpdateType {
    PLUS,
    MINUS
};

class SlamStateMonitor {
public:
    explicit SlamStateMonitor(const int max_boundary);
    ~SlamStateMonitor() = default;
    void UpdateCounter(const UpdateType type);
    bool IsRunEnoughGood() const;
    bool IsRunNotEnoughGood() const;
    bool IsRunEnoughBad() const;
    void ResetCounter();
private:
    int max_boundary_;
    int min_boundary_;
    int counter_;
};
}  // namespace slam_handler
#endif  // SLAM_HANDLER_SLAM_MONITOR_H_
