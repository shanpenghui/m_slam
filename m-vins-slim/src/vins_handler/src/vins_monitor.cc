#include "vins_handler/vins_monitor.h"

#include "glog/logging.h"

namespace vins_handler {
SlamStateMonitor::SlamStateMonitor(const int max_boundary) {
    max_boundary_ = max_boundary;
    min_boundary_ = -max_boundary;
    ResetCounter();
}

void SlamStateMonitor::UpdateCounter(const UpdateType type) {
    if (type == PLUS) {
        counter_ = (counter_ + 1 > max_boundary_)
                        ? max_boundary_ : counter_ + 1;
    } else if (type == MINUS) {
        counter_ = (counter_ - 1 < min_boundary_)
                        ? min_boundary_ : counter_ - 1;
    }
    VLOG(2) << "Current SLAM state level: " << counter_;
}

bool SlamStateMonitor::IsRunEnoughGood() const {
    return counter_ == max_boundary_;
}

bool SlamStateMonitor::IsRunNotEnoughGood() const {
    return counter_ <= 0 && counter_ > min_boundary_;
}

bool SlamStateMonitor::IsRunEnoughBad() const {
    return counter_ == min_boundary_;
}

void SlamStateMonitor::ResetCounter() {
    counter_ = max_boundary_;
}
}  // namespace vins_handler
