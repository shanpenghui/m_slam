# For a given velocity command, the poses that the robot will go to in the allotted time.

# Input Velocity
geometry_msgs/Twist velocity

# Poses the robot will go to, given our kinematic model
geometry_msgs/Pose2D[] poses

# Time difference between the current pose and the poses in the trajectory.
# Parallel array to poses. Length should be the same.
#duration[] time_offsets
