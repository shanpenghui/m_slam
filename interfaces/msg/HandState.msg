# hand 状态
uint64 state 0
# 0   正常
# 1   无法找到设备
# 2   初始状态异常
# 3   电机异常
# 4   六维力传感器异常
# 5   触觉传感器异常

# 推杆电机
LinearActuatorState[] linear_actuator_states
# 六维力传感器故障码
ForceSensorState force_sensor_state
# 触觉传感器状态
TactileSensorState tactile_sensor_state
