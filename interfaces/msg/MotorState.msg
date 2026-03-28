# 电机命名，参考urdf
string name
# 电机运行模式
uint8 mode 0
# 电机位置 (弧度)
float32 q 0.0
# 电机速度 (弧度/秒)
float32 dq 0.0
# 电机加速度 (弧度/秒^2)
float32 ddq 0.0
# 估计扭矩 (Nm)
float32 tau_est 0.0
# 原始位置值 (未经滤波)
float32 q_raw 0.0
# 原始速度值 (未经滤波)
float32 dq_raw 0.0
# 原始加速度值 (未经滤波)
float32 ddq_raw 0.0
# 电机温度 (摄氏度)
uint8 temperature 0
# 通信丢失计数
uint32 lost 0
# 电机状态标志位
uint32 state 0
# 0x0000  正常
# 0x0001  过流
# 0x0002  过压
# 0x0004  欠压
# 0x0008  过载保护
# 0x0010  过温
# 0x0020  初始化异常
# 0x0040  软件错误
# 0x0080  电机超速
# 0x0100  电机堵转
# 0x0200  心跳掉线
# 0x0400  速度偏差过大
# 0x0800  位置偏差过大
# 0x1000  通讯异常  
# 0x2000  电机参数错误
# 0x4000  编码器错误
# 0x8000  DRV驱动错误    
# 通信频率 (Hz)
float32 communication_frequency 0.0
