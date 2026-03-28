uint16                     action_index     # 0只走路点，1走目标笛卡尔pose in torso up link，2走delta_pose in torso up link, 3走delta_q,4走delta_pose in EE link,default 0 
uint16                     action_param     # 机械臂运行模式(0:MOVE_J,1:MOVE_L)
uint16                     arm_index        # 臂号
geometry_msgs/Pose         goal
geometry_msgs/Pose[]       waypoints_pose   # 笛卡尔空间的路点  
std_msgs/Float32MultiArray waypoints        # 关节空间的路点
std_msgs/Float32MultiArray force_thres      # 力位混合模式(大于设定的力就完成任务)
std_msgs/Float32MultiArray ik_initial_joint # ik initial joint angle
string                     tcp_frame_id
float32                    speed_percentage # 0-100，default 10
