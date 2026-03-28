import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from math import pi
import random

last_pose = None
cum_pose = None

def normalize_angle(angle):
    return ((angle + pi) % (2*pi)) - pi

def odom_callback(msg):
    global odom_pub
    global last_pose
    global cum_pose

    if last_pose is None :
        last_pose = msg.pose.pose
        cum_pose = msg.pose.pose

    else :
        # Get the current pose
        curr_pose = msg.pose.pose

        # Compute the delta pose since the last time
        dx = curr_pose.position.x - last_pose.position.x
        dy = curr_pose.position.y - last_pose.position.y
        (_, _, dyaw) = euler_from_quaternion([
            curr_pose.orientation.x,
            curr_pose.orientation.y,
            curr_pose.orientation.z,
            curr_pose.orientation.w])
        (_, _, last_yaw) = euler_from_quaternion([
            last_pose.orientation.x,
            last_pose.orientation.y,
            last_pose.orientation.z,
            last_pose.orientation.w])
        dyaw = normalize_angle(dyaw - last_yaw)

        if dx != 0 or dy != 0 or dyaw != 0 :
            # Add some Gaussian noise to the delta pose
            dx += random.gauss(0, dx_noise)
            dy += random.gauss(0, dy_noise)
            dyaw += random.gauss(0, dyaw_noise)

        # Update the cumulative pose
        cum_pose.position.x += dx
        cum_pose.position.y += dy
        (_, _, cum_yaw) = euler_from_quaternion([
            cum_pose.orientation.x,
            cum_pose.orientation.y,
            cum_pose.orientation.z,
            cum_pose.orientation.w])
        q = quaternion_from_euler(0, 0, cum_yaw + dyaw)
        cum_pose.orientation.x = q[0]
        cum_pose.orientation.y = q[1]
        cum_pose.orientation.z = q[2]
        cum_pose.orientation.w = q[3]

        last_pose = msg.pose.pose

    # Publish the odometry message
    odom_msg = Odometry()
    odom_msg.header = msg.header
    odom_msg.pose.pose = cum_pose
    odom_pub.publish(odom_msg)

if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('odom_simulator')

    # Get the noise parameters
    dx_noise = rospy.get_param('~dx_noise', 0.0005)
    dy_noise = rospy.get_param('~dy_noise', 0.0)
    dyaw_noise = rospy.get_param('~dyaw_noise', 0.0005)

    # Subscribe to the ground truth odometry topic
    rospy.Subscriber('odom', Odometry, odom_callback)

    # Advertise the simulated odometry topic
    odom_pub = rospy.Publisher('odom_noised', Odometry, queue_size=10)

    # Spin until shutdown
    rospy.spin()
