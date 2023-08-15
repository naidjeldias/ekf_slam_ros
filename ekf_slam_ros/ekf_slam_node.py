import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from apriltag_msgs.msg import AprilTagDetectionArray
from nav_msgs.msg import Odometry
import numpy as np
import math

from ekf_slam_ros.ekf_slam import EKFSLAM


class EKFSLAMNode(Node):
    def __init__(self):
        super().__init__('ekf_slam_node')
        
        # self.tag_detect_sub = self.create_subscription(
        #     AprilTagDetectionArray,
        #     '/tag_detection/detections',
        #     self.tag_detection_callback,
        #     10)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        Rt = np.diag([0.2, 0.2, np.deg2rad(20.0)]) ** 2
        Qt = np.diag([0.2, 0.2]) ** 2
        
        self.odom_last_msg_time = None

        self.ekf = EKFSLAM(Rt, Qt)
    
    def odom_callback(self, msg):
        # Extract linear and angular velocities from Twist message
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        current_time = self.get_clock().now()
        if self.odom_last_msg_time is not None:
            dt = (current_time - self.odom_last_msg_time).nanoseconds / 1e9
            self.get_logger().info('dt: %f' % dt)
            self.ekf.predict(np.array([v, w]).reshape((2, 1)), dt)

        self.odom_last_msg_time = current_time
        self.get_logger().info('Linear velocity and angular velocity: %f, %f' % (v, w))
        
        # Implement motion model to predict state change
        # Update self.state_estimate and self.covariance
        
    def tag_detection_callback(self, msg):
        self.get_logger().info('Tag detection callback')
        
        # Implement EKF update step using AprilTag detections
        # Update self.state_estimate and self.covariance
        # Set map_msg with the updated state_estimate and covariance
    
    def initialize_ekf(self, msg):
        # Initialize EKF with the first detected AprilTag pose
        tag_pose = msg.detections[0].pose.pose.pose
        self.state_estimate = np.array([
            tag_pose.position.x,
            tag_pose.position.y,
            math.atan2(2 * tag_pose.orientation.w * tag_pose.orientation.z,
                       1 - 2 * tag_pose.orientation.z ** 2)  # Extract yaw from quaternion
        ]).reshape((3, 1))
        
        self.ekf_initialized = True

def main(args=None):
    rclpy.init(args=args)
    ekf_slam_node = EKFSLAMNode()
    rclpy.spin(ekf_slam_node)
    ekf_slam_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()