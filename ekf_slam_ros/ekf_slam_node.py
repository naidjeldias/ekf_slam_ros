import rclpy
from rclpy.node import Node
from apriltag_msgs.msg import AprilTagDetectionArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
import numpy as np
import tf2_ros
from geometry_msgs.msg import Pose

from ekf_slam_ros.ekf_slam import EKFSLAM
from scipy.spatial.transform import Rotation as Rot
import tf2_geometry_msgs

class EKFSLAMNode(Node):
    def __init__(self):
        super().__init__('ekf_slam_node')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_frame = 'camera_link_optical'
        self.base_frame = 'base_link'
        self.camera_to_base = None
        
        Rt = np.diag([0.2, 0.2, np.deg2rad(20.0)]) ** 2
        Qt = np.diag([0.2, 0.2]) ** 2

        self.tag_size = 0.106
        self.v = 0.0
        self.w = 0.0
        self.Pinv = None
        # List of landmarks in the world frame
        self.landmarks = {
            1: np.array([2.77, 0.55]),
            2: np.array([2.77, -3]),
            3: np.array([2.77, -6.75]),
            4: np.array([1.8, -10.31])
        }
        
        self.last_update = None

        self.ekf = EKFSLAM(Rt, Qt, self.landmarks)
        
        self.tag_detect_sub = self.create_subscription(
            AprilTagDetectionArray,
            '/tag_detection/detections',
            self.tag_detection_callback,
            1)
        
        self.subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',  
            self.camera_info_callback,
            1)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.control_callback,
            1)
        
        # filter frequency
        frequency = 1.0
        self.timer = self.create_timer(1./frequency, self.filter_loop)
    
    def camera_info_callback(self, msg):
        P = np.array(msg.p).reshape((3, 4))
        P = np.vstack((P, np.array([0, 0, 0, 1])))
        self.Pinv = np.linalg.inv(P)

    def control_callback(self, msg):
        # Extract linear and angular velocities from Twist message
        self.v = msg.twist.twist.linear.x
        self.w = msg.twist.twist.angular.z
        
        # Implement motion model to predict state change
        # Update self.state_estimate and self.covariance
        
    def tag_detection_callback(self, msg):
        for tag in msg.detections:
            H = np.array(tag.homography).reshape((3, 3))
            H = np.vstack((H, np.array([0, 0, 1])))
            measure = self.get_range_bearing(self.Pinv, H)
            # lm = [id, x, y]
            # lm = np.array([tag.id, measure[0], measure[1]])
            # self.ekf.update(lm)
            
    def get_range_bearing(self, Pinv, H):
        #compute extrinsic camera parameter
        #https://dsp.stackexchange.com/a/2737/31703
        #H = K * T  =>  T = K^(-1) * H
        if Pinv is not None and self.camera_to_base is not None:
            T = np.matmul(Pinv, H)
            t = T[:3, -1]
            average_norm = (np.linalg.norm(T[:, 0]) + np.linalg.norm(T[:, 0])) / 2.0
            t = t  / average_norm * self.tag_size / 2.0

            R = np.zeros((3, 3))
            R[:, 0] = T[:3, 0] / np.linalg.norm(T[:3, 0])
            R[:, 1] = T[:3, 1] / np.linalg.norm(T[:3, 1])
            R[:, 2] = np.cross(R[:, 0], R[:, 1])

            R[:, 1] *= -1
            R[:, 2] *= -1

            r = Rot.from_matrix(R)

            tag_pose = Pose()
            tag_pose.position.x = t[0]
            tag_pose.position.y = t[1]
            tag_pose.position.z = t[2]
            tag_pose.orientation.x = r.as_quat()[0]
            tag_pose.orientation.y = r.as_quat()[1]
            tag_pose.orientation.z = r.as_quat()[2]
            tag_pose.orientation.w = r.as_quat()[3]

            # print("Tag position: ", tag_pose.position.x, tag_pose.position.y, tag_pose.position.z)
            # print("Tag orientation: ", tag_pose.orientation.x, tag_pose.orientation.y, tag_pose.orientation.z, tag_pose.orientation.w)

            pose_transformed = tf2_geometry_msgs.do_transform_pose(tag_pose, self.camera_to_base)
            print("Position: ", pose_transformed.position.x, pose_transformed.position.y, pose_transformed.position.z)
            print("Orientation: ", pose_transformed.orientation.x, pose_transformed.orientation.y, pose_transformed.orientation.z, pose_transformed.orientation.w)
            
            r = Rot.from_quat([pose_transformed.orientation.x, pose_transformed.orientation.y, pose_transformed.orientation.z, pose_transformed.orientation.w])
            print("Rotation: ", r.as_euler('xyz', degrees=True))
            # z_rotation = np.array([
            #     [-1, 0, 0],
            #     [0, -1, 0],
            #     [0, 0, 1]
            # ])
            # x_rotation = np.array([
            #     [1, 0, 0],
            #     [0, -1, 0],
            #     [0, 0, -1]
            # ])
            
            # # Apply rotations
            # R = np.dot(R, z_rotation)
            # R = np.dot(R, x_rotation)

            # print("R: ", R)

            return np.array([t[2], -t[0]])
        
    def filter_loop(self):
        try:
            
            self.camera_to_base = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rclpy.time.Time())

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            self.get_logger().info(
                        f'Could not transform {self.base_frame} to {self.camera_frame}: {ex}')
            pass

        current_time = self.get_clock().now()
        if self.last_update is not None:
            dt = (current_time - self.last_update).nanoseconds / 1e9
            self.ekf.predict(np.array([self.v, self.w]).reshape((2, 1)), dt)
        self.last_update = current_time

def main(args=None):
    rclpy.init(args=args)
    ekf_slam_node = EKFSLAMNode()
    rclpy.spin(ekf_slam_node)
    ekf_slam_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()