#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math


GOAL_POSES = [[1, 0, -0.8], [6, 0, 0.8], [8.5, -2, -1.57], [-4, 2, 3.14], [-4.5, -3, -1.57]]


class NavigationMetrics:
    def __init__(self, automatic: bool = False):
        # whether to use hard-coded list of goal poses to automatically assign to the robot
        self.automatic = automatic
        if self.automatic:
            assert len(GOAL_POSES) != 0, "No goal poses are given!"
            self.goal_pose_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
            self.cur_goal_idx = 0
            self.finished = False

        # collision related variables
        self.collision_counts = 0
        self.last_collision_time = None
        self.COLLISION_IGNORE_DURATION = 3.0  # continuous collisions within this seconds will be ignored

        # navigation time related variables
        self.navigation_start_time = None
        self.navigation_duration = 0
        self.NAVIGATION_TIMEOUT_SEC = 120

        # report as a dictionary of all navigation metrics
        self.report = {'start_poses': [], 'goal_poses': [], 'time_costs': [], 'collision_counts': []}

        # location variables updated in callbacks
        self.robot_name = 'turtlebot3_waffle'
        self.robot_pose = None
        self.goal_pose = None

        # contants to judge robot's proximity to goal
        self.DIST_THR = 0.2  # meter
        self.ANGLE_THR = 0.6  # radian

        # declare all the subscribers and keep the process running
        rospy.init_node('navigation_metrics_node', anonymous=True)
        self.collision_sub = rospy.Subscriber('/robot/bumper_states', ContactsState,
                                              self.collision_callback, queue_size=1)
        self.goal_pose_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        self.robot_pose_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.robot_callback, queue_size=1)
        rospy.loginfo("Navigation metrics are running ...")
        rospy.spin()

    def __call__(self, *args, **kwargs):
        rospy.loginfo(f"Navigation reports for {len(self.report['start_poses'])} run(s):")
        rospy.loginfo(f"time costs: {self.report['time_costs']}")
        rospy.loginfo(f"collision counts: {self.report['collision_counts']}")
        rospy.loginfo(f"Total navigation time in sec: {sum(self.report['time_costs'])}, total collision counts: {sum(self.report['collision_counts'])}")

    def reset(self, reset_goal: bool = False):
        self.collision_counts = 0
        self.last_collision_time = None
        self.navigation_duration = 0
        if reset_goal:
            self.goal_pose = None

    def collision_callback(self, msg):
        contact_states = msg.states
        if len(contact_states) > 0:
            current_collision_time = msg.header.stamp
            if self.last_collision_time is None:
                self.collision_counts += 1
                self.last_collision_time = current_collision_time
            elif (current_collision_time - self.last_collision_time).to_sec() > self.COLLISION_IGNORE_DURATION:
                self.collision_counts += 1
                self.last_collision_time = current_collision_time
            rospy.loginfo_throttle(1, f"Your current collision counts: {self.collision_counts}.")

    def goal_callback(self, msg):
        if self.automatic:
            rospy.logwarn(f"No manual goal pose is accepted in automatic grading mode!")
            return

        # current navigation is preempted, metrics should be differentiable
        if self.goal_pose is not None:
            rospy.logwarn("Current navigation is not finished, but new goal comes, preempt!")
            self.report['time_costs'].append(self.navigation_duration + self.NAVIGATION_TIMEOUT_SEC)
            self.report['collision_counts'].append(self.collision_counts)

        rospy.loginfo("New goal received!")
        self.goal_pose = msg.pose
        if self.robot_pose is None:
            rospy.logwarn("Still waiting for robot pose when received goal pose, abort this goal!")
            return

        self.navigation_start_time = msg.header.stamp
        self.report['start_poses'].append(self.robot_pose)
        self.report['goal_poses'].append(self.goal_pose)
        self.reset()

    @staticmethod
    def dist(x1: float, y1: float, x2: float, y2: float):
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    @staticmethod
    def pose2d_to_rospose(x, y, yaw):
        rospose = Pose()
        rospose.position.x = x
        rospose.position.y = y
        q = quaternion_from_euler(0, 0, yaw)
        rospose.orientation = Quaternion(*q)
        return rospose

    def reached_goal(self):
        if self.goal_pose is None:
            return False

        while self.robot_pose is None:
            rospy.logwarn_throttle(0.5, "Waiting for robot pose to show up ...")

        # judge Euclidean distance
        robot_x, robot_y = self.robot_pose.position.x, self.robot_pose.position.y
        goal_x, goal_y = self.goal_pose.position.x, self.goal_pose.position.y
        if self.dist(robot_x, robot_y, goal_x, goal_y) > self.DIST_THR:
            return False

        # judge heading difference
        robot_quaternion = (self.robot_pose.orientation.x, self.robot_pose.orientation.y,
                            self.robot_pose.orientation.z, self.robot_pose.orientation.w)
        goal_quaternion = (self.goal_pose.orientation.x, self.goal_pose.orientation.y,
                           self.goal_pose.orientation.z, self.goal_pose.orientation.w)
        robot_theta = euler_from_quaternion(robot_quaternion)[2]  # (roll, pitch, yaw)
        goal_theta = euler_from_quaternion(goal_quaternion)[2]
        if abs(robot_theta - goal_theta) > self.ANGLE_THR:
            return False

        return True

    def robot_callback(self, msg):
        # won't do anything if all goal poses are finished
        if self.finished:
            return
        cur_goal_failed = False

        # update robot pose
        if self.robot_name not in msg.name:
            rospy.logwarn_throttle(0.5, f"Waiting for robot model to be spawned ...")
            return

        robot_id = msg.name.index(self.robot_name)
        self.robot_pose = msg.pose[robot_id]

        # update navigation time
        if self.goal_pose is not None:
            assert self.navigation_start_time is not None
            self.navigation_duration = (rospy.Time.now() - self.navigation_start_time).to_sec()
            # timeout then select the next goal pose if there is any
            if self.navigation_duration > self.NAVIGATION_TIMEOUT_SEC:
                rospy.logwarn("Navigation timeout, preempt!")
                cur_goal_failed = True

        elif self.automatic:
            self.goal_pose = self.pose2d_to_rospose(*GOAL_POSES[self.cur_goal_idx])
            rospy.loginfo(f"Publishing goal pose {self.goal_pose}!")
            goal_pose_stamped = PoseStamped()
            goal_pose_stamped.header.stamp = rospy.Time.now()
            goal_pose_stamped.pose = self.goal_pose
            self.goal_pose_pub.publish(goal_pose_stamped)
            self.navigation_start_time = rospy.Time.now()
            self.report['start_poses'].append(self.robot_pose)
            self.report['goal_poses'].append(self.goal_pose)
            self.reset()

        # if reached goal or timeout, fill the report for this session, then reset variables
        if self.reached_goal() or cur_goal_failed:
            self.report['time_costs'].append(self.navigation_duration)
            self.report['collision_counts'].append(self.collision_counts)
            self.reset(reset_goal=True)
            rospy.loginfo(f"Navigation finished, waiting for new goal...")
            self.__call__()
            if self.automatic:
                self.cur_goal_idx += 1
                if self.cur_goal_idx == len(GOAL_POSES):
                    rospy.loginfo(f"All goal poses are finished!")
                    self.finished = True


if __name__ == '__main__':
    try:
        nav_metrics = NavigationMetrics(automatic=True)
    except rospy.ROSInterruptException:
        pass
    finally:
        nav_metrics()

