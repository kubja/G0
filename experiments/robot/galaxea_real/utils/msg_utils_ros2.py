import rclpy
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
import numpy as np


def act_to_joint(action, dummy = None):
    """
    Converts an action (np.ndarray or list of joint positions) into a JointState message.
    Ensures it's a flat list of Python floats.
    """
    # Ensure action is a numpy array (in case it's a list)
    if not isinstance(action, np.ndarray):
        action = np.array(action, dtype=float)

    # Flatten the action array just in case there are nested arrays
    action = action.flatten()

    # Convert each element to a pure Python float
    action_list = [float(x) for x in action]

    # Create ROS JointState message
    joint_msg = JointState()
    joint_msg.position = action_list  # Assign the action as position in the message

    return joint_msg


def threshold_input(value, thres=0.0):
    """
    Thresholds the input value to zero if it is below a certain threshold.
    This is useful for ignoring small values in the input to avoid disturbances.

    Args:
        value (float): The input value to threshold.
        thres (float): The minimum absolute value to allow. Defaults to 0.0.

    Returns:
        float: The value or 0.0 if under threshold.
    """
    return value if abs(value) > thres else 0.0


def act_to_twist(action, action_thres=0.0):
    """
    Converts an action array into a ROS2 TwistStamped message with optional thresholding.

    Args:
        action (iterable): A sequence of at least six numerical values representing the desired
            linear and angular velocities.
        action_thres (float, optional): The threshold value to be applied to the action inputs.
            Defaults to 0.0.

    Returns:
        TwistStamped: A ROS2 message with its timestamp set to the current time and its twist
            attributes (linear and angular) populated with thresholded values.
    """
    action_cmd_msg = TwistStamped()
    # Set timestamp using ROS2 time
    now = Time().to_msg()
    action_cmd_msg.header.stamp = now

    # Populate twist message fields with thresholded values
    action_cmd_msg.twist.linear.x = threshold_input(action[0], action_thres)
    action_cmd_msg.twist.linear.y = threshold_input(action[1], action_thres)
    action_cmd_msg.twist.linear.z = threshold_input(action[2], action_thres)
    action_cmd_msg.twist.angular.x = threshold_input(action[3], action_thres)
    action_cmd_msg.twist.angular.y = threshold_input(action[4], action_thres)
    action_cmd_msg.twist.angular.z = threshold_input(action[5], action_thres)
    return action_cmd_msg