# robot_interface_ros2.py

from collections import deque
from dataclasses import dataclass, field
from functools import partial
import time
from typing import Dict, Literal

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import TwistStamped
from loguru import logger  # optional; you can use self.get_logger() instead

R1_LITE = "R1_LITE"
R1 = "R1"


# ------------------------------
# Config Data Classes
# ------------------------------
@dataclass
class R1LiteTopicsConfig:
    joint_state_input: Dict[str, str] = field(
        default_factory=lambda: {
            "left_arm": "/hdas/feedback_arm_left",
            "left_gripper": "/hdas/feedback_gripper_left",
            "right_arm": "/hdas/feedback_arm_right",
            "right_gripper": "/hdas/feedback_gripper_right",
            "torso": "/hdas/feedback_torso",
            "chassis": "/hdas/feedback_chassis"
        }
    )
    image_input: Dict[str, str] = field(
        default_factory=lambda: {
            "head_rgb": "/hdas/camera_head/left_raw/image_raw_color/compressed",
            "left_rgb": "/hdas/camera_wrist_left/color/image_rect_raw/compressed",
            "right_rgb": "/hdas/camera_wrist_right/color/image_rect_raw/compressed"
        }
    )
    joint_state_output: Dict[str, str] = field(
        default_factory=lambda: {
            "left_arm": "/motion_target/target_joint_state_arm_left",
            "left_gripper": "/motion_target/target_position_gripper_left",
            "right_arm": "/motion_target/target_joint_state_arm_right",
            "right_gripper": "/motion_target/target_position_gripper_right"
        }
    )
    twist_output: Dict[str, str] = field(
        default_factory=lambda: {
            "torso": "/motion_target/target_speed_torso",
            "chassis": "/motion_target/target_speed_chassis"
        }
    )


@dataclass
class GalaxeaInferfaceConfig:
    hardware: Literal['R1', 'R1_LITE'] = R1_LITE
    topic: R1LiteTopicsConfig = field(default_factory=R1LiteTopicsConfig)
    msg_time_diff_threshold: float = 1
    with_torso: bool = True
    with_chassis: bool = True
    camera_deque_length: int = 7
    deque_length: int = 200
    control_freq: int = 15
    dry_run: bool = False
    torso_chassis_thres: float = 0.01


# ------------------------------
# Main ROS2 Node
# ------------------------------
class GalaxeaInterface(Node):
    def __init__(self, config: GalaxeaInferfaceConfig):
        super().__init__('galaxea_real')
        self.config = config

        logger.info("Initialized ROS2 node [galaxea_real]")
        self.br = CvBridge()
        self.inputs_dict = {}  # Stores deque for each topic
        self.last_camera_time = 0.0
        self.lastest_instruction = ""
        self._pubs = {}
        self._subs = {}

        self._init_topics()
        time.sleep(1)

    # ------------------------------
    # Callbacks
    # ------------------------------
    def _camera_callback(self, msg: CompressedImage, que: deque, topic: str):
        #logger.info(f"[Callback] Camera topic triggered: {topic}")
        img_cv_bgr = self.br.compressed_imgmsg_to_cv2(msg)
        if len(img_cv_bgr.shape) == 3 and img_cv_bgr.shape[2] == 3:
            img_cv = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2RGB)
        elif len(img_cv_bgr.shape) == 3 and img_cv_bgr.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGRA2RGBA)
        else:
            raise ValueError(f"Unexpected image format: {img_cv_bgr.shape}")

        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        que.append(dict(data=img_cv, message_time=msg_time))

    def _joint_states_callback(self, msg: JointState, que: deque, topic: str):
        #logger.info(f"[Callback] JointState topic triggered: {topic}")
        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        que.append(dict(
            position=np.array(msg.position, dtype=np.float32),
            velocity=np.array(msg.velocity, dtype=np.float32),
            message_time=msg_time
        ))

    # ------------------------------
    # Topic Setup
    # ------------------------------
    def _init_topics(self):
        config_topic = self.config.topic

        # JointState & Image subscribers
        for topic_dict, topic_type, topic_callback, deque_maxlen in zip(
            [config_topic.joint_state_input, config_topic.image_input],
            [JointState, CompressedImage],
            [self._joint_states_callback, self._camera_callback],
            [self.config.deque_length, self.config.camera_deque_length],
        ):
            for topic, topic_name in topic_dict.items():
                torso_flag = "torso" not in topic or self.config.with_torso
                chassis_flag = "chassis" not in topic or self.config.with_chassis
                if torso_flag and chassis_flag:
                    self.inputs_dict[topic] = deque(maxlen=deque_maxlen)
                    callback = partial(topic_callback, que=self.inputs_dict[topic], topic=topic)
                    self.create_subscription(topic_type, topic_name, callback, 10)
                    logger.info(f"Subscriber {topic} created on {topic_name}")

        # Publishers
        for topic_dict, topic_type in zip(
            [config_topic.joint_state_output, config_topic.twist_output],
            [JointState, TwistStamped],
        ):
            for topic, topic_name in topic_dict.items():
                torso_flag = "torso" not in topic or self.config.with_torso
                chassis_flag = "chassis" not in topic or self.config.with_chassis
                if torso_flag and chassis_flag:
                    self._pubs[topic] = self.create_publisher(topic_type, topic_name, 10)
                    logger.info(f"Publisher {topic} created on {topic_name}")

    # ------------------------------
    # Data Lookup / Sync
    # ------------------------------
    def find_nearest_message(self, topic_name, timestamp):
        min_diff = float("inf")
        nearest_msg = None
        data_queue = list(self.inputs_dict[topic_name])
        for msg in data_queue:
            diff = abs(msg['message_time'] - timestamp)
            if diff < min_diff:
                nearest_msg = msg
                min_diff = diff
        return nearest_msg

    def lookup_by_camera_under_tolerance(self, camera_timestamp, threshold):
        msgs = {}
        for topic, que in self.inputs_dict.items():
            if topic == "head_rgb":
                if len(que) == 0:
                    logger.warning(f"No head camera messages yet")
                    return None
                msgs[topic] = que[-1]
            else:
                msg = self.find_nearest_message(topic, camera_timestamp)
                if msg is None:
                    logger.warning(f'Channel: {topic} has no messages')
                    return None
                time_diff = abs(msg['message_time'] - camera_timestamp)
                if time_diff > threshold:
                    if topic not in ['left_rgb', 'right_rgb']:
                        logger.warning(f'Channel {topic} out of sync (Δt={time_diff:.3f}s > {threshold}s)')
                        return None
                msgs[topic] = msg
        return msgs

    def get_observations(self):
        if "head_rgb" not in self.inputs_dict or len(self.inputs_dict["head_rgb"]) == 0:
            logger.warning("No camera_head message")
            return None

        latest_camera_msg = self.inputs_dict["head_rgb"][-1]
        latest_camera_time = latest_camera_msg['message_time']

        if latest_camera_time <= self.last_camera_time:
            logger.warning("No new head camera frame.")
            return None

        obs = self.lookup_by_camera_under_tolerance(latest_camera_time, self.config.msg_time_diff_threshold)
        if obs is None:
            logger.warning("Failed to get synchronized messages")
            return None

        self.last_camera_time = latest_camera_time
        return obs

    # ------------------------------
    # Action Publishing
    # ------------------------------
    def _publish_action(self, action_dict):
        from utils import msg_utils_ros2  # keep your existing utils

        config_topic = self.config.topic
        for topic_dict, topic_fn in zip(
            [config_topic.joint_state_output, config_topic.twist_output],
            [msg_utils_ros2.act_to_joint, msg_utils_ros2.act_to_twist],
        ):
            for topic, _ in topic_dict.items():
                if topic in action_dict:
                    if not self.config.dry_run:
                        msg = topic_fn(action_dict[topic],
                                       self.config.torso_chassis_thres if 'torso' in topic or 'chassis' in topic else None)
                        self._pubs[topic].publish(msg)

    # ------------------------------
    # Step / Loop
    # ------------------------------
    def step(self, action_dict):
        self._publish_action(action_dict)
        obs = self.get_observations()
        time.sleep(1.0 / self.config.control_freq)
        return obs

    def get_latest_instruction(self):
        return self.lastest_instruction

    def is_close(self) -> bool:
        """
        Returns True if the robot interface should stop or is shutting down.
        """
        # Example: check if ROS node is shutting down or some internal flag
        return not rclpy.ok()  # True if ROS2 context is shut down