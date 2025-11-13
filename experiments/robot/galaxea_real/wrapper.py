import numpy as np
from robot_interface_ros2 import GalaxeaInterface

class Wrapper:
    def __init__(self, interface: GalaxeaInterface):
        self.interface = interface
        self.node = interface

    def step(self, action):
        """
        Convert flat action vector into dictionary, send to interface, and wrap observation.
        """
        if action.shape[0] == (6 + 1) * 2 + 6 + 6:
            action_dict = dict(
                left_arm=action[:6],
                left_gripper=action[6:7],
                right_arm=action[7:13],
                right_gripper=action[13:14],
                torso=action[14:20],
                chassis=action[20:],
            )
        else:
            raise NotImplementedError(f"Unexpected action size: {action.shape[0]}")
        
        obs = self.interface.step(action_dict)
        return self.wrap_obs(obs)

    def wrap_obs(self, obs):
        """
        Convert raw ROS2 messages into a consistent dictionary format.
        Missing fields are filled with zeros.
        """
        if obs is None:
            return None

        # Ensure all keys exist
        defaults = {
            "left_arm": {"position": np.zeros(6, dtype=np.float32), "velocity": np.zeros(6, dtype=np.float32)},
            "right_arm": {"position": np.zeros(6, dtype=np.float32), "velocity": np.zeros(6, dtype=np.float32)},
            "left_gripper": {"position": np.zeros(1, dtype=np.float32)},
            "right_gripper": {"position": np.zeros(1, dtype=np.float32)},
            "torso": {"position": np.zeros(6, dtype=np.float32)},
            "chassis": {"position": np.zeros(3, dtype=np.float32)},
            "head_rgb": {"data": np.zeros((480, 640, 3), dtype=np.uint8)},
            "left_rgb": {"data": np.zeros((480, 640, 3), dtype=np.uint8)},
            "right_rgb": {"data": np.zeros((480, 640, 3), dtype=np.uint8)}
        }

        for k, v in defaults.items():
            if k not in obs or obs[k] is None:
                obs[k] = v

        # Map to expected observation dict
        obs_dict = {
            "head_rgb": obs["head_rgb"]["data"],
            "left_hand_rgb": obs["left_rgb"]["data"],
            "right_hand_rgb": obs["right_rgb"]["data"],
            "/hdas/feedback_arm_left": {
                "position": obs["left_arm"]["position"],
                "velocity": obs["left_arm"]["velocity"]
            },
            "/hdas/feedback_arm_right": {
                "position": obs["right_arm"]["position"],
                "velocity": obs["right_arm"]["velocity"]
            },
            "/hdas/feedback_gripper_left": {"position": obs["left_gripper"]["position"]},
            "/hdas/feedback_gripper_right": {"position": obs["right_gripper"]["position"]},
            "/hdas/feedback_torso": {"position": obs["torso"]["position"]},
            "/hdas/feedback_chassis": {"position": obs["chassis"]["position"]}
        }

        return obs_dict

    def get_observations(self):
        obs = self.interface.get_observations()
        return self.wrap_obs(obs)

    def is_close(self):
        return self.interface.is_close()

    def get_latest_instruction(self):
        return self.interface.get_latest_instruction()


if __name__ == "__main__":
    import time
    from robot_interface_ros2 import GalaxeaInterface, GalaxeaInterfaceConfig

    interface = GalaxeaInterface(GalaxeaInterfaceConfig())
    wrapper = Wrapper(interface)
    time.sleep(1)

    for i in range(10):
        obs = wrapper.get_observations()
        for k, v in obs.items():
            if isinstance(v, dict):
                print(k, {subk: subv.shape for subk, subv in v.items()})
            else:
                print(k, v.shape)
        time.sleep(0.1)