from pathlib import Path
import numpy as np
import time
import torch
import threading
import tyro
import rclpy  # ROS 2 import

from robot_interface_ros2 import GalaxeaInferfaceConfig
from galaxea_real_utils import get_wrapped_env
from experiments.policy_r1_lite import PiZeroPolicy


def main(
    interface_config: GalaxeaInferfaceConfig,
    run_dir: Path,
    ckpt_id: int,
    num_action_steps: int = 16,
    dtype: str = 'fp32'
):
    # Initialize ROS 2
    rclpy.init()

    try:
        # -----------------------------
        # Setup environment and policy
        # -----------------------------
        INSTRUCTION_PATH = Path(run_dir) / "instruction.txt"
        env = get_wrapped_env(interface_config)

        # Start spinning the actual ROS 2 node in a background thread
        spin_thread = threading.Thread(target=rclpy.spin, args=(env.node,), daemon=True)
        spin_thread.start()

        policy = PiZeroPolicy(
            cfg_file=str(run_dir / "config.yaml"),
            checkpoint_path=str(run_dir / f"model_{ckpt_id}.pt"),
            dtype=dtype,
        )

        input("Press Enter to start the robot...")

        # Wait for the first observations to arrive
        obs = env.get_observations()
        while obs is None:
            time.sleep(0.1)
            obs = env.get_observations()

        # Initialize last action
        last_action = np.concatenate(
            [
                obs["/hdas/feedback_arm_left"]["position"],
                obs["/hdas/feedback_gripper_left"]["position"],
                obs["/hdas/feedback_arm_right"]["position"],
                obs["/hdas/feedback_gripper_right"]["position"],
                np.zeros(12, dtype=np.float32)
            ]
        )

        # -----------------------------
        # Main control loop
        # -----------------------------
        while not env.is_close():
            if obs is None:
                time.sleep(0.1)
                obs = env.get_observations()
                continue

            # Read instruction safely
            instruction = INSTRUCTION_PATH.read_text().strip() if INSTRUCTION_PATH.exists() else ""

            obs["last_action"] = last_action
            if instruction in ['', 'nothing']:
                obs = None
                continue

            with torch.inference_mode():
                action = policy.infer(obs=obs, instruction=instruction)

            for i in range(num_action_steps):
                obs = env.step(action[i])
                last_action = action[i]

    finally:
        # Clean shutdown
        rclpy.shutdown()
        print("ROS 2 shutdown complete.")


if __name__ == '__main__':
    tyro.cli(main)
