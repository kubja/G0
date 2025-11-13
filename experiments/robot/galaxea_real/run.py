from pathlib import Path
import numpy as np
import time
import torch
import threading
import queue
import tyro
import rclpy  # ROS 2 import

from robot_interface_ros2 import GalaxeaInferfaceConfig
from galaxea_real_utils import get_wrapped_env
from experiments.policy_r1_lite import PiZeroPolicy


def main(
    interface_config: GalaxeaInferfaceConfig,
    run_dir: Path,
    ckpt_id: int,
    num_action_steps: int = 32,
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
        # Parallel Inference Setup
        # -----------------------------
        action_queue = queue.Queue(maxsize=1)
        stop_flag = threading.Event()

        def inference_thread():
            nonlocal obs, last_action
            print("Inference thread started.")
            while not stop_flag.is_set():
                instruction = INSTRUCTION_PATH.read_text().strip() if INSTRUCTION_PATH.exists() else ""
                if instruction in ['', 'nothing']:
                    time.sleep(0.1)
                    continue

                obs_copy = obs.copy()
                obs_copy["last_action"] = last_action

                try:
                    with torch.inference_mode():
                        new_action = policy.infer(obs=obs_copy, instruction=instruction)
                except Exception as e:
                    print(f"[Inference Thread] Error: {e}")
                    time.sleep(0.5)
                    continue

                # Keep only the newest action
                if not action_queue.empty():
                    try:
                        action_queue.get_nowait()
                    except queue.Empty:
                        pass
                action_queue.put(new_action)

        infer_thread = threading.Thread(target=inference_thread, daemon=True)
        infer_thread.start()

        # -----------------------------
        # Main Control Loop
        # -----------------------------
        print("Starting control loop...")

        while not env.is_close():
            try:
                if obs is None:
                    time.sleep(0.1)
                    obs = env.get_observations()
                    continue

                # Try to get latest action (non-blocking)
                if not action_queue.empty():
                    action = action_queue.get_nowait()
                else:
                    action = [last_action] * num_action_steps  # fallback

                for i in range(num_action_steps):
                    obs = env.step(action[i])
                    last_action = action[i]

            except KeyboardInterrupt:
                print("Interrupted by user.")
                break
            except Exception as e:
                print(f"[Control Loop] Error: {e}")
                time.sleep(0.2)

    finally:
        # Signal stop and shutdown
        stop_flag.set()
        infer_thread.join(timeout=2.0)
        rclpy.shutdown()
        print("ROS 2 shutdown complete.")


if __name__ == '__main__':
    tyro.cli(main)
