"""Example of using the differential_ik functionality."""

import logging
import pathlib
import time

import zenoh
from rich.logging import RichHandler

from mujoco_simulator_py.mujoco_interface import MuJoCoInterface
from mujoco_simulator_py.robot import Robot

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
LOGGER = logging.getLogger(__name__)

zenoh.init_log_from_env_or("error")

robot = Robot(pathlib.Path("robots/rrr/configs.toml"))

mujoco_interface = MuJoCoInterface()
mujoco_interface.reset(model_filename=robot.model_filename)

for target_pose in [
    [-0.4, 0.0, 0.2, 0.0, 0.0, 1.0, 0.0],
    [0.4, 0.0, 0.2, 0.0, 0.0, 1.0, 0.0],
    [0.4, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
]:
    target_joint_positions = robot.ik(
        target_pose,
        mujoco_interface.qpos(),
    )
    if target_joint_positions is None:
        LOGGER.info("IK failed")
    else:
        LOGGER.info(f"IK succeeded: {target_joint_positions}")
        current_target_pose = robot.get_frame_pose(
            target_joint_positions,
            robot.groups["arm"].tcp_link_name,
        )
        LOGGER.info(
            f"TCP Pose for target joint positions: {current_target_pose}",
        )
        mujoco_interface.ctrl(target_joint_positions)
    time.sleep(1.0)
