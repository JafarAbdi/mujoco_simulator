"""Example of using the differential_ik functionality."""

import logging
import pathlib
import time

import zenoh
from rich.logging import RichHandler
from loop_rate_limiters import RateLimiter

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

robot = Robot(pathlib.Path("robots/panda/configs.toml"))
# robot = Robot(pathlib.Path("robots/ur5e/configs.toml"))
# robot = Robot(pathlib.Path("robots/kinova/configs.toml"))

mujoco_interface = MuJoCoInterface()
mujoco_interface.reset(keyframe="home")

rate = RateLimiter(25, warn=False)
while True:
    mocap = mujoco_interface.mocap()
    target_joint_positions = robot.ik(
        mocap[0] + mocap[1],
        robot.from_mj_joint_positions(mujoco_interface.qpos()),
    )
    if target_joint_positions is None:
        LOGGER.info("IK failed")
    else:
        mujoco_interface.ctrl(target_joint_positions + [0.0])
    rate.sleep()
