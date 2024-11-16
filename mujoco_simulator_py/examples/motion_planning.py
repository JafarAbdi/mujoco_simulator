"""Example of using the MotionPlanner functionality."""

import pathlib
import time

import zenoh

from mujoco_simulator_py.mujoco_interface import MuJoCoInterface
from mujoco_simulator_py.robot import MotionPlanner, Robot

SLEEP = 0.025

zenoh.init_log_from_env_or("error")

robot = Robot(pathlib.Path("robots/panda/configs.toml"))

group_name = "arm"
mujoco_interface = MuJoCoInterface()
mujoco_interface.reset(model_filename=robot.model_filename, keyframe="home")

start_state = robot.from_mj_joint_positions(mujoco_interface.qpos())
goal_state = [0.0, -0.785, 0.5, -2.356, 0.5, 1.5, 1.2]
planner = MotionPlanner(robot)

if path := planner.plan(start_state, goal_state):
    for state in path:
        mujoco_interface.ctrl(state + [0.0])
        time.sleep(SLEEP)

current_joint_positions = robot.from_mj_joint_positions(mujoco_interface.qpos())
target_joint_positions = robot.ik(
    [0.2, 0.5, 0.4, 0.0, 1.0, 0.0, 0.0],
    current_joint_positions,
)
if target_joint_positions is None:
    raise ValueError("No IK solution found")
if path := planner.plan(
    current_joint_positions,
    target_joint_positions,
):
    for state in path:
        mujoco_interface.ctrl(state + [0.0])
        time.sleep(SLEEP)
