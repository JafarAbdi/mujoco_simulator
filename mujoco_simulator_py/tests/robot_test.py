"""Test the Robot class."""

import pathlib

import numpy as np
import pytest
import transforms3d

from mujoco_simulator_py.robot import (
    MissingAccelerationLimitError,
    MissingBaseLinkError,
    MissingJointError,
    MotionPlanner,
    Robot,
    RobotDescriptionNotFoundError,
)

FILE_PATH = pathlib.Path(__file__).parent


def test_no_gripper():
    """Test the Robot class with no gripper and tcp_link."""
    robot = Robot(FILE_PATH / ".." / "robots" / "rrr" / "configs.toml")
    assert robot.base_link == "base_link"
    assert robot.groups["arm"].tcp_link_name == "end_effector"
    assert robot.groups["arm"].joints == ["joint1", "joint2", "joint3"]
    assert robot.groups["arm"].gripper is None
    target_pose = [0.4, 0.0, 0.2, 0.0, 0.0, 1.0, 0.0]
    target_joint_positions = robot.ik(
        target_pose,
        robot.named_states["home"],
    )
    assert target_joint_positions is not None
    pose = robot.get_frame_pose(
        target_joint_positions,
        robot.groups["arm"].tcp_link_name,
    )

    assert np.allclose(
        transforms3d.affines.compose(
            target_pose[:3],
            transforms3d.quaternions.quat2mat(
                [target_pose[3], target_pose[4], target_pose[5], target_pose[6]],
            ),
            np.ones(3),
        ),
        pose,
        atol=1e-3,
    )

    # Should fail, rrr robot has end_effector_joint as revolute joint with [0.0, 0.0] limits
    target_pose = [0.4, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
    target_joint_positions = robot.ik(
        target_pose,
        robot.named_states["home"],
    )
    assert target_joint_positions is None


def test_motion_planning():
    """Test motion planning interface."""
    robot = Robot(FILE_PATH / ".." / "robots" / "rrr" / "configs.toml")
    start_state = np.asarray([0.0, 2.0, 1.0])
    goal_state = np.asarray([0.0, 1.0, 1.0])
    planner = MotionPlanner(robot)
    plan = planner.plan(start_state, goal_state)
    assert plan is not None, "Expected a plan to be found"


def test_no_gripper_and_tcp_link():
    """Test the Robot class with no gripper and tcp_link."""
    robot = Robot(FILE_PATH / ".." / "robots" / "acrobot" / "configs.toml")
    assert robot.base_link == "world"
    assert robot.groups["arm"].tcp_link_name is None
    assert robot.groups["arm"].joints == ["elbow"]
    assert robot.groups["arm"].gripper is None
    assert robot.named_states["home"] == [0.0]


def test_robot():
    """Test the Robot class."""
    robot = Robot(FILE_PATH / ".." / "robots" / "panda" / "configs.toml")
    assert robot.base_link == "link0"
    assert robot.groups["arm"].tcp_link_name == "hand"
    assert robot.groups["arm"].joints == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    gripper = robot.groups["arm"].gripper
    assert gripper.actuated_joint == "finger_joint1"
    assert robot.joint_names == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
    ]

    # Test non-existing base link
    with pytest.raises(MissingBaseLinkError):
        robot = Robot(FILE_PATH / "non_existing_base_link.toml")

    # Test non-existing group joint
    with pytest.raises(MissingJointError):
        robot = Robot(FILE_PATH / "non_existing_group_joint.toml")


ROBOTS = ["panda", "kinova", "ur5e"]


@pytest.mark.parametrize("robot_name", ROBOTS)
def test_ik(robot_name):
    """Test the ik function."""
    robot = Robot(
        pathlib.Path(
            FILE_PATH / ".." / "robots" / robot_name / "configs.toml",
        ),
    )
    seed_joint_positions = robot.named_states["home"][
        :-1
    ]  # TODO: Remove this hack (This is to remove the gripper joint)
    # TODO: Add a loop to check for 100 different poses
    # Use fk to check if the pose is actually same as the input one
    target_joint_positions = robot.ik(
        [0.2, 0.2, 0.2, 1.0, 0.0, 0.0, 0.0],
        seed_joint_positions,
    )
    assert target_joint_positions is not None

    # Outside workspace should fail
    target_joint_positions = robot.ik(
        [2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        seed_joint_positions,
    )
    assert target_joint_positions is None


def test_robot_descriptions():
    """Test robot description."""
    with pytest.raises(RobotDescriptionNotFoundError):
        Robot(
            pathlib.Path(
                FILE_PATH / "non_existing_robot_descriptions.toml",
            ),
        )
    Robot(
        pathlib.Path(
            FILE_PATH / ".." / "robots" / "panda" / "configs.toml",
        ),
    )


def test_acceleration_limits():
    """Test the Robot class with missing joint acceleration limits."""
    with pytest.raises(MissingAccelerationLimitError):
        Robot(
            pathlib.Path(
                FILE_PATH / "extra_acceleration_joint.toml",
            ),
        )
    with pytest.raises(MissingAccelerationLimitError):
        Robot(
            pathlib.Path(
                FILE_PATH / "missing_acceleration_joint.toml",
            ),
        )
