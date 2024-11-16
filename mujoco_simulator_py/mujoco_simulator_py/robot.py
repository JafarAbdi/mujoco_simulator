"""A robot class based on mujoco."""

import logging
import os
import platform
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import ompl
import toml
import trac_ik_py
from ompl import base as ob
from ompl import geometric as og
from rich import pretty
from rich.logging import RichHandler

# TODO: Remove hardcoded group_name
GROUP_NAME = "arm"


def get_ompl_log_level(level: str) -> ompl.util.LogLevel:
    """Get OMPL log level.

    Args:
        level: Log level

    Returns:
        OMPL log level
    """
    level = level.upper()
    # Why using ompl.util.LOG_DEBUG doesn't work on MacOS?
    logger_module = ompl.base if platform.system() == "Darwin" else ompl.util
    if level == "DEBUG":
        return logger_module.LOG_DEBUG
    if level == "INFO":
        return logger_module.LOG_INFO
    if level == "WARN":
        return logger_module.LOG_WARN
    if level == "ERROR":
        return logger_module.LOG_ERROR
    msg = f"Unknown log level: {level}"
    raise ValueError(msg)


ompl.util.setLogLevel(get_ompl_log_level(os.getenv("LOG_LEVEL", "ERROR")))

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=os.getenv("LOG_LEVEL", "INFO").upper())


def get_ompl_planners() -> list[str]:
    """Get OMPL planners.

    Returns:
        List of OMPL planners.
    """
    from inspect import isclass

    module = ompl.geometric
    planners = []
    for obj in dir(module):
        planner_name = f"{module.__name__}.{obj}"
        planner = eval(planner_name)  # noqa: S307
        if isclass(planner) and issubclass(planner, ompl.base.Planner):
            planners.append(
                planner_name.split("ompl.geometric.")[1],
            )  # Name is ompl.geometric.<planner>
    return planners


# Differential-IK Hyperparameters
MAX_ITERATIONS = 500
MAX_ERROR = 1e-3


def filter_values_by_joint_names(
    keys: list[str],
    values: list[float],
    joint_names: list[str],
) -> list[float]:
    """Filter values by joint names."""
    filtered_values = []
    for joint_name in joint_names:
        try:
            index = keys.index(joint_name)
        except ValueError:
            msg = f"Joint name '{joint_name}' not in input keys {keys}"
            raise ValueError(msg) from None
        filtered_values.append(values[index])
    return filtered_values


@dataclass(slots=True)
class Gripper:
    """Gripper configs.

    Args:
        open_value: Open position
        close_value: Close position
        actuated_joint: Name of the actuated joint
    """

    open_value: float
    close_value: float
    actuated_joint: str


@dataclass(slots=True)
class Group:
    """Group configs.

    Args:
        joints: List of joint names
        tcp_link_name (optional): Name of the TCP link
        gripper (optional): Gripper configs
    """

    joints: list[str]
    tcp_link_name: str | None = None
    gripper: Gripper | None = None


class MissingGroupError(Exception):
    """Missing group error."""


# TODO: Rename?
class MissingBaseLinkError(Exception):
    """Missing base link error."""


# TODO: Rename?
class MissingJointError(Exception):
    """Missing joint error."""


class MissingGripperError(Exception):
    """Missing gripper error."""


class RobotDescriptionNotFoundError(Exception):
    """Robot description error."""

    def __init__(self, robot_description_filename):
        """Init.

        Args:
            robot_description_filename: Robot description filename
        """
        message = f"Model file does not exist: {robot_description_filename}"
        super().__init__(message)


class MissingVelocityLimitError(Exception):
    """Velocity limits error."""

    def __init__(self, joint_names, defined_joint_names):
        """Init.

        Args:
            joint_names: List of joint names
            defined_joint_names: List of defined joint names
        """
        message = f"Velocity limits only defined for {defined_joint_names} - Missing {set(joint_names) - set(defined_joint_names)}"
        super().__init__(message)


class MissingAccelerationLimitError(Exception):
    """Acceleration limits error."""

    def __init__(self, joint_names, defined_joint_names):
        """Init.

        Args:
            joint_names: List of joint names
            defined_joint_names: List of defined joint names
        """
        message = f"Acceleration limits only defined for {defined_joint_names} - Missing {set(joint_names) - set(defined_joint_names)}"
        super().__init__(message)


class Robot:
    """Robot base class."""

    def __init__(  # noqa: C901
        self,
        config_path: Path,
    ) -> None:
        """Init.

        Args:
            config_path: Path to the config file configs.toml
        """
        if not config_path.exists():
            msg = f"Config file does not exist: {config_path}"
            raise FileNotFoundError(msg)
        configs = toml.load(config_path)

        self.model_filename = config_path.parent / configs["robot"]["description"]
        if not self.model_filename.exists():
            raise RobotDescriptionNotFoundError(self.model_filename)

        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(
            str(self.model_filename),
        )
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        mimic_joint_indices = []
        mimic_joint_multipliers = []
        mimic_joint_offsets = []
        mimicked_joint_indices = []

        for obj1id, obj2id, equality_type, equality_solref in zip(
            self.model.eq_obj1id,
            self.model.eq_obj2id,
            self.model.eq_type,
            self.model.eq_solref,
            strict=False,
        ):
            # TODO: ID == INDEX???
            if mujoco.mjtEq(equality_type) == mujoco.mjtEq.mjEQ_JOINT:
                mimicked_joint_indices.append(self.model.joint(obj1id).qposadr)
                mimic_joint_indices.append(self.model.joint(obj2id).qposadr)
                mimic_joint_multipliers.append(
                    equality_solref[1],
                )  # TODO: Make index a variable
                mimic_joint_offsets.append(
                    equality_solref[0],
                )  # TODO: Make index a variable
        self.mimic_joint_indices = np.asarray(mimic_joint_indices)
        self.mimic_joint_multipliers = np.asarray(mimic_joint_multipliers)
        self.mimic_joint_offsets = np.asarray(mimic_joint_offsets)
        self.mimicked_joint_indices = np.asarray(mimicked_joint_indices)

        if LOGGER.level == logging.DEBUG:
            pretty.pprint(configs)

        self.link_names = [self.model.body(i).name for i in range(self.model.nbody)]

        self.base_link = configs["robot"]["base_link"]
        # TODO: Add a test with base_link not in link_names
        if self.base_link not in self.link_names:
            msg = f"Base link '{self.base_link}' not in link names: {self.link_names}"
            raise MissingBaseLinkError(
                msg,
            )

        self.groups = self._make_groups(configs)
        # Group -> Joint indices
        self.group_actuated_joint_indices = {}
        for group_name, group in self.groups.items():
            actuated_joint_indices = []
            for joint_name in group.joints:
                try:
                    joint = self.model.joint(joint_name)
                except KeyError as e:
                    msg = f"Joint '{joint_name}' not in model joints. {e}"
                    raise MissingJointError(msg) from e
                actuated_joint_indices.append(
                    joint.qposadr[0],
                )  # TODO: How about multi-dof joints?
            # TODO: Handle gripper actuated joint
            self.group_actuated_joint_indices[group_name] = np.asarray(
                actuated_joint_indices,
            )

        self.joint_names = []
        for group in self.groups.values():
            self.joint_names.extend(group.joints)
            if group.gripper is not None:
                self.joint_names.append(group.gripper.actuated_joint)

        self.named_states = self._make_named_states(configs)

        self.velocity_limits = self._make_velocity_limits(
            self.joint_names,
            configs["velocity_limits"],
        )

        self.acceleration_limits = self._make_acceleration_limits(
            self.joint_names,
            configs["acceleration_limits"],
        )

    @staticmethod
    def _make_velocity_limits(joint_names: list[str], velocity_limits_configs: dict):
        velocity_limits = []
        for joint_name in joint_names:
            if (
                joint_velocity_limit := velocity_limits_configs.get(joint_name)
            ) is None:
                raise MissingVelocityLimitError(
                    joint_names,
                    velocity_limits_configs.keys(),
                )
            velocity_limits.append(joint_velocity_limit)
        return np.asarray(velocity_limits)

    @staticmethod
    def _make_acceleration_limits(
        joint_names: list[str],
        acceleration_limits_configs: dict,
    ):
        acceleration_limits = []
        for joint_name in joint_names:
            if (
                joint_acceleration_limit := acceleration_limits_configs.get(joint_name)
            ) is None:
                raise MissingAccelerationLimitError(
                    joint_names,
                    acceleration_limits_configs.keys(),
                )
            acceleration_limits.append(joint_acceleration_limit)
        return np.asarray(acceleration_limits)

    @staticmethod
    def _make_gripper_from_configs(gripper_configs):
        if gripper_configs is None:
            return None
        return Gripper(
            open_value=gripper_configs["open"],
            close_value=gripper_configs["close"],
            actuated_joint=gripper_configs["actuated_joint"],
        )

    def _make_groups(self, configs):
        groups = {}
        mujoco_joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        for group_name, group_config in configs["group"].items():
            groups[group_name] = Group(
                joints=group_config["joints"],
                tcp_link_name=group_config.get("tcp_link_name"),
                gripper=self._make_gripper_from_configs(group_config.get("gripper")),
            )
            if groups[group_name].tcp_link_name is not None:
                assert (
                    groups[group_name].tcp_link_name in self.link_names
                ), f"Group {group_name} TCP link '{groups[group_name].tcp_link_name}' not in link names: {self.link_names}"
            if groups[group_name].gripper is not None:
                assert (
                    groups[group_name].gripper.actuated_joint in mujoco_joint_names
                ), f"Gripper's actuated joint '{groups[group_name].gripper.actuated_joint}' not in model joints: {mujoco_joint_names}"
            for joint_name in groups[group_name].joints:
                if joint_name not in mujoco_joint_names:
                    msg = f"Joint '{joint_name}' for group '{group_name}' not in model joints: {mujoco_joint_names}"
                    raise MissingJointError(
                        msg,
                    )
        return groups

    def _make_named_states(self, configs):
        named_states = {}
        for state_name, state_config in configs["named_states"].items():
            assert (
                len(state_config)
                == len(
                    self.joint_names,
                )
            ), f"Named state '{state_name}' has {len(state_config)} joint positions, expected {len(self.joint_names)} for {self.joint_names}"
            named_states[state_name] = state_config

        return named_states

    def check_collision(self, joint_positions, *, verbose=False):
        """Check if the robot is in collision with the given joint positions.

        Args:
            joint_positions: Joint positions of the robot.
            verbose: Whether to print the collision results.

        Returns:
            True if the robot is in collision, False otherwise.
        """
        data = mujoco.MjData(self.model)
        data.qpos = self.as_mj_joint_positions(joint_positions)
        mujoco.mj_forward(self.model, data)
        if verbose:
            contacts = Counter()
            for contact in data.contact:
                body1_id = self.model.geom_bodyid[contact.geom1]
                body1_name = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body1_id,
                )
                body2_id = self.model.geom_bodyid[contact.geom2]
                body2_name = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body2_id,
                )
                contacts[(body1_name, body2_name)] += 1
            LOGGER.debug(f"Contacts: {contacts}")
        return data.ncon > 0

    def from_mj_joint_positions(self, q):
        """Convert mujoco joint positions to joint positions."""
        joint_positions = np.copy(q)
        return joint_positions[self.group_actuated_joint_indices[GROUP_NAME]]

    def as_mj_joint_positions(self, joint_positions):
        """Convert joint positions to mujoco joint positions."""
        # TODO: Normalize/Denormalize gripper values
        q = self.model.qpos0.copy()
        q[self.group_actuated_joint_indices[GROUP_NAME]] = joint_positions
        if self.mimic_joint_indices.size != 0:
            q[self.mimic_joint_indices] = (
                q[self.mimicked_joint_indices] * self.mimic_joint_multipliers
                + self.mimic_joint_offsets
            )
        return q

    def get_frame_pose(self, joint_positions, target_frame_name):
        """Get the pose of a frame."""
        data: mujoco.MjData = mujoco.MjData(self.model)
        data.qpos = joint_positions
        mujoco.mj_forward(self.model, data)
        target_frame_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            target_frame_name,
        )
        transform = np.eye(4)
        transform[:3, :3] = data.xmat[target_frame_id].reshape(3, 3)
        transform[:3, 3] = data.xpos[target_frame_id]
        return transform

    def ik(self, target_pose, initial_configuration=None, iteration_callback=None):
        """Compute the inverse kinematics of the robot for a given target pose.

        Args:
            target_pose: The target pose [x, y, z, qw, qx, qy, qz] or 4x4 homogeneous transformation matrix
            initial_configuration: The initial configuration
            iteration_callback: Callback function after each iteration

        Returns:
            The joint positions for the target pose or None if no solution was found
        """
        assert self.groups[
            GROUP_NAME
        ].tcp_link_name, f"tcp_link_name is not defined for group '{GROUP_NAME}'"
        solver = IKSolver(
            self.model_filename,
            self.base_link,
            self.groups[GROUP_NAME].tcp_link_name,
        )
        return solver.ik(
            target_pose,
            initial_configuration,
        )

    def joint_limit_velocity_scaling_factor(self, velocities):
        """Compute scaling factor based on joint limits."""
        velocity_scaling_factors = [1.0]
        for velocity, max_velocity in zip(
            velocities,
            self.velocity_limits,
            strict=True,
        ):
            if abs(velocity) < 1e-3:
                continue
            velocity_scaling_factors.append(
                np.clip(velocity, -max_velocity, max_velocity) / velocity,
            )
        return np.min(velocity_scaling_factors)  # scaling override

    @property
    def position_limits(self) -> list[tuple[float, float]]:
        """Return the joint limits.

        Returns:
            List of tuples of (lower, upper) joint limits.
        """
        # TODO: Handle continuous joints (limited = 0)
        return self.model.jnt_range[self.group_actuated_joint_indices[GROUP_NAME]]

    @property
    def effort_limits(self) -> list[float]:
        """Return the effort limits.

        Returns:
            List of effort limits.
        """
        return self.model.effortLimit[self.group_actuated_joint_indices[GROUP_NAME]]


# TODO: Should we have a different collision model for the planning scene?
# Should contains the robot + objects???
# We will need a check collision function for the planning scene as well
class PlanningScene:
    """A class to represent the planning scene."""

    def __init__(self):
        """Initialize the planning scene."""


class IKSolver:
    """A wrapper for the TRAC-IK solver."""

    def __init__(self, mjcf_filename, base_link_name, tcp_link_name) -> None:
        """Initialize the IK solver.

        Args:
            mjcf_filename: The MJCF filename.
            base_link_name: The base link name.
            tcp_link_name: The TCP link name.
        """
        self._ik_solver = trac_ik_py.TRAC_IK(
            base_link_name,
            tcp_link_name,
            str(mjcf_filename),
            0.005,  # timeout
            1e-5,  # epsilon
            trac_ik_py.SolveType.Speed,
        )

    def tcp_pose(self, joint_positions):
        """Get the pose of the tcp_link as [x, y, z, rw, rx, ry, rz].

        Args:
            joint_positions: Input joint positions
        """
        assert len(joint_positions) == self._ik_solver.getNrOfJointsInChain()
        return self._ik_solver.JntToCart(joint_positions)

    def ik(self, target_pose, seed=None):
        """Compute the inverse kinematics of the robot for a given target pose.

        Args:
            target_pose: The target pose [x, y, z, qw, qx, qy, qz] of the tcp_link_name w.r.t. the base_link_name
            seed: IK Solver seed

        Returns:
            The joint positions for the target pose or None if no solution was found
        """
        assert len(seed) == self._ik_solver.getNrOfJointsInChain()
        joint_positions = self._ik_solver.CartToJnt(
            seed,
            target_pose,
            [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
        )
        if len(joint_positions) == 0:
            return None
        return joint_positions


class MotionPlanner:
    """A wrapper for OMPL planners."""

    def __init__(self, robot: Robot, planner=None) -> None:
        """Initialize the motion planner.

        Args:
            robot: The robot to plan for.
            planner: The planner to use. If None, RRTConnect is used.
        """
        self._robot = robot
        self._bounds = ob.RealVectorBounds(len(self._robot.joint_names))
        for i, (lower, upper) in enumerate(self._robot.position_limits):
            self._bounds.setLow(i, lower)
            self._bounds.setHigh(i, upper)
        self._space = ob.RealVectorStateSpace(len(self._robot.joint_names))
        self._space.setBounds(self._bounds)
        self._setup = og.SimpleSetup(self._space)
        if planner is None:
            planner = "RRTConnect"
        self._setup.setPlanner(self._get_planner(planner))
        self._setup.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.is_state_valid),
        )

    def _get_planner(self, planner):
        try:
            return eval(  # noqa: S307
                f"og.{planner}(self._setup.getSpaceInformation())",
            )
        except AttributeError:
            LOGGER.exception(
                f"Planner '{planner}' not found - Available planners: {get_ompl_planners()}",
            )
            raise

    def as_ompl_state(self, joint_positions):
        """Convert joint positions to ompl state."""
        assert len(joint_positions) == len(self._robot.groups[GROUP_NAME].joints)
        state = ob.State(self._space)
        for i, joint_position in enumerate(joint_positions):
            state[i] = joint_position
        return state

    def from_ompl_state(self, state):
        """Convert ompl state to joint positions."""
        return [state[i] for i in range(len(self._robot.groups[GROUP_NAME].joints))]

    def plan(
        self,
        start_joint_positions: list[float],
        goal_joint_positions: list[float],
    ) -> list[list[float]] | None:
        """Plan a trajectory from start to goal.

        Args:
            start_joint_positions: The start joint positions.
            goal_joint_positions: The goal joint positions.

        Returns:
            The trajectory as a list of joint positions or None if no solution was found.
        """
        assert len(start_joint_positions) == len(self._robot.groups[GROUP_NAME].joints)
        assert len(goal_joint_positions) == len(self._robot.groups[GROUP_NAME].joints)
        self._setup.clear()
        start = self.as_ompl_state(start_joint_positions)
        # TODO: We need to check the bounds as well
        if self._robot.check_collision(start_joint_positions):
            LOGGER.info("Start state is in collision")
            self._robot.check_collision(start_joint_positions, verbose=True)
        goal = self.as_ompl_state(goal_joint_positions)
        if self._robot.check_collision(goal_joint_positions):
            LOGGER.info("Goal state is in collision")
            self._robot.check_collision(goal_joint_positions, verbose=True)
        self._setup.setStartAndGoalStates(start, goal)
        solved = self._setup.solve()
        if not solved:
            LOGGER.info("Did not find solution!")
            return None
        self._setup.simplifySolution()
        path = self._setup.getSolutionPath()
        path.interpolate()
        solution = []
        for state in path.getStates():
            solution.append(self.from_ompl_state(state))
        LOGGER.info(f"Found solution with {len(solution)} waypoints")
        return solution

    def is_state_valid(self, state):
        """Check if the state is valid, i.e. not in collision or out of bounds.

        Args:
            state: The state to check.

        Returns:
            True if the state is valid, False otherwise.
        """
        return self._setup.getSpaceInformation().satisfiesBounds(
            state,
        ) and not self._robot.check_collision(self.from_ompl_state(state))
