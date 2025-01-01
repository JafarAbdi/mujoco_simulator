import zenoh
from loop_rate_limiters import RateLimiter
import pathlib

from mujoco_simulator_py.mujoco_interface import MuJoCoInterface, AttachModelRequest

FILE_PATH = pathlib.Path(__file__).parent


def main():
    zenoh.init_log_from_env_or("error")
    mujoco_interface = MuJoCoInterface()
    mujoco_interface.attach_model(
        AttachModelRequest(
            "/home/juruc/workspaces/robotics_playground/ramp/external/mujoco_menagerie/trs_so_arm100/so_arm100.xml",
            "right_attachment_site",
            [-0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            "right/",
            "",
        )
    )
    mujoco_interface.attach_model(
        AttachModelRequest(
            "/home/juruc/workspaces/robotics_playground/ramp/external/mujoco_menagerie/trs_so_arm100/so_arm100.xml",
            "left_attachment_site",
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            "left/",
            "",
        )
    )


if __name__ == "__main__":
    main()
