"""Python interface to mujoco_simulator."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import zenoh
from rich.logging import RichHandler

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=os.getenv("LOG_LEVEL", "INFO").upper())


@dataclass
class AttachModelRequest:
    model_filename: str
    site_name: str
    pos: list[float]
    quat: list[float]
    prefix: str
    suffix: str


class MuJoCoInterface:
    """Python interface to mujoco_simulator."""

    def __init__(self):
        """Initialize the interface."""
        self._session = zenoh.open(zenoh.Config())
        self._qpos_subscriber = self._session.declare_subscriber(
            "robot/qpos",
            zenoh.handlers.RingChannel(1),
        )
        self._qvel_subscriber = self._session.declare_subscriber(
            "robot/qvel",
            zenoh.handlers.RingChannel(1),
        )
        self._mocap_subscriber = self._session.declare_subscriber(
            "robot/mocap",
            zenoh.handlers.RingChannel(1),
        )
        self._ctrl_publisher = self._session.declare_publisher("robot/ctrl")

    def attach_model(
        self,
        attach_model_request: AttachModelRequest,
    ):
        """Send a reset request to the simulator.

        Args:
            model_filename: The filename of the model to load.
            keyframe: The name of the keyframe to use after loading/resetting the model.

        Raises:
            RuntimeError: If the reset request fails.
        """
        attach_model_request.model_filename = str(
            Path(attach_model_request.model_filename).resolve(),
        )
        replies = list(
            self._session.get(
                "attach_model",
                payload=zenoh.ZBytes(
                    json.dumps(attach_model_request.__dict__).encode(),
                ),
            ),
        )
        assert len(replies) == 1
        ok = zenoh.ext.z_deserialize(bool, replies[0].ok.payload)
        if not ok:
            msg = f"Failed to reset the simulation: {error_msg}"
            raise RuntimeError(msg)

    def reset(self, *, model_filename: str | None = None, keyframe: str | None = None):
        """Send a reset request to the simulator.

        Args:
            model_filename: The filename of the model to load.
            keyframe: The name of the keyframe to use after loading/resetting the model.

        Raises:
            RuntimeError: If the reset request fails.
        """
        attachments = {}
        if model_filename:
            attachments["model_filename"] = str(Path(model_filename).resolve())
        if keyframe:
            attachments["keyframe"] = keyframe
        replies = list(
            self._session.get(
                "reset",
                payload=zenoh.ext.z_serialize(obj=True),
                attachment=zenoh.ext.z_serialize(attachments),
            ),
        )
        assert len(replies) == 1
        ok, error_msg = zenoh.ext.z_deserialize(tuple[bool, str], replies[0].ok.payload)
        if not ok:
            msg = f"Failed to reset the simulation: {error_msg}"
            raise RuntimeError(msg)

    def get_model_filename(self):
        """Get the filename of the model loaded in the simulator.

        Returns:
            The filename of the model loaded in the simulator.

        Raises:
            RuntimeError: If the request fails.
        """
        replies = list(self._session.get("model"))
        assert len(replies) == 1
        try:
            return replies[0].ok.payload.to_string()
        except Exception:
            LOGGER.exception("Failed to get the model filename")
            return None

    def qpos(self):
        """Get the current qpos.

        Returns:
            The current qpos.
        """
        while (qpos := self._qpos_subscriber.try_recv()) is None:
            pass
        return zenoh.ext.z_deserialize(list[float], qpos.payload)

    def qvel(self):
        """Get the current qvel.

        Returns:
            The current qvel.
        """
        while (qvel := self._qvel_subscriber.try_recv()) is None:
            pass
        return zenoh.ext.z_deserialize(list[float], qvel.payload)

    def mocap(self):
        """Get the current mocap.

        Returns:
            The current mocap as ([x, y, z], [qw, qx, qy, qz]).
        """
        while (mocap := self._mocap_subscriber.try_recv()) is None:
            pass
        return zenoh.ext.z_deserialize(tuple[list[float], list[float]], mocap.payload)

    def ctrl(self, ctrl: dict[int, float]):
        """Send ctrl to the simulator.

        Args:
            ctrl: The control signal to send to the simulator.
        """
        self._ctrl_publisher.put(zenoh.ext.z_serialize(ctrl))
