import time

import zenoh


# Add query to joint names and others
class MuJoCoInterface:

    def __init__(self):
        self._session = zenoh.open(zenoh.Config())

    def reset(self):
        replies = self._session.get("reset", payload=zenoh.ext.z_serialize(True))


def main():
    zenoh.init_log_from_env_or("error")
    mujoco_interface = MuJoCoInterface()
    mujoco_interface.reset()

    # conf = zenoh.Config()
    # with zenoh.open(conf) as session:
    #
    #     def listener(sample: zenoh.Sample):
    #         print(
    #             f">> [Subscriber] Received {sample.kind} ('{sample.key_expr}': '{zenoh.ext.z_deserialize(list[float], sample.payload)}')"
    #         )
    #
    #     session.declare_subscriber("robot/qpos", listener)
    #
    #     print("Press CTRL-C to quit...")
    #     while True:
    #         time.sleep(1)
    #


if __name__ == "__main__":
    main()
