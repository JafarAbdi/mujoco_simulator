## MuJoCo Simulator CPP

```bash
cd mujoco_simulator_cpp/
pixi run install-zenoh-cpp
pixi run build
pixi run lint
```

## Examples

```bash
pixi run build && .build/mujoco/bin/mujoco_simulator .build/mujoco/_deps/mujoco-src/model/car/car.xml
```

## MuJoCo Simulator Python

## Acknowledgements

- The simulator is based on [MuJoCo's simulate](https://github.com/google-deepmind/mujoco/tree/main/simulate) with some modifications. See commit history for what exactly was changed.
