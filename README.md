## MuJoCo Simulator CPP

## Usage

You can install the appimage from [releases](https://github.com/JafarAbdi/mujoco_simulator/releases)

## Examples

```bash
mujoco_simulator SCENE.xml
```

## Build the package locally

```bash
cd mujoco_simulator_cpp/
pixi run build
pixi run lint
```

## Build AppImage locally

```bash
pixi run build-appimage
```

## Acknowledgements

- The simulator is based on [MuJoCo's simulate](https://github.com/google-deepmind/mujoco/tree/main/simulate) with some modifications. See commit history for what exactly was changed.
