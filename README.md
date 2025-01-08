## MuJoCo Simulator CPP

## Usage

You can install the appimage from [releases](https://github.com/JafarAbdi/mujoco_simulator/releases)

```bash
wget https://github.com/JafarAbdi/mujoco_simulator/releases/download/continuous/mujoco_simulator-x86_64.AppImage -O mujoco_simulator
chmod +x mujoco_simulator
```

## Examples

```bash
mujoco_simulator SCENE.xml
```

## Build the package locally

```bash
pixi run build
pixi run lint
```

## Build AppImage locally

```bash
pixi run build-appimage
```

## Acknowledgements

- The simulator is based on [MuJoCo's simulate](https://github.com/google-deepmind/mujoco/tree/main/simulate) with some modifications. See commit history for what exactly was changed.
