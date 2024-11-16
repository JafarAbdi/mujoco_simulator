# MuJoCo Simulator python interface

## Introduction

TODO

## Installation

This project is managed by [pixi](https://pixi.sh).
You can install the package in development mode using:

```bash
git clone https://github.com/JafarAbdi/mujoco_simulator_py
cd mujoco_simulator_py

pixi run install-mujoco
pixi run trac-ik-install
pixi install -a
```

## Testing the package

```bash
pixi run test
```

## Linting the package

```bash
pixi run lint
```

## Examples

To change the logging verbosity

```bash
LOG_LEVEL=Debug pixi run python examples/...
```

To change the logging verbosity for trac-ik solver

```
SPDLOG_LEVEL=debug pixi run python XXX
```

## Acknowledgements

TODO
