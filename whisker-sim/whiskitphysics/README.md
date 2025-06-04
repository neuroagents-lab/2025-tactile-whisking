# WHISKiT Whisker Simulator

This is a modified version of [WHISKiT Physics](https://github.com/SeNSE-lab/whiskitphysics).

The main simulation code is located at [code/src/Simulation.cpp](./code/src/Simulation.cpp).

Changes from original WHISKiT:
- Uses mouse whisker model (data from [Bresee et al. [2023]](https://github.com/SeNSE-lab/BreseeEtAl_JEB2023_dataFiles)) instead of rat
- Load objects from ShapeNet
  - Additional parameters for adjusting object position, rotation, and velocity.
- GUI: Keyboard shortcuts
  - Camera settings
    - `o` for orthographic projection
    - `p` for perspective projection
    - `1`~`4` for different camera views/positions
  - `q` to quit

## Installation

Linux: `sudo apt-get install freeglut3-dev libboost-all-dev`
MacOS: `brew install freeglut boost`

```bash
mkdir build
cd build
cmake ..
make
```
