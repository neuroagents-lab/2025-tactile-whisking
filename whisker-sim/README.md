# Whisker Simulation

This project uses [WHISKiT Physics](https://github.com/SeNSE-lab/whiskitphysics) to simulate rat whiskers and [ShapeNet](https://shapenet.org) for objects.

## Steps to run

1. (Optional) Download [ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)
`git clone https://huggingface.co/datasets/ShapeNet/ShapeNetCore.git`
2. Build WHISKiT ([see instructions](./whiskitphysics/README.md))
3. Run simulation with the `whiskit_gui` or `whiskit` (headless) executable (see scripts in [whiskitphysics/scripts](./whiskitphysics/scripts/))
    - `./whiskit --help` to see all parameter options