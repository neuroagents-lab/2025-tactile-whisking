import argparse
import random
import glob
import os

# see whisker_calculations.ipynb for POSITION and ORIENTATION
DEFAULT_FLAGS = "--SAVE 1 --HZ 1000 --SUBSTEPS 100 --OBJ_Y 16  --WHISKER_LINKS_PER_MM 1 --POSITION -6.934 -11.642 -2.598 --ORIENTATION 0 15.366 -16.198"

SWEEP_ORIENTATIONS = [0, 90]
OBJ_ROTATION = [0, 30]
OBJ_HEIGHTS = [-5, 0]
OBJ_DISTANCE = [5, 8]
OBJ_SPEED = 30
OBJ_SCALE = 40

# paths
SHAPENET_DIR = os.path.abspath("../shapenet/ShapeNetCore")
OUT_DIR = os.path.abspath("../output/sweeps")
WHISKIT = os.path.abspath("../whiskitphysics/build/whiskit")

def gen_sweep_set(shape, args, subfolder='train'):
    cmds = []
    out_dir = args.out + "/" + subfolder
    for theta in SWEEP_ORIENTATIONS:
        for rot in OBJ_ROTATION:
            for h in OBJ_HEIGHTS:
                for d in OBJ_DISTANCE:
                    cmds.append(
                        f"{WHISKIT}{'_gui' if args.gui else ''} {DEFAULT_FLAGS}"
                        + f' --dir_out "{out_dir}/{shape}/h{h}_d{d}_t{rot + theta}"'
                        + f' --OBJ_PATH "{SHAPENET_DIR}/{shape}/models/model_normalized.obj"'
                        + f" --OBJ_SCALE {OBJ_SCALE}"
                        + f" --OBJ_SPEED {OBJ_SPEED}"
                        + f" --OBJ_THETA {rot + theta}"
                        + f" --OBJ_X {d}"
                        + f" --OBJ_Z {h}"
                        + "\n"
                    )
    return cmds


def get_shape_id(shape_path):
    return os.path.join(
        os.path.basename(os.path.dirname(shape_path)), os.path.basename(shape_path)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate all sweep commands for each object in shapenet"
    )
    parser.add_argument(
        "-g", "--gui", action="store_true", help="run in gui mode for debugging"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="do not generate all sweeps"
    )
    parser.add_argument(
        "-o", "--out", type=str, help="output directory", default=OUT_DIR
    )
    args = parser.parse_args()

    filename = ""
    lines = []
    shape_paths = []
    # shape_paths = glob.glob(f"{SHAPENET_DIR}/*/*")
    # print(f"found {len(shape_paths)} shapes in {SHAPENET_DIR}")
    with open("object_choice.txt", "r") as f:
        shape_paths = [line.strip() for line in f.readlines()]
    print(f"{len(shape_paths)} shapes in object_choice.txt")

    if args.debug:
        filename = "run_sweeps_debug.sh"
        lines = ["#!/bin/bash\n", "set -e\n", "cd ../whiskitphysics/build\n\n"]
        shape_id = get_shape_id(shape_paths[0])
        lines += f"\n# {shape_id}\n"
        lines += gen_sweep_set(shape_id, args)
    else:
        filename = "all_sweeps.txt"
        for shape_path in shape_paths:
            shape_id = get_shape_id(shape_path)
            lines += gen_sweep_set(shape_id, args, subfolder="train")

    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"wrote {len(lines)} lines to {filename}")
