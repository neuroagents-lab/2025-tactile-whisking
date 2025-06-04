import argparse
import random
import glob
import os

# see whisker_calculations.ipynb for POSITION and ORIENTATION
DEFAULT_FLAGS = "--SAVE 1 --HZ 110 --SUBSTEPS 100 --OBJ_X 5 --OBJ_Y 20 --WHISKER_LINKS_PER_MM 1 --POSITION -6.934 -11.642 -2.598 --ORIENTATION 0 15.366 -16.198"

# these values are based on Zhuang2017, adjusted for our mouse (~1/2 size)
SWEEP_HEIGHTS = [-3, 0, 3]
SWEEP_ORIENTATIONS = [0, 90, 180, 270]
OBJ_INIT_ORIENTATION = [0, 89]
OBJ_SPEED_RANGE = [30, 60]
OBJ_SCALE_RANGE = [20, 60]

# paths
SHAPENET_DIR = os.path.abspath("../shapenet/ShapeNetCore")
OUT_DIR = os.path.abspath("../output/sweeps")
WHISKIT = os.path.abspath("../whiskitphysics/build/whiskit")

def gen_sweep_set(shape, args, is_train=True):
    cmds = []
    speed = random.randint(*OBJ_SPEED_RANGE)
    scale = random.randint(*OBJ_SCALE_RANGE)
    ori = random.randint(*OBJ_INIT_ORIENTATION)
    out_dir = args.out + "/" + ("train" if is_train else "test")
    for height in SWEEP_HEIGHTS:
        for theta in SWEEP_ORIENTATIONS:
            cmds.append(
                f"{WHISKIT}{'_gui' if args.gui else ''} {DEFAULT_FLAGS}"
                + f' --dir_out "{out_dir}/{shape}/h{height}-to{ori}-t{theta}-v{speed}-s{scale}"'.replace('.', '_')
                + f' --OBJ_PATH "{SHAPENET_DIR}/{shape}/models/model_normalized.obj"'
                + f" --OBJ_SCALE {scale}"
                + f" --OBJ_SPEED {speed}"
                + f" --OBJ_Z {height}"
                + f" --OBJ_THETA {ori + theta}"
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

            for _ in range(24):
                lines += gen_sweep_set(shape_id, args, is_train=True)
            for _ in range(2):
                lines += gen_sweep_set(shape_id, args, is_train=False)

    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"wrote {len(lines)} lines to {filename}")
