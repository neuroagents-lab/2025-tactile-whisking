import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize whisker forces from simulation")
    parser.add_argument(
        "-p",
        "--path",
        default="../output/obj_sweep_test",
        type=str,
        help="path to simulation data (should contain dynamics/)",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="save animation as gif",
    )


    args = parser.parse_args()

    whisker_ids = pd.read_csv(args.path + "/whisker_ID.csv", header=None).values.flatten()
    fx = pd.read_csv(args.path + "/dynamics/Fx.csv", header=None)
    fy = pd.read_csv(args.path + "/dynamics/Fy.csv", header=None)
    fz = pd.read_csv(args.path + "/dynamics/Fz.csv", header=None)
    force_magnitude = np.sqrt(fx.pow(2) + fy.pow(2) + fz.pow(2))
    force_min = force_magnitude.min().min()
    force_max = force_magnitude.max().max()
    # print('forces', force_magnitude.shape)
    # print(force_magnitude.head())

    num_cells = fx.shape[1]
    frames = fx.shape[0]
    fps = 110

    grid_layout = [
        (0, 4), (1, 4), (2, 4), (3, 4), (4, 4),
        (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 4), (6, 3),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 3), (6, 2),
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 2)
    ]
    fig, ax = plt.subplots()
    scat = ax.scatter(
        [], [], c=[],
        cmap=cm.viridis,
        vmin=force_min,
        vmax=force_max,
        s=1000
    )

    def init():
        ax.set_xlim(-0.5, 7 - 0.5)
        ax.set_ylim(-0.5, 5 - 0.5)
        ax.axis('off')
        for i, whisker_id in enumerate(whisker_ids):
            x, y = grid_layout[i]
            ax.annotate(whisker_id, (x, y), textcoords="offset points", xytext=(0, -25), ha='center')
        return (scat,)

    def update(frame):
        magnitudes = force_magnitude.iloc[frame].values
        # print("magnitudes", len(magnitudes), magnitudes)
        x = [pos[0] for pos in grid_layout]
        y = [pos[1] for pos in grid_layout]
        scat.set_offsets(np.c_[x, y])
        scat.set_array(magnitudes)
        return (scat,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=1/fps * 1000,
        blit=True,
        repeat=True,
    )
    plt.colorbar(scat, ax=ax, label="|F| (µNm)")
    if args.save:
        ani.save('whisker_forces.gif', writer='pillow', fps=fps, dpi=100)
        print("saved animation to whisker_forces.gif")
    else:
        plt.show()
