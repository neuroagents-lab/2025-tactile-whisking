import os
import numpy as np
import argparse
from einops import rearrange
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from brainmodel_utils.metrics.utils import rdm
from concurrent.futures import ProcessPoolExecutor
from utils.model_eval import load_activation_data
from utils.plotting import get_model_shortname, get_enc_att_loss

def plot_rdm(data, title, outpath):
    fig = plt.figure()
    fig.set_size_inches(2.6, 2.6)
    ax = fig.add_axes([0, 0, 1, 1])

    im = ax.imshow(data, cmap='viridis', interpolation='nearest')
    ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # cbar_ax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.04,
    #                         ax.get_position().width, 0.03])
                            
    # fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    # xpad = 0.2
    xpad = 0.1
    cbar_ax = fig.add_axes([ax.get_position().x0 + xpad/2, ax.get_position().y0 - 0.04,
                            ax.get_position().width - xpad, 0.03])

    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    # cbar.set_ticks([0.0, 0.004, 0.008, 0.012])
    # cbar.set_ticks([0.0, 0.4, 0.8, 1.2])
    plt.savefig(outpath)
    # outpath = outpath.replace(".png", ".pdf")
    # plt.savefig(outpath, transparent=True)
    print(f"saved to {outpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linear regression analysis on Rodgers data.")
    parser.add_argument("--outdir", "-o", type=str, default="./out/rdm", help="Path to save the results npz files.")
    parser.add_argument("--cpus", type=int, default=1, help="number of parallel jobs to run")
    parser.add_argument("--tactile_npz", type=str, default="./out/rdm/tactile_val.npz", help="load tactile data from this npz file")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # # animal data
    # data_per_animal = load_rodgers_data(path="./data/rodgers6_data.npz")
    # animal_data = concatenate_sessions(list(data_per_animal.values()))
    # animal_data = np.nanmean(animal_data, axis=0)  # average over trials

    # rdm_data = rdm(animal_data)
    # plot_rdm(rdm_data, "RDM of Animal Data", args.outdir + "/animal_rdm.png")


    # stimuli data
    stimuli_datapath = "/data/group_data/neuroagents_lab/tactile/neural_data/rodgers6_simulated.npz"
    input_data = np.load(stimuli_datapath)['data']
    input_data = input_data.mean(axis=1)  # shape: (6, 30, 5, 7)
    input_data = input_data.reshape(6, -1)  # shape: (6, 30*5*7)
    plot_rdm(rdm(input_data), "RDM of Simulated Stimuli", args.outdir + "/rdm_stimuli.png")



    # tactile dataset
    if args.tactile_npz is not None:
        data_by_label = np.load(args.tactile_npz)["data"]
    else:
        n_times = 22
        H, W = 5, 7

        def transform(data):
            return rearrange(data, "(ntime step) (H W) f -> ntime (step f) H W", ntime=n_times, H=H, W=W)

        def load_and_transform(path):
            np_data = np.load(path)
            sample = np_data["data"]  # shape (110, 35, 6)
            sample = transform(sample)      # (22, 30, 5, 7)
            label = int(np_data["label"].item())
            sample = np.nanmean(sample, axis=(0))
            return label, sample.reshape(-1)

        data_dir = "/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed/validate"
        filepaths = glob(os.path.join(data_dir, "*.npz"))

        with ProcessPoolExecutor(max_workers=args.cpus) as executor:
            all_samples = list(tqdm(executor.map(load_and_transform, filepaths), total=len(filepaths)))

        data_dict = {}
        for label, sample in all_samples:
            if label not in data_dict:
                data_dict[label] = []
            data_dict[label].append(np.squeeze(sample))

        data_by_label = np.array([
            np.mean(np.stack(samples, axis=0), axis=0)
            for samples in data_dict.values()
        ])

        np.savez(args.outdir + "/tactile_val.npz", data=data_by_label)

    rdm_data = rdm(data_by_label)
    plot_rdm(rdm_data, "RDM of Tactile Val. Dataset", args.outdir + "/rdm_tactile_val.png")


    # model activations
    base_path = "/data/group_data/neuroagents_lab/tactile/model_to_animal/activations/2025-05-12"

    plot_rdm(rdm(load_activation_data(f"{base_path}/tactile1000hz_Zhuang_inter_ln_simclr_rot_tflip110_lr1e_1/activations.npz")["fc8"]),
                "Inter+SimCLR: Layer FC-8",
                args.outdir + f"/rdm_inter_simclr_fc8.png")

    plot_rdm(rdm(load_activation_data(f"{base_path}/tactile1000hz_enc_Zhuang_inter_ln_att_GPT/activations.npz")["att_gpt_lm_head"]),
                "Inter+GPT+SL: GPT LM Head",
                args.outdir + f"/rdm_inter_gpt_sup_gptlm.png")
