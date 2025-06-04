import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import COLORS, FONTSIZE


def plot_combined_3bar_augs_with_labels(
    tact_aug_score_1, rdm_score_1, model_label_1,
    tact_aug_score_2, rdm_score_2, model_label_2,
    save_path=None
):
    methods = ['random init.', 'image aug.', 'tactile aug.']
    bar_colors = ['#aaa', 'white',  COLORS['orange']]

    scores_top = [rdm_score_1, 0.0, tact_aug_score_1]      # task acc
    scores_bottom = [rdm_score_2, 0.0, tact_aug_score_2]   # neural fit

    # Increase gap by shifting the bottom group further down
    d = 0.2
    gap = 0.5
    y_top = np.array([5-d, 4, 3+d]) + gap/2
    y_bottom = np.array([2-d, 1, 0+d]) - gap/2

    fig, ax = plt.subplots(figsize=(2.2, 4))

    ax.barh(y_top, scores_top, color=bar_colors, height=0.7)
    ax.barh(y_bottom, scores_bottom, color=bar_colors, height=0.7)

    for y, label in zip(y_top, methods):
        ax.text(0.01, y, label, va='center', ha='left',
                fontsize=FONTSIZE + 1, color='black')
    for y, label in zip(y_bottom, methods):
        ax.text(0.01, y, label, va='center', ha='left',
                fontsize=FONTSIZE + 1, color='black')

    ax.set_xlim(0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("RSA Pearson's r", fontsize=FONTSIZE + 1)

    ax_top = ax.secondary_xaxis('top')
    ax_top.set_xticks([0.0, 0.5, 1.0])
    ax_top.set_xlim(0, 1.0)
    ax_top.set_xlabel("Top-5 Test Cat. Acc.", fontsize=FONTSIZE + 1)

    ax.set_yticks([4.25, 0.75])
    ax.set_yticklabels([model_label_1, model_label_2], fontsize=FONTSIZE + 1, rotation=90, va='center')

    ax.axhline(2.5, linestyle='--', color='black', linewidth=1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"saved to {save_path}")
    plt.close()


# Example usage
if __name__ == "__main__":
    plot_combined_3bar_augs_with_labels(
        tact_aug_score_1=0.6953131556510925,
        rdm_score_1=0.05198289453983307,
        model_label_1="Zhuang+GPT/Sup.", # tactile1000hz_enc_Zhuang_att_GPT
        tact_aug_score_2=0.9558947492895166,
        rdm_score_2=0.6584261139444939,
        model_label_2="Inter/SimCLR", # Zhuang_inter_ln_simclr_lr1e_1
        save_path="out/model_eval/6bar_augmentations.png"
    )