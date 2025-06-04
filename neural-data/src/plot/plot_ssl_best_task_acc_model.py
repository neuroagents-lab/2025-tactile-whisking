import matplotlib.pyplot as plt
from utils.plotting import COLORS, FONTSIZE


def plot_3bar_augs(tact_aug_score, rdm_score, xlabel="", ylabel="", save_path=None):
    methods = ['image aug.', 'random init.', 'tactile aug.']
    scores = [0.0, rdm_score, tact_aug_score]
    bar_colors = ['white', "#aaa", COLORS["orange"]]
    fig, ax = plt.subplots(figsize=(2.3, 2.5))

    # Custom closer x-positions
    x_pos = list(range(len(scores)))
    bars = ax.bar(x_pos, scores, color=bar_colors, width=0.8)

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)

    # Remove x-axis ticks
    ax.set_xticks([])
    ax.set_xlabel(xlabel)

    # Add vertical text fully inside the bars
    for bar, label in zip(bars, methods):
        bar_x = bar.get_x() + bar.get_width() / 2
        bar_height = bar.get_height()

        ax.text(bar_x, 0.01, label,
                ha='center', va='bottom',
                rotation=90, fontsize=FONTSIZE + 2,
                color='black')

    plt.savefig(save_path)
    print(f"saved to {save_path}")


if __name__ == "__main__":
    # best task acc model:
    # tactile1000hz_enc_Zhuang_att_GPT
    plot_3bar_augs(
        tact_aug_score=0.6953131556510925, # Top-5 Test categorization acc
        rdm_score=0.05198289453983307, # for tactile1000hz_enc_Zhuang_att_GPT_rdm_init
        xlabel="Zhuang+GPT",
        ylabel="Top-5 Test Cat. Acc.",
        save_path="out/figures/3bar_task_acc.png"
    )

    # best neural fit model:
    # Zhuang_inter_ln_simclr_lr1e_1
    plot_3bar_augs(
        tact_aug_score=0.9558947492895166,
        rdm_score= 0.6584261139444939,
        xlabel="Inter+None/SimCLR",
        ylabel="RSA Pearson's r",
        save_path="out/figures/3bar_neural_fit.png"
    )