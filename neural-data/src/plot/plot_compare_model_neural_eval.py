import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from glob import glob
from collections import OrderedDict

from dataset.rodgers_data import load_linregress_results
from utils.linregress_utils import get_rxy_ryy_filtered
from utils.plotting import *
from concurrent.futures import ProcessPoolExecutor


RYY_CUTOFF = 0.5
MAX_ANIMAL_SCORE = 1.336684120103193
MEAN_ANIMAL_SCORE = 0.17541514365056302
SEM_ANIMAL_SCORE = 0.16078577750944922

marker_dict = make_color_dict(ATTENDERS, MARKERS)
def get_marker_label(model_name):
    for item in marker_dict.keys():
        if item.lower() in model_name.lower():
            return marker_dict.get(item), item
    return DEFAULT_MARKER, "none"

def get_model_rdm_name(n, for_model_task_score=False):
    return get_model_shortname(n, no_loss=True) + ("_rdm_init" if for_model_task_score else "")

def process_file(filepath):
    try:
        results = load_linregress_results(filepath)
    except Exception as error:
        print("could not open file", filepath)
        print(error)
        return None
    
    rxysb, ryysb = get_rxy_ryy_filtered(results["test"], filter_ryy=True, filter_rxx=True, cutoff=RYY_CUTOFF)
    n_mean_rxysb = np.nanmean(rxysb, axis=(0, 1))
    # n_mean_ryysb = np.nanmean(ryysb, axis=(0, 1))
    median_rxy = np.nanmedian(n_mean_rxysb)
    # median_ryy = np.nanmedian(n_mean_ryysb)
    sem_rxy = np.nanstd(n_mean_rxysb) / np.sqrt(np.sum(~np.isnan(n_mean_rxysb)))
    # sem_ryy = np.nanstd(n_mean_ryysb) / np.sqrt(np.sum(~np.isnan(n_mean_ryysb)))

    #                       3 for model/layer/target     .npz
    filename = ("-".join(filepath.split("/")[-3:])).split(".np")[0]

    return get_model_shortname(filename), { 
        # "n_mean_rxysb": n_mean_rxysb,
        # "n_mean_ryysb": n_mean_ryysb,
        "median_rxy": median_rxy,
        # "median_ryy": median_ryy,
        "sem_rxy": sem_rxy,
        # "sem_ryy": sem_ryy
    }


def get_median_sem_from_results(results_dir, cpus, sem_key):
    if cpus == 1:
        files = glob(results_dir + "/*/*/*.npz")
        output = [process_file(file) for file in files]
        results = OrderedDict(output)
    else:
        with ProcessPoolExecutor(max_workers=cpus) as executor:
            files = glob(results_dir + "/*/*/*.npz")
            output = [item for item in executor.map(process_file, files) if item]
        results = OrderedDict(output)
    
    # remove any layers with name dec_mlp* that are in self-supervised models
    results = {k: v for k, v in results.items() if (
        not any_in_string(k, SELF_SUPERVISED) or
        any_in_string(k, SELF_SUPERVISED) and not any_in_string(k, DECODERS))}
            
    # get sem_animal
    if sem_key == "sem_animal":
        group_vals = {}
        for name, res in results.items():
            model_name, layer_name, *_ = name.split("-")
            group_vals.setdefault((model_name, layer_name), []).append(res["median_rxy"])
        for name, res in results.items():
            model_name, layer_name, *_ = name.split("-")
            vals = np.array(group_vals[(model_name, layer_name)])
            res["sem_animal"] = (np.nanstd(vals) /
                                np.sqrt(np.sum(~np.isnan(vals))))

    return results


def make_results_dict(results):
    results_dict = {}
    for name, result in results.items():
        name_split = name.split("-")
        model_name = name_split[0]
        layer_name = name_split[1]

        if model_name not in results_dict:
            results_dict[model_name] = {}
        if layer_name not in results_dict[model_name]:
            results_dict[model_name][layer_name] = []
        results_dict[model_name][layer_name].append(result)
    
    return results_dict


def get_max_per_model(results):
    results_dict = make_results_dict(results)

    model_results = {}
    for model_name, layers in results_dict.items():
        model_results[model_name] = max(
            # (res for layer in layers.values() for res in layer if res["sem_rxy"] <= 0.1),
            (res for layer in layers.values() for res in layer),
            key=lambda x: x["median_rxy"]
        )
    
    return model_results


def get_max_per_layer_per_model(results):
    results_dict = make_results_dict(results)
    all_layer_results = {}
    for model_name, layers in results_dict.items():
        layer_results = {}
        for layer_name, layer_data in layers.items():
            layer_results[layer_name] = max(
                [res for res in layer_data],
                key=lambda x: x["median_rxy"]
            )
        all_layer_results[model_name] = layer_results
    return all_layer_results


def plot_layer_neuralfits(title, results, figsize, out, sem, plt_func=None):
    names = sorted(results.keys(), key=get_layer_order)
    
    colors = [get_color_for_layer(name) for name in names]
    medians = [results[name]["median_rxy"] for name in names]
    sems = [results[name][sem] for name in names]

    plt.figure(figsize=figsize)

    if np.all((np.abs(sems) < 0.0001) | np.isnan(sems)):
        for i, (median, color) in enumerate(zip(medians, colors)):
            plt.bar(i, median, color=color)
    else:
        for i, (median, sem, color) in enumerate(zip(medians, sems, colors)):
            plt.bar(i, median, color=color)
            plt.errorbar(i, median, yerr=sem, capsize=3, color=darken(color, 0.5))

    labels = [name.replace("dec_", "").replace("att_", "") for name in names]
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=60, ha="right")
    plt.yticks([0, 0.5, 1.0])
    ax = plt.gca()
    for label, color in zip(ax.get_xticklabels(), colors):
        label.set_color(darken(color, 0.5))

    if plt_func:
        plt_func()
    plt.xticks(rotation=60, ha="right")
    # plt.xlabel("Model Layer")
    plt.ylabel("RSA Pearson's r")
    # plt.ylabel("Neural Fit (RSA Pearson's r)")
    # ax = plt.gca()
    # ax.set_ylabel("Neural Fit (RSA Pearson's r)")
    # ax.yaxis.set_label_coords(-0.07, 0.2)

    plt.title(title)
    plt.tight_layout()

    filename = out + "/" + title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300)
    print("figure saved to", filename)

def plot_task_score_models_double(title, model_scores, rdm_model_scores, model_names, figsize, out, color_by, plt_func=None):
    groups = {}
    group_xpos = {}
    group_colors = {}

    def get_group(name):
        if color_by == "encoder":
            _, _, group = get_enc_att_loss(name)
        else:  # color_by == "loss"
            group, _, _ = get_enc_att_loss(name)
        return group

    for model_name in model_names:
        group = get_group(model_name)
        if group not in groups:
            groups[group] = []
            group_colors[group] = get_color_for_model(group, by="encoder" if color_by == "loss" else "loss")
        groups[group].append(model_name)

    plt.figure(figsize=figsize)
    bar_w = 1
    spacing = 3
    group_spacing = 2
    # xs = np.arange(len(model_names)) * spacing
    xs = []
    i = 0
    colors = []
    xtick_positions = []
    xtick_labels = []

    for g, (group, model_names) in enumerate(groups.items()):
        group_xpos[group] = []
        for name in model_names:
            x = (g * group_spacing) + (i * spacing)
            xs.append(x)
            group_xpos[group].append(x)
            color = get_color_for_model(name, by=color_by)
            colors.append(color)
            dark_color = darken(color, 0.4)
            marker, marker_label = get_marker_label(name)
            score = model_scores[name]

            plt.bar(x + bar_w / 2, score, bar_w, color=color)
            plt.errorbar(x + bar_w / 2, score, yerr=0, fmt=marker, capsize=2, color=dark_color)

            enc, att, _ = get_enc_att_loss(name)
            rdm_key = f"{enc}+{att}"
            if rdm_key in rdm_model_scores:
                score_rdm = rdm_model_scores[rdm_key]
                plt.bar(x - bar_w / 2, score_rdm, bar_w, color=color, alpha=0.35)
                plt.errorbar(x - bar_w / 2, score_rdm, yerr=0, fmt=marker, capsize=2, color=dark_color, alpha=0.35)

            i += 1

        if g < len(groups) - 1:
            plt.axvline(x=xs[-1] + group_spacing + 0.5, color=COLORS["gray"], linestyle="--", alpha=0.5)

    plt.xlim(xs[0] - spacing / 2, xs[-1] + spacing / 2)

    xtick_positions = []
    xtick_labels = []

    for group, xpos_list in group_xpos.items():
        min_x = min(xpos_list)
        max_x = max(xpos_list)
        center_x = (min_x + max_x) / 2
        xtick_positions.append(center_x)
        xtick_labels.append(group)

        # Horizontal line under group
        plt.hlines(
            y=0,
            xmin=min_x - spacing * 0.3,
            xmax=max_x + spacing * 0.3,
            color=group_colors[group],
            linewidth=10,
            alpha=0.6
        )

    plt.xticks(xtick_positions, xtick_labels, fontsize=FONTSIZE+2)
    for ticklabel in plt.gca().get_xticklabels():
        color = darken(group_colors[ticklabel.get_text()], 0.5)
        ticklabel.set_color(color)

    plt.xlabel("Model")
    plt.ylabel("Top-5 Test Cat. Acc.")
    ly = 1.0
    if color_by == "encoder":
        plt_add_legend(by="encoder", loc='upper left', bbox_to_anchor=(0.1, ly), ncol=2, columnspacing=0.5, handletextpad=0.4)
        plt_add_legend(by="attender", loc='upper left', bbox_to_anchor=(0, ly))
    else: # color_by == "loss"
        plt_add_legend(by="loss", loc='upper left', bbox_to_anchor=(0, ly))
    # plt.title(title)
    ax = plt.gca()
    ax.set_title(title, y=0.85, bbox=dict(facecolor='white', edgecolor='none'))
    if plt_func:
        plt_func()

    filename = out + "/" + to_filename(title) + ".png"
    plt.savefig(filename)
    print("figure saved to", filename)


def plot_median_sem_models_double(title, results, results_rand, figsize, out, sem, color_by, plt_func=None, sortby=None):
    if sortby:
        names = sorted(results.keys(), key=sortby)
    else:
        names = results.keys()
    
    colors = [get_color_for_model(name, by=color_by) for name in names]

    plt.figure(figsize=figsize)

    bar_w = 1
    spacing = 3
    xs = np.arange(len(names) + 1) * spacing

    animal_color = "black"
    plt.bar(0, MEAN_ANIMAL_SCORE, bar_w*2, color=animal_color)
    plt.errorbar(0, MEAN_ANIMAL_SCORE, yerr=SEM_ANIMAL_SCORE, capsize=2, color=COLORS["slab"])

    for i, name in enumerate(names):
        x = xs[i+1]
        color = colors[i]
        dark_color = darken(color, 0.4)
        marker, marker_label = get_marker_label(name)
        median = results[name]["median_rxy"]
        err = results[name][sem]

        plt.bar(x + bar_w/2, median, bar_w, color=color)
        plt.errorbar(x + bar_w/2, median, yerr=err, fmt=marker, capsize=2, color=dark_color)

        rdm_name = get_model_rdm_name(name)
        if rdm_name in results_rand:
            median_rand = results_rand[rdm_name]["median_rxy"]
            err_rand = results_rand[rdm_name][sem]

            plt.bar(x - bar_w/2, median_rand, bar_w, color=color, alpha=0.35)
            plt.errorbar(x - bar_w/2, median_rand, yerr=err_rand, fmt=marker, capsize=2, color=dark_color, alpha=0.35)

    plt.xlim(xs[0] - spacing/2, xs[-1] + spacing/2)

    def get_label(name):
        if name == "model_input":
            return "model input"
        else:
            if color_by == "encoder":
                _, _, label = get_enc_att_loss(name)
            else:
                label, _, _ = get_enc_att_loss(name)
            return label

    plt.xticks([])

    plt.text(
        bar_w/8, 0.01,
        # "animal-to-animal",
        "a2a",
        rotation=90,
        ha='center',
        va='bottom',
        color="white"
    )
    for i, name in enumerate(names):
        x = xs[i+1]
        label_text = get_label(name)
        plt.text(
            x + bar_w/8-0.2, 0.02,
            label_text,
            rotation=90,
            ha='center',
            va='bottom',
            color=darken(colors[i], 0.1),
            fontsize=FONTSIZE+1 if len(label_text) < 10 else FONTSIZE - 1
        )

    if plt_func:
        plt_func()
    plt.title(title)

    filename = out + "/" + to_filename(title) + ".png"
    plt.savefig(filename, pad_inches=0.35)
    print("figure saved to", filename)

def plt_add_legend(by, ssl_only=False, **kwargs):
    handles = []
    if by == "encoder":
        for enc in ENCODERS:
            color = get_color_for_model(enc, by="encoder", default=COLORS["gray"])
            handles.append(Line2D([0], [0], label=enc, marker='o', color=color, linestyle='None'))

    if by == "attender":
        for att in ATTENDERS + ["none"]:
            marker, marker_label = get_marker_label(att)
            handles.append(Line2D([0], [0], label=marker_label, marker=marker, color='black', linestyle='None'))

    if by == "loss":
        for loss in LOSSES if not ssl_only else SELF_SUPERVISED:
            color = get_color_for_model(loss, by="loss", default=COLORS["gray"])
            handles.append(Line2D([0], [0], label=loss, marker='o', color=color, linestyle='None'))

    legend = plt.legend(handles=handles, title=by.title(), borderpad=0.3, **kwargs)
    plt.gca().add_artist(legend)


def plot_model_score_vs_neural_fit(
    model_scores,
    per_model_fit,
    title,
    out,
    figsize=(5, 5),
    sem="sem_rxy",
    color_by="encoder",
    nochkpt=False,
    fit_line=False,
    xlabel="Top-5 Test Categorization Accuracy",
    ylabel="Neural Fit (RSA Pearson's r)",

):
    color_by_label = get_color_for_model_dict(by=color_by)
    label_by_color = {v: k for k, v in color_by_label.items()}

    plt.figure(figsize=figsize)

    x = []
    y = []
    for model_name in per_model_fit.keys():
        task_score = model_scores.get(get_model_rdm_name(model_name, for_model_task_score=True) if nochkpt else model_name, np.nan)
        neural_score = per_model_fit[model_name]["median_rxy"]
        sem_val = per_model_fit[model_name].get(sem, np.nan)
        color = get_color_for_model(model_name, by=color_by_label, default="gray")
        marker, marker_label = get_marker_label(model_name)
        if np.isfinite(task_score) and np.isfinite(neural_score):
            x.append(task_score)
            y.append(neural_score)

        _, caplines, barlinecols = plt.errorbar(
            task_score, neural_score, yerr=sem_val,
            fmt=marker, capsize=3, color=color)
        for element in barlinecols + caplines:
            element.set_alpha(0.4)

    if fit_line:
        x = np.array(x)
        y = np.array(y)
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x + b, color=COLORS['yellow'], linestyle='-')

        r = np.corrcoef(x, y)[0, 1]

        mid_x = 0.5 * (x.min() + x.max())
        mid_y = m * mid_x + b

        plt.text(
            mid_x, mid_y,
            f"r = {r:.2f}",
            # color=darken(COLORS['yellow'], 0.4),
            fontsize=FONTSIZE+3,
            ha='center',
            va='bottom',
            rotation=15,
            rotation_mode='anchor'
        )

    # if "supervised" in title.lower():
    #     plt_add_legend(by=color_by, loc='upper left', bbox_to_anchor=(0.0, 1.0), ncols=2, columnspacing=0.5, handletextpad=0.4)
    #     plt_add_legend(by="attender", loc='upper left', bbox_to_anchor=(0.4, 1.0))
    # else:
    #     plt_add_legend(by=color_by, ssl_only=True, loc='lower right', bbox_to_anchor=(1.0, 0.0))
    plt_add_legend(by="encoder", loc='lower right', bbox_to_anchor=(1.02, 0.0), ncols=2, columnspacing=0.5, handletextpad=0.4)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if "." not in out:
        out += "/model_performance_vs_neural_fit.png"
    plt.savefig(out, dpi=300)
    print("figure saved to", out)


def get_layer_scores_across_models(results):
    layer_scores = {}
    for model_layers in make_results_dict(results).values():
        for layer_name, layer_data in model_layers.items():
            for res in layer_data:
                layer_scores.setdefault(layer_name, []).append(res["median_rxy"])
    return layer_scores

def plot_layer_compare(layer_scores, out, figsize):
    layer_names, data = [], []
    for layer in sorted(layer_scores.keys(), key=get_layer_order):
        vals = np.asarray(layer_scores[layer])
        vals = vals[~np.isnan(vals)]
        if vals.size:
            layer_names.append(layer)
            data.append(vals)

    medians_per_layer = [np.median(vals, axis=0) for vals in data]

    plt.figure(figsize=figsize)
    parts = plt.violinplot(medians_per_layer, showmeans=False, showmedians=True, widths=0.8)

    # for i, layer in enumerate(layer_names):
    #     pc = parts['bodies'][i]
    #     pc.set_facecolor(get_color_for_layer(layer))
    #     pc.set_edgecolor('black')
    #     pc.set_alpha(0.7)

    plt.xticks(ticks=np.arange(1, len(layer_names) + 1),
               labels=layer_names, rotation=60, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("Median Rxy per Model")
    plt.axhline(y=1.0, color="black", linestyle="--")
    plt.title("Median Neural Consistency per Layer (per Model)")
    plt.tight_layout()

    filename = out + "/layer_compare_violin.png"
    plt.savefig(filename, dpi=300)
    print("figure saved to", filename)


def plt_animal_score_lines():
    plt.axhline(y=1.0, color=COLORS["slab"], linestyle="--", label="1.0", alpha=0.5)
    # plt.axhline(y=MAX_ANIMAL_SCORE, color=COLORS["red"], linestyle="--", label="Max Animal Score")
    mean_color = COLORS["gray"]
    plt.axhline(y=MEAN_ANIMAL_SCORE, color=mean_color, linestyle="--", label="Mean Animal Score", alpha=0.6)
    plt.axhspan(MEAN_ANIMAL_SCORE - SEM_ANIMAL_SCORE, MEAN_ANIMAL_SCORE + SEM_ANIMAL_SCORE, color=mean_color, alpha=0.1)


# def get_results_dict(results_dir, cpus, sem_key, json_path=None):
def get_results_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    for k,v in results.items():
        # replace nulls with np.nan
        for ki, vi in v.items():
            if vi == None:
                results[k][ki] = np.nan
    return results


def plot_model_score_vs_params(model_names, model_scores, model_params, color_by, out, figsize=(7, 6)):
    plt.figure(figsize=figsize)

    for model_name in model_names:
        score = model_scores.get(model_name, np.nan)
        shortname = get_model_shortname(model_name.replace("_rdm_init", ""), no_loss=True)
        params = model_params.get(shortname, np.nan)
        color = get_color_for_model(model_name, by=color_by)
        marker, marker_label = get_marker_label(model_name)
        plt.scatter(params, score, color=color, marker=marker, label=model_name, alpha=0.7)

    plt.xlabel("Total Parameters")
    plt.ylabel("Top-5 Test Categorization Acc.")
    plt.title("Model Parameters vs Task Performance")
    plt.yticks(np.arange(0, 0.8, 0.25))
    # plt.xscale("log")

    # plt_add_legend(by=color_by, loc='lower right', bbox_to_anchor=(1.0, 0.0))
    # plt_add_legend(by="attender", loc='lower center', bbox_to_anchor=(0.7, 0.0))
    plt_add_legend(by=color_by, loc='lower right', bbox_to_anchor=(1.02, 0.31))
    plt_add_legend(by="attender", loc='lower center', bbox_to_anchor=(0.5, 0.31))

    plt.tight_layout()
    if ".png" not in out:
        out += "/model_params_vs_task_perf.png"
    plt.savefig(out, dpi=300)
    print("figure saved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare linear regression results.")
    parser.add_argument("--out", type=str, default="out/model_eval", help="Path to dir save the output images/json.")
    parser.add_argument("--metric", type=str, default="rsa_pearsonr", help="metric used; used to find paths /<metric>, /<metric>_random")
    parser.add_argument("--results_dir", "-d", type=str, default="/data/group_data/neuroagents_lab/tactile/model_to_animal/", help="directory to load the data from")
    parser.add_argument("--json", type=str, help="json filepath from previous run to load the data from")
    parser.add_argument("--cpus", type=int, default=1, help="Number of CPUs to use for parallel processing.")
    parser.add_argument("--subtitle", type=str, default="", help="subtitle for plot")
    parser.add_argument("--figsize", type=float, nargs=2, default=None, help="plot figure size")
    parser.add_argument("--sem", type=str, default="sem_animal",
                    choices=["sem_rxy", "sem_animal"],
                    help="Which SEM estimate to plot (sem_rxy | sem_animal)")
    parser.add_argument("--mode", type=str, nargs="+",
                    default=["layer", "modelscatter", "modelparams", "modelneural", "modeltask", "layercompare"],
                    choices=["layer", "modelscatter", "modelparams", "modelneural", "modeltask", "layercompare"],
                    help="plot max per `model`, plots for each `layer`, "
                         "plot model performance vs neural fit, or compare layers")
    parser.add_argument("--cutoff", type=float, default=RYY_CUTOFF, help="ryy/rxy geq cutoff for filtering")
    args = parser.parse_args()

    RYY_CUTOFF = args.cutoff

    os.makedirs(args.out, exist_ok=True)

    with open("data/all_test_results.json", "r", encoding="utf-8") as f:
        model_scores = json.load(f)["test_acc_top5"]
    model_scores = {get_model_shortname(k, no_loss=False): v
                    for k, v in model_scores.items()}

    with open("data/model_total_params_mapped.json", "r", encoding="utf-8") as f:
        model_params = json.load(f)

    
    if args.json:
        results = get_results_from_json(args.json)
        results_rand = get_results_from_json(args.json.replace(".json", "_random.json"))
    else:
        results = get_median_sem_from_results(results_dir=args.results_dir+"/"+args.metric, cpus=args.cpus, sem_key=args.sem)
        results_rand = get_median_sem_from_results(results_dir=args.results_dir+"/"+args.metric+"_random", cpus=args.cpus, sem_key=args.sem)
        with open(args.out + "/results_summary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        with open(args.out + "/results_summary_random.json", "w", encoding="utf-8") as f:
            json.dump(results_rand, f, indent=4)

    per_model_results = get_max_per_model(results)
    per_model_results_rand = get_max_per_model(results_rand)
    per_model_results_rand = {get_model_shortname(k, no_loss=True): v for k, v in per_model_results_rand.items()}

    
    if "layer" in args.mode:
        per_layer_results = get_max_per_layer_per_model(results)
        def plt_func():
            plt_animal_score_lines()
            plt.xlabel("Layer")
            plt.ylabel("r_xy Score")
        # for model_name, layer_results in per_layer_results.items():
        #     plot_layer_neuralfits(
        #         f"Linregress per Layer in {model_name}",
        #         layer_results,
        #         figsize=args.figsize or [2 + len(layer_results) * 0.2, 4],
        #         out=args.out,
        #         sem=args.sem,
        #         color_func=get_color_for_layer,
        #         sortby=get_layer_order)

        # per_layer_results_rand = get_max_per_layer_per_model(results_rand)
        # for model_name, layer_results in per_layer_results_rand.items():
        #     plot_median_sem(
        #         f"Linregress per Layer in {model_name} (Randomly Initialized)",
        #         layer_results,
        #         figsize=args.figsize or [2 + len(layer_results) * 0.2, 4],
        #         out=args.out,
        #         sem=args.sem,
        #         color_func=get_color_for_layer,
        #         sortby=get_layer_order)

        layer_results = per_layer_results["enc_Zhuang_inter_ln_att_GPT"]
        plot_layer_neuralfits(
            f"Neural Fit per Layer in Inter+GPT+SL",
            layer_results,
            figsize=args.figsize or [2 + len(layer_results) * 0.2, 3],
            out=args.out,
            sem=args.sem)

        layer_results = per_layer_results["Zhuang_inter_ln_simclr_lr1e_1"]
        plot_layer_neuralfits(
            f"Neural Fit per Layer in Inter+SimCLR",
            layer_results,
            figsize=args.figsize or [2 + len(layer_results) * 0.2, 3],
            out=args.out,
            sem=args.sem)


    if "modelneural" in args.mode:
        
        model_input_result = np.load("data/model_input_rsa.npz", allow_pickle=True)["arr_0"].item()["test"]["r_xy_n_sb"]
        sem = np.nanstd(model_input_result) / np.sqrt(np.sum(~np.isnan(model_input_result)))
        per_model_results["model_input"] = {
            "median_rxy": np.nanmedian(model_input_result),
            "sem_rxy": sem,
            "sem_animal": sem
        }

        def make_plt_func(color_by):
            def plt_func():
                plt.axhline(y=1.0, color=COLORS["slab"], linestyle="--", label="1.0", alpha=0.5)
                plt.xlabel("Model")
                plt.ylabel("Neural Fit (RSA Pearson's r)")
                ly = 1.25
                if color_by == "encoder":
                    plt_add_legend(by="attender", loc='upper left', bbox_to_anchor=(0, ly))
                    plt_add_legend(by=color_by, loc='upper left', bbox_to_anchor=(0.1, ly), ncol=2, columnspacing=0.5, handletextpad=0.4)
                else: # color_by == "loss"
                    plt_add_legend(by=color_by, loc='upper left', bbox_to_anchor=(0, ly), ncol=2, columnspacing=0.5, handletextpad=0.4)
            return plt_func

        kwargs = dict(
            results=per_model_results,
            results_rand=per_model_results_rand,
            figsize=args.figsize or [2 + len(per_model_results) * 0.2, 3],
            out=args.out,
            sem=args.sem,
            sortby=lambda k: float(per_model_results[k]["median_rxy"]) if not np.isnan(per_model_results[k]["median_rxy"]) else -np.inf
        )

        plot_median_sem_models_double(
            "Neural Fit per Model\n(Colored by Loss, Labeled by Encoder)" + (f": {args.subtitle}" if args.subtitle else ""),
            color_by="loss",
            plt_func=make_plt_func("loss"),
            **kwargs)
        plot_median_sem_models_double(
            "Neural Fit per Model\n(Colored by Encoder, Labeled by Loss)" + (f": {args.subtitle}" if args.subtitle else ""),
            color_by="encoder",
            plt_func=make_plt_func("encoder"),
            **kwargs)


    if "modeltask" in args.mode:
        per_model_results_copy = per_model_results.copy()
        per_model_results_copy.pop("model_input", None)
        model_names = per_model_results_copy.keys()
        model_names = sorted(model_names, key=lambda n: model_scores[n])
        rdm_model_scores = {}
        for n, score in model_scores.items():
            if "rdm_init" in n:
                enc, att, _ = get_enc_att_loss(n)
                key = f"{enc}+{att}"
                rdm_model_scores[key] = score

        plot_task_score_models_double(
            "Task Performance per Model\n(Colored by Encoder, Labeled by Loss)" + (f": {args.subtitle}" if args.subtitle else ""),
            model_scores,
            rdm_model_scores,
            model_names,
            figsize=args.figsize or [2 + len(per_model_results) * 0.2, 2],
            out=args.out,
            color_by="encoder",
        )
        # plot_task_score_models_double(
        #     "Task Performance per Model\n(Colored by Loss, Labeled by Encoder)" + (f": {args.subtitle}" if args.subtitle else ""),
        #     model_scores,
        #     model_names,
        #     figsize=args.figsize or [2 + len(per_model_results) * 0.2, 3],
        #     out=args.out,
        #     color_by="loss",
        # )


    if "modelscatter" in args.mode:
        with open("data/model_total_params.json", "r", encoding="utf-8") as f:
            model_params = json.load(f)
        model_params = {get_model_shortname(k, no_loss=False): v
                        for k, v in model_params.items()}
        
        per_model_fit_supervised = {}
        per_model_fit_ssl = {}
        for model_name, res in per_model_results.items():
            if any_in_string(model_name, SELF_SUPERVISED):
                per_model_fit_ssl[model_name] = res
            else:
                per_model_fit_supervised[model_name] = res
                 
        plot_model_score_vs_neural_fit(
            per_model_fit=per_model_fit_supervised,
            model_scores=model_scores,
            title="Task Performance vs Neural Fit: Supervised",
            color_by="encoder",
            fit_line=True,
            out=args.out+"/model_performance_vs_neural_fit_enc_supervised.png",
            sem=args.sem,
            figsize=args.figsize or [5.5, 3.5],
        )
        plot_model_score_vs_neural_fit(
            per_model_fit=per_model_fit_ssl,
            model_scores=model_scores,
            title="Task Performance vs Neural Fit: SSL",
            color_by="loss",
            fit_line=False,
            out=args.out+"/model_performance_vs_neural_fit_loss_ssl.png",
            sem=args.sem,
            figsize=args.figsize or [4, 3.5],
        )


    if "modelparams" in args.mode:
        per_model_results = get_max_per_model(results)
        per_model_results_rand = get_max_per_model(results_rand)

        plot_model_score_vs_params(
            model_names=[n for n in model_scores.keys() if "rdm_init" not in n],
            model_scores=model_scores,
            model_params=model_params,
            color_by="loss",
            out=args.out + "/model_params_vs_task_perf_byloss.png",
            figsize=args.figsize or [4.5, 3.5])


        with open("data/model_total_params.json", "r", encoding="utf-8") as f:
            model_params = json.load(f)
        model_params = {get_model_shortname(k, no_loss=False): v
                        for k, v in model_params.items()}
        
        per_model_fit_supervised = {}
        per_model_fit_ssl = {}
        for model_name, res in per_model_results.items():
            if any_in_string(model_name, SELF_SUPERVISED):
                per_model_fit_ssl[model_name] = res
            else:
                per_model_fit_supervised[model_name] = res

        plot_model_score_vs_neural_fit(
            per_model_fit=per_model_results,
            model_scores=model_params,
            title="Model Parameters vs Neural Fit",
            color_by="encoder",
            out=args.out+f"/model_params_vs_neural_fit_enc.png",
            xlabel="Total Parameters",
            sem=args.sem,
            figsize=args.figsize or [4.5, 3.5],
        )


    # if "layercompare" in args.mode:
    #     # Violin plot of layer score spread across models
    #     layer_scores = get_layer_scores_across_models(results)
    #     plot_layer_compare(
    #         layer_scores,
    #         out=args.out,
    #         figsize=args.figsize or [2 + len(layer_scores) * 0.4, 4],
    #     )