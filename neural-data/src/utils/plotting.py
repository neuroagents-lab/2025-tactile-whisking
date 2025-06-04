import re
import os
import json
import colorsys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex

FONTSIZE = 11
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['axes.titlesize'] = FONTSIZE + 2
mpl.rcParams['axes.labelsize'] = FONTSIZE + 2
mpl.rcParams['xtick.labelsize'] = FONTSIZE
mpl.rcParams['ytick.labelsize'] = FONTSIZE
mpl.rcParams['legend.fontsize'] = FONTSIZE - 1
mpl.rcParams['legend.title_fontsize'] = FONTSIZE - 1
# for pdf export
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

COLORS = dict(
    red="#f44f55",
    blue="#3378d8",
    green="#24c54c",
    orange="#ff8C42",
    purple="#996eff",
    cyan="#2fcdd5",
    yellow="#ffc233",
    slab="#395e66",
    steel="#6fb8c8",
    magenta="#b449c4",
    pink="#ed73a2",
    lime="#81c226",
    gray="#777285",
)
MARKERS = ["o", "^", "s", "h"]
DEFAULT_MARKER = "x"

ENCODERS = ["resnet", "ugrnn", "inter", "gru", "lstm", "zhuang"]
ATTENDERS = ["gpt", "mamba"]
SELF_SUPERVISED = ["simclr", "simsiam", "ae"]
LOSSES = ["supervised"] + SELF_SUPERVISED
DECODERS = ["mlp"]

ENCODER_LAYERS = ["relu", "pool", "fc", "conv", "layer"]
ATTENDER_LAYERS = ["att", "mamba", "gpt"]
DECODER_LAYERS = ["dec"]


def to_filename(n):
    return re.sub(r"\.|\(|\)|\>|\:|\,", "", re.sub(r"\s+", "_", n.lower()))


def make_color_dict(values, colors=list(COLORS.values())):
    """
    Create a dictionary mapping each value in `values` to a color from `colors`.
    If there are more values than colors, it will cycle through the colors.
    """
    return {value: colors[i % len(colors)] for i, value in enumerate(values)}


def darken(color, factor=0.7):
    """Darken a color via HLS by reducing its lightness (0 < factor < 1)."""
    r, g, b = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l *= factor
    r_d, g_d, b_d = colorsys.hls_to_rgb(h, l, s)
    return to_hex((r_d, g_d, b_d))


def choose_in_string(s, arr, default="?", case_sensitive=False):
    """Select the first string in `arr` that is in `s`, or return `default` if none are found."""
    if not case_sensitive:
        s = s.lower()
    for i in arr:
        if i in s:
            return i
    return default


def any_in_string(s, arr, case_sensitive=False):
    """Check if any of the strings in `arr` is in `s`."""
    if not case_sensitive:
        s = s.lower()
    for i in arr:
        if i in s:
            return True
    return False


def plot_r_histogram(ax, data, title):
    """
    Draw a histogram in given plot axis for x in [-1, 1]
    """
    ax.hist(data, bins=np.linspace(-1, 1, 20))
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    return ax


def plot_linregress_result(results, title, path=".", debug=False):
    def debug_print(*args):
        if debug:
            print(*args)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 6))
    fig.suptitle(title)

    keys = ["r_xx", "r_yy", "r_xy", "r_xy_n_sb"]

    for j, (split, split_result) in enumerate(results.items()):
        debug_print("---\n" + split)
        for key, result in split_result.items():
            # debug_print(key, result.shape)
            # debug_print("# NaNs:", np.isnan(result).sum(), "\t # infs:", np.isinf(result).sum())
            if result.ndim > 2:
                # average over trials
                result = np.mean(result, axis=0)

            # flattened = result[~np.isnan(result) & ~np.isinf(result)]
            flattened = result.flatten()

            key_i = keys.index(key) if key in keys else -1
            if key_i >= 0:
                plot_r_histogram(axes[j][key_i], flattened, f"{key} {split}")

    title = title.replace(" ", "_")
    fig.savefig(f"{path}/{title}.png")
    print(f"Saved figure to {path}/{title}.png")
    plt.clf()


def get_layer_order(layer_name):
    str_num = "".join(re.findall(r"\d", layer_name)) or "-1"
    num = int(str_num)
    if "relu" in layer_name:
        return 1 + num
    if "maxpool" in layer_name:
        return 2 + num
    if "layer" in layer_name:
        return 3 + num
    if "avgpool" in layer_name:
        return 50 + num
    if "fc" in layer_name:
        return 60 + num
    if "att" in layer_name:
        num += 100
        if "ln" in layer_name:
            return 10 + num
        if "lm" in layer_name:
            return 11 + num
        if "norm" in layer_name:
            return 12 + num
    if "dec" in layer_name:
        return 1000 + num
    return num


def get_model_shortname(filename, no_loss=False):
    filename = filename.replace("baku", "resnet_gpt")
    if no_loss:
        filename = re.sub(r"(_simclr)|(_simsiam)|(_AE)|(_lr1e_.)", "", filename)
    return re.sub(r"(tactile1000hz_)|(_rot_tflip110)|(_lbs)|(\.npz)", "", os.path.basename(filename))

def get_enc_att_loss(filename):
    enc = choose_in_string(filename, ENCODERS)
    att = choose_in_string(filename, ATTENDERS, default="none")
    loss = choose_in_string(filename, LOSSES, default="supervised")
    return enc, att, loss

def get_colors(color_names):
    return [COLORS[color_name] for color_name in color_names if color_name in COLORS]

def get_color_for_model_dict(by="encoder"):
    if by == "encoder":
        return make_color_dict(ENCODERS)
    elif by == "attender":
        return make_color_dict(ATTENDERS)
    elif by == "loss":
        return make_color_dict(LOSSES, colors=get_colors(["steel", "magenta", "pink", "lime"]))
    else:
        raise ValueError(f"Invalid value for get_color_for_model_dict: expected 'enc' or 'loss', got '{by}'.")


def get_color_for_model(name, by="encoder", default=COLORS["gray"]):
    """
    Get a color for a model based on its name.
    by can be 'enc', 'att', or 'loss'.
    """
    enc, att, loss = get_enc_att_loss(name)

    color_dict = by
    if type(by) is str:
        color_dict = get_color_for_model_dict(by)
    assert type(color_dict) == dict

    for key, color in color_dict.items():
        if key in enc or key in att or key in loss:
            return color
    
    return default


def get_color_for_layer(name):
    if any_in_string(name, ENCODER_LAYERS):
        return COLORS["red"]
    if any_in_string(name, ATTENDER_LAYERS):
        return COLORS["green"]
    if any_in_string(name, DECODER_LAYERS):
        return COLORS["blue"]
    return "gray"
