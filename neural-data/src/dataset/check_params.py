import json
import re
import os

def get_model_shortname(filename, no_loss=False):
    filename = filename.replace("baku", "resnet_gpt")
    if no_loss:
        filename = re.sub(r"(_simclr)|(_simsiam)|(_AE)|(_lr1e_.)", "", filename)
    return re.sub(r"(tactile1000hz_)|(_rot_tflip110)|(_lbs)|(\.npz)", "", os.path.basename(filename))

# Load json
with open("data/model_total_params.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Map shortnames
shortname_to_value = {}
for key, value in data.items():
    shortname = get_model_shortname(key, no_loss=True)
    if shortname in shortname_to_value:
        if shortname_to_value[shortname] != value:
            print(f"Mismatch for {shortname}: {shortname_to_value[shortname]} vs {value}")
    else:
        shortname_to_value[shortname] = value

with open("data/model_total_params_mapped.json", "w", encoding="utf-8") as f:
    json.dump(shortname_to_value, f, indent=4)