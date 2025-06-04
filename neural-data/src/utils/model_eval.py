import numpy as np
import re


def load_activation_data(activation_data_path):
    data = np.load(activation_data_path, allow_pickle=True)

    enc_data = data['enc'].item()
    att_data = data['att'].item()
    dec_data = data['dec'].item()

    all_layers = {}
    for data in [enc_data, att_data, dec_data]:
        layers = get_layers_from_activation_data(data)
        all_layers.update(layers)
    return all_layers


def get_layers_from_activation_data(model_data):
    model_data_keys = list(model_data.keys())
    is_tnn = np.all([isinstance(k, int) for k in model_data_keys])
    layers = {}
    if is_tnn:
        for t in sorted(model_data.keys()):
            layers_t = model_data[t]
            assert isinstance(layers_t, dict), f"expected dict for tnn, got {type(layers_t)}"

            for layer_name, layer_data in layers_t.items():
                l = int(re.search(r'\d+', layer_name).group(0))
                if t >= l - 1:
                    if layer_name not in layers:
                        layers[layer_name] = [layer_data]
                    else:
                        layers[layer_name].append(layer_data)
        layers = {k: format_activations_tensor(np.array(v)) for k, v in layers.items()}
    else:
        for key, value in model_data.items():
            if isinstance(value, np.ndarray):
                layers[key] = format_activations_tensor(value)
            else:
                print("layer", key, "has unexpected type:", type(value))

    return layers

def format_activations_tensor(tensor):
    if tensor.ndim >= 1 and tensor.shape[0] == 132:
        # 132 -> 6 stimuli * 22 timesteps
        tensor = tensor.reshape(6, 22, -1)
    elif tensor.shape[0] != 6 and tensor.shape[1] == 6: # tnn
        # tensor = np.nanmean(tensor, axis=2)
        tensor = tensor.transpose(1, 0, *range(2, tensor.ndim))
        tensor = tensor.reshape(6, tensor.shape[1], -1)

    if tensor.ndim > 2:
        tensor = np.nanmean(tensor, axis=1)

    assert tensor.shape[0] == 6, f"expected first dimension to be 6 (stimuli), is {tensor.shape}"
    assert tensor.ndim == 2, f"should be (stimuli, neurons), is {tensor.shape}"

    return tensor