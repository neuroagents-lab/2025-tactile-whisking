import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import random
import os
from hydra.core.hydra_config import HydraConfig
from enc_att_dec.model import EncAttDec


def set_seed(seed: int = 42) -> None:
    # https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def dict_to_numpy(d):
    if isinstance(d, dict):
        return {key: dict_to_numpy(value) for key, value in d.items()}
    elif isinstance(d, torch.Tensor):
        return d.detach().clone().cpu().numpy()
    else:
        return d


@hydra.main(version_base='1.3', config_path='hydra_configs', config_name='config')
def main(cfg: DictConfig):
    set_seed(seed=cfg.train.seed)
    hydra_cfg = HydraConfig.get()

    run_name = cfg.train.run_name

    assert run_name is not None, 'please specify a name for this run'

    ckpt_dir = '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers'
    if '1000' in cfg.data.data_dir:
        ckpt_dir = f'{ckpt_dir}/1000hz/ckpt'
    else:  # one can add different datasets by adding `elif` here
        print(f'the dataset is not supported: {cfg.data.data_dir}')
        raise NotImplementedError

    if cfg.train.ssl_finetune:
        decoder_choice = hydra_cfg.runtime.choices['model/decoder']
        ckpt_path = f'{ckpt_dir}/{run_name}/lin_ft_{decoder_choice}/val_best.ckpt'
    else:
        ckpt_path = f'{ckpt_dir}/{run_name}/val_best.ckpt'

    results_save_path = f'results/{run_name}/neural_fitting'

    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    device = 'cuda:0'
    # create and load the model
    model = EncAttDec(cfg=cfg)

    if cfg.test.load_ckpt_for_nf:
        print('=' * 20, f'loading checkpoint from {ckpt_path} for neural data fitting...', '=' * 20)
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict=state_dict, strict=True)
    else:
        print('=' * 20, 'a randomly initialized model!', '=' * 20)
    model.to(device=device)

    # get activations
    input_loaded = np.load(cfg.test.activation_stimuli)
    input_data = torch.tensor(input_loaded["data"], device=device, dtype=torch.float)
    pred, activations = model(x=input_data, return_activations=True)

    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / 1024 ** 2  # ≈ size for float32 weights
    print(f"Total parameters : {total_params:,}")
    print(f"Model size       : {size_mb:,.2f} MB (FP32)")
    print(f"%%% {run_name}: {total_params},")

    print('enc keys', activations['enc'].keys())
    print('att keys', activations['att'].keys())
    print('dec keys', activations['dec'].keys())

    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / 1024 ** 2  # ≈ size for float32 weights
    print(f"Total parameters : {total_params:,}")
    print(f"Model size       : {size_mb:,.2f} MB (FP32)")

    output_path = hydra_cfg.runtime.output_dir + "/activations.npz"
    activations_cpu = dict_to_numpy(activations)
    np.savez(output_path, **activations_cpu)
    print("saved to", output_path)


if __name__ == "__main__":
    main()
