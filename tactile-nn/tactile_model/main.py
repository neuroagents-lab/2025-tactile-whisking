import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import numpy as np
import torch
import random
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from model import TactileModel
from datamodule import TactileDataModule
import sys
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import json


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


@hydra.main(version_base='1.3', config_path='hydra_configs', config_name='config')
def main(cfg: DictConfig):
    set_seed(seed=cfg.train.seed)
    hydra_cfg = HydraConfig.get()

    run_command = ' '.join(sys.argv)
    run_name = cfg.train.run_name

    run_date = hydra_cfg.run.dir.replace('outputs/', '')

    assert run_name is not None, 'please specify a name for this run'

    ckpt_dir = '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers'
    if '1000' in cfg.data.data_dir:
        ckpt_dir = f'{ckpt_dir}/1000hz/ckpt'
    else:  # one can add different datasets by adding `elif` here
        print(f'the dataset is not supported: {cfg.data.data_dir}')
        raise NotImplementedError
    if cfg.train.ssl_finetune:
        decoder_choice = hydra_cfg.runtime.choices['model/decoder']
        model_save_path = f'{ckpt_dir}/{run_name}/lin_ft_{decoder_choice}'
        results_save_path = f'results/{run_name}/lin_ft_{decoder_choice}'
    else:
        model_save_path = f'{ckpt_dir}/{run_name}'
        results_save_path = f'results/{run_name}'

    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    logger = None  # pl.Trainer default
    if cfg.train.use_wandb:
        # Initialize the WandbLogger
        os.environ["WANDB_DIR"] = f'{results_save_path}/wandb'
        if cfg.train.ssl:
            project_name = 'tactile_ssl_pt'
        elif cfg.train.ssl_finetune:
            project_name = 'tactile_ssl_ft'
        else:
            project_name = 'tactile_tnn'

        if '1000' in cfg.data.data_dir:
            project_name = f'{project_name}_1000hz'
        else:  # one can add different datasets by adding `elif` here
            print(f'the dataset is not supported: {cfg.data.data_dir}')
            raise NotImplementedError

        if cfg.train.ssl_finetune:
            run_name = f"{run_name}_lin_ft"
        logger = WandbLogger(project=project_name, name=run_name, save_dir=results_save_path)
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
        logger.experiment.config.update({'run_command': run_command}, allow_val_change=True)

    tactile_model = TactileModel(cfg=cfg)
    data_module = TactileDataModule(cfg=cfg)

    # Callback to save the last training checkpoint
    last_ckpt_callback = ModelCheckpoint(
        dirpath=model_save_path,  # Directory to save the last checkpoint
        filename='train_last',  # Name of the last training checkpoint
        save_top_k=0,  # Don't monitor a metric, only save the last
        save_last=True,  # Save the last training checkpoint
        verbose=True  # Print info about saving
    )
    if not cfg.train.ssl:
        val_monitor, val_mode = 'val_acc', 'max'
    elif cfg.train.ssl in ['SimCLRLoss', 'SimSiamLoss', 'AutoEncoderLoss']:
        val_monitor, val_mode = 'val_loss', 'min'
    else:
        raise NotImplementedError
    # Define the checkpoint callback
    val_ckpt_callback = ModelCheckpoint(
        monitor=val_monitor,  # The metric to monitor
        mode=val_mode,  # 'min' because you want to save the model with the smallest validation loss
        save_top_k=1,  # Save only the best model
        dirpath=model_save_path,
        filename='val_best',
        # Name of the saved model file
        verbose=True  # Print information about saving
    )

    encoder_model_config_file = cfg.model.encoder.get("model_config_file", "no_tnn_config")
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True if 'bn_timevary' in encoder_model_config_file else False),
        # TODO: not the optimal solution
        callbacks=[last_ckpt_callback, val_ckpt_callback],
        logger=logger,
        max_epochs=cfg.train.epochs,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        devices=cfg.train.num_devices,  # or 0 if you're using CPU
    )

    trainer.fit(model=tactile_model, datamodule=data_module,
                ckpt_path=f'{model_save_path}/last.ckpt' if cfg.train.resume_from_ckpt else None)

    # Save the config file
    with open(f'{results_save_path}/hydra_run_config.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    if not cfg.train.ssl:  # using supervised loss
        # using only 1 device for testing following: "It is recommended to test with Trainer(devices=1) since distributed
        # strategies such as DDP use DistributedSampler internally, which replicates some samples to make sure all
        # devices have same batch size in case of uneven inputs. This is helpful to make sure benchmarking for research
        # papers is done the right way." https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html
        test_trainer = pl.Trainer(
            devices=1,
            logger=logger,
            enable_progress_bar=True,
        )

        test_dict = test_trainer.test(model=tactile_model, datamodule=data_module,
                                      ckpt_path=f'{model_save_path}/val_best.ckpt')
        print(test_dict)

        # maybe save the test results here
        with open(f'{results_save_path}/test_result.json', 'w') as f:
            json.dump(test_dict, f, indent=4)


if __name__ == "__main__":
    main()
