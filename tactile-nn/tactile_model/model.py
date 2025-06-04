from pytorch_lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy
from utils.lr_scheduler import get_openai_lr, get_cosine_schedule_with_warmup, CosineAnnealingWarmupScheduler
from utils.metric_recorder import MetricRecorder
from utils.input_transform import SSLTransform
from utils.optimizer import LARS
import torch
import torch.nn as nn
from enc_att_dec.model import EncAttDec
import loss_functions


class WeightChangeTracker:
    """
    Check if the weights of NN change
    """

    def __init__(self, model, model_name):
        super(WeightChangeTracker, self).__init__()
        self.model = model
        self.model_name = model_name
        assert isinstance(self.model, nn.Module)
        self.weights = {}
        self._store_weights()

    def _store_weights(self):
        for name, param in self.model.named_parameters():
            self.weights[name] = param.clone().detach()
            print(f"{name} of shape={param.shape} registered")

    def assert_change(self):
        for name, param in self.model.named_parameters():
            if self.weights[name].to(param.device).equal(param):
                continue
            print('\"{}\" weights changed: {}'.format(self.model_name, name),
                  f'by {(self.weights[name].to(param.device) - param).sum()}')


class TactileModel(LightningModule):
    def __init__(self, cfg):
        super(TactileModel, self).__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train
        self.test_cfg = cfg.test

        self.model = EncAttDec(cfg=cfg)

        if self.train_cfg.ssl_finetune:

            assert not self.train_cfg.ssl, f'fine-tuning SSL-trained model, loss can not be SSLs, but got {self.train_cfg.ssl}'

            ckpt_dir = '/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers'
            if '1000' in cfg.data.data_dir:
                ckpt_dir = f'{ckpt_dir}/1000hz/ckpt'
            else:  # one can add different datasets by adding `elif` here
                print(f'the dataset is not supported: {cfg.data.data_dir}')
                raise NotImplementedError
            ckpt_path = f'{ckpt_dir}/{self.train_cfg.run_name}/val_best.ckpt'

            print(f'loading checkpoint from {ckpt_path} for linear classification fine-tuning...')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            # "state_dict" is typically the key used by PL for model weights
            state_dict = checkpoint["state_dict"]

            filtered_state_dict = {
                k: v
                for k, v in state_dict.items()
                if k.startswith("model.encoder") or k.startswith("model.attender")
            }

            self.load_state_dict(filtered_state_dict, strict=False)

            # Freeze encoder and attender
            for name, param in self.model.encoder.named_parameters():
                param.requires_grad = False
                assert not torch.isnan(param).any(), f"parameter={name} of shape={param.shape} contains NaN values!"
            for name, param in self.model.attender.named_parameters():
                param.requires_grad = False
                assert not torch.isnan(param).any(), f"parameter={name} of shape={param.shape} contains NaN values!"

        # train configurations
        self.lr = self.train_cfg.lr
        self.step_size = self.train_cfg.step_size
        self.epochs = 100 if self.train_cfg.epochs is None else self.train_cfg.epochs
        self.warmup_epochs = self.epochs // 4 if self.train_cfg.warmup_epochs is None else self.train_cfg.warmup_epochs
        self.weight_decay = self.train_cfg.weight_decay
        self.momentum = self.train_cfg.momentum

        self.ssl = self.train_cfg.ssl  # either False (supervised training) or a string indicating the SSL class

        # metrics to compute (e.g., loss, acc)
        if not self.ssl:
            print('=' * 50, 'supervised learning', '=' * 50)
            self.loss = torch.nn.CrossEntropyLoss()
            self.top1_acc = MulticlassAccuracy(num_classes=self.data_cfg.num_classes, average='micro', top_k=1)
            self.top5_acc = MulticlassAccuracy(num_classes=self.data_cfg.num_classes, average='micro', top_k=5)
        else:
            print('=' * 50, f'unsupervised learning with {self.ssl}', '=' * 50)
            self.ssl_transform = SSLTransform(use_temporal=self.train_cfg.ssl_temp_tran)
            args = self.train_cfg.ssl_args or {}  # empty dictionary to deal with ssl_cfg is None

            if self.ssl == 'AutoEncoderLoss':
                args['C'], args['H'], args['W'] = self.data_cfg.input_shape
                print(f"AutoEncoder loss with C={args['C']}-H={args['H']}-W={args['W']}")
            self.loss = getattr(loss_functions, self.ssl)(model_output_dim=self.model_cfg.hidden_dim, **args)

        # train & validation dynamics
        self.train_recorder = MetricRecorder(name='train_dynamics', verbose=True)
        self.val_recorder = MetricRecorder(name='val_dynamics', verbose=True)
        self.train_losses = []
        self.val_losses = []

    def configure_optimizers(self):
        if self.lr is None:
            self.lr = get_openai_lr(self.model)
            print(f"Using OpenAI max lr of {self.lr}.")

        if not self.ssl:  # supervised or linear ft
            if self.train_cfg.ssl_finetune:
                print('=' * 50,
                      'adding only decoder parameters to optimizer',
                      '=' * 50)
                all_parameters = list(self.model.decoder.parameters())  # only allow training the linear decoder
            else:
                all_parameters = list(self.model.parameters())
        elif self.ssl in ['SimCLRLoss', 'SimSiamLoss', 'AutoEncoderLoss']:
            # self-supervised
            all_parameters = list(self.model.parameters()) + list(self.loss.parameters())
            if self.ssl == 'SimSiamLoss' and (self.train_cfg.batch_size * self.train_cfg.num_devices) < 1024:
                print('in SimSiamLoss, setting warmup epochs=0')
                self.warmup_epochs = 0
        else:
            raise NotImplementedError

        if self.train_cfg.optimizer == 'sgd':
            print('=' * 50,
                  f'using SGD optimizer-lr={self.lr}-momentum={self.momentum}-weight-decay={self.weight_decay}',
                  '=' * 50)
            optimizer = torch.optim.SGD(all_parameters, lr=self.lr, momentum=self.momentum,
                                        weight_decay=self.weight_decay)
        elif self.train_cfg.optimizer == 'adamw':
            print('=' * 50, f'using AdamW optimizer-lr={self.lr}-weight-decay={self.weight_decay}', '=' * 50)
            optimizer = torch.optim.AdamW(all_parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.train_cfg.optimizer == 'adam':
            print('=' * 50, f'using Adam optimizer-lr={self.lr}-weight-decay={self.weight_decay}', '=' * 50)
            optimizer = torch.optim.Adam(all_parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.train_cfg.optimizer == 'lars':
            print('=' * 50,
                  f'using LARS optimizer-lr={self.lr}-momentum={self.momentum}-weight-decay={self.weight_decay}',
                  '=' * 50)
            optimizer = LARS(all_parameters, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        if self.train_cfg.scheduler == 'step_lr':
            print('=' * 50, f'using StepLR schedule-step-size={self.step_size}', '=' * 50)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.step_size, )
        elif self.train_cfg.scheduler == 'cos_w_warmup':
            print('=' * 50,
                  f'using cosine schedule with warmup-epochs={self.warmup_epochs}-train-epochs={self.epochs}',
                  '=' * 50)
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=self.warmup_epochs,
                                                        num_training_steps=self.epochs)
        elif self.train_cfg.scheduler == 'constant_lr':
            print('=' * 50, f'using ConstantLR schedule-train-epochs={self.epochs}', '=' * 50)
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=1.0, total_iters=self.epochs)
            # multiply the lr with 1.0 with every self.epochs (constant lr over all epochs)
        elif self.train_cfg.scheduler == 'cos_annealing_w_warmup':
            print('=' * 50,
                  f'using cosine annealing with warmup schedule with train-epochs={self.epochs}'
                  f'-initial-lr={self.lr}-min-lr={self.train_cfg.min_lr}'
                  f'-warmup-epochs={self.warmup_epochs}-warmup-ratio={self.train_cfg.warmup_ratio}',
                  '=' * 50)
            scheduler = CosineAnnealingWarmupScheduler(optimizer=optimizer, num_epochs=self.epochs, initial_lr=self.lr,
                                                       min_lr=self.train_cfg.min_lr, warmup_epochs=self.warmup_epochs,
                                                       warmup_ratio=self.train_cfg.warmup_ratio)
        else:
            raise NotImplementedError

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Metric to monitor for schedulers like `ReduceLROnPlateau`
                'interval': 'epoch',  # The unit of the scheduler's step size, could also be 'step'. 'epoch' updates
                # the scheduler on epoch end whereas 'step' updates it after a optimizer update.
                'frequency': 1,  # How many epochs/steps should pass between calls to `scheduler.step()`. 1 corresponds
                # to updating the learning rate after every epoch/step.
            }
        }

    # Dummy forward pass to initialize uninitialized parameters (e.g., LazyLayers) before DDP
    def setup(self, stage=None):
        data_n_times = self.data_cfg.get("n_times", None)  # whether we have the time dimension in the data
        if data_n_times is None:
            dummy_input = torch.randn((5,) + tuple(self.data_cfg.input_shape), device=self.device)  # batch_size=5
        else:
            dummy_input = torch.randn((5, data_n_times) + tuple(self.data_cfg.input_shape),
                                      device=self.device)  # batch_size=5

        self.model.to(self.device)
        # move to GPU to enable joint inference on shapes with att=Mamba (gpu-only) + dec=lazy-linear

        print('=' * 20, 'dummy input shape', dummy_input.shape)
        self.forward(dummy_input)

    def forward(self, x):
        # x: batch of image(-alike) inputs (N, T, C, H, W) or (N, C, H, W). (batch, time, channel, height, width)
        if not self.ssl:  # supervised learning
            return self.model(x)  # (bs, num_class)
        elif self.ssl == 'SimCLRLoss':
            bs, *rest = x.shape
            x = torch.stack([self.ssl_transform(x), self.ssl_transform(x)], dim=1).reshape(2 * bs, *rest)
            # (2*N, T, C, H, W) or (2*N, C, H, W)
            return self.model(x)  # (2*bs, d)
        elif self.ssl == 'SimSiamLoss':
            return self.model(self.ssl_transform(x)), self.model(self.ssl_transform(x))
        elif self.ssl == 'AutoEncoderLoss':
            return self.model(x), x
        else:
            raise NotImplementedError

    def compute_loss(self, pred, targets):
        if not self.ssl:  # supervised learning
            return self.loss(pred, targets)
        elif self.ssl == 'SimCLRLoss':
            return self.loss(x=pred)
        elif self.ssl == 'SimSiamLoss':
            return self.loss(x1=pred[0], x2=pred[1])
        elif self.ssl == 'AutoEncoderLoss':
            return self.loss(hidden_vec=pred[0], inp=pred[1])
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch  # batch returns: inputs, targets, indices
        pred = self.forward(inputs)
        loss = self.compute_loss(pred=pred, targets=targets)
        self.train_recorder.update(loss=loss, num_samples=inputs.shape[0])

        if not torch.isfinite(loss).item():
            # 1) mark the run as failed in the logs (optional)
            self.log("train_loss_nonfinite", loss, prog_bar=True, logger=True, rank_zero_only=True)
            # 2) ask the Trainer to shut down gracefully
            self.trainer.should_stop = True

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        # (bs, 22, 30, 5, 7)
        pred = self.forward(inputs)  # (bs, N, C, H, W) ---> (bs, num_class)
        if not self.ssl:  # supervised learning
            val_loss = self.compute_loss(pred=pred, targets=targets)
            val_acc = self.top1_acc(preds=pred, target=targets)
        else:
            if self.ssl in ['SimCLRLoss', 'SimSiamLoss', 'AutoEncoderLoss']:
                val_loss = self.compute_loss(pred=pred, targets=targets)
                val_acc = None
            else:
                raise NotImplementedError

        self.val_recorder.update(loss=val_loss, num_samples=inputs.shape[0], acc=val_acc)

        return val_loss

    def test_step(self, batch, batch_idx):
        # Defines a single test step. Similar to validation_step, but for test data.
        inputs, targets, _ = batch
        # Forward pass to get predictions
        pred = self.forward(inputs)
        # Compute accuracy
        test_acc_top1 = self.top1_acc(preds=pred, target=targets)
        test_acc_top5 = self.top5_acc(preds=pred, target=targets)
        # Log the test accuracy
        self.log('test_acc_top1', test_acc_top1, sync_dist=True, prog_bar=True)
        self.log('test_acc_top5', test_acc_top5, sync_dist=True, prog_bar=True)

        return {'test_acc_top1': test_acc_top1, 'test_acc_top5': test_acc_top5}

    def on_train_epoch_start(self) -> None:
        self.train_recorder.reset()  # reset the recorded dynamics for the next epoch

    def on_train_epoch_end(self) -> None:
        lr = self.lr_schedulers().get_last_lr()[0]
        train_metric = self.train_recorder.fetch_and_print(epoch=self.current_epoch, lr=lr)
        self.log('train_loss', train_metric['avr_loss'], sync_dist=True)
        self.train_losses.append(train_metric['avr_loss'])

    def on_validation_epoch_start(self) -> None:
        self.val_recorder.reset()  # reset the recorded dynamics for the next epoch

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            val_metric = self.val_recorder.fetch_and_print(epoch=self.current_epoch, lr=None)
            self.log('val_loss', val_metric['avr_loss'], sync_dist=True)
            self.log('val_acc', val_metric['avr_acc'], sync_dist=True)
            self.val_losses.append(val_metric['avr_loss'])

    def on_save_checkpoint(self, checkpoint):
        # Save the lists of train and val losses
        checkpoint['train_losses'] = self.train_losses
        checkpoint['val_losses'] = self.val_losses

    def on_load_checkpoint(self, checkpoint):
        # Load the lists of train and val losses
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print('-' * 20)
        print(f'getting the train losses of length {len(self.train_losses)} & the val losses of length '
              f'{len(self.val_losses)} from the latest ckpt')
        train_losses_len = len(self.train_losses)
        val_losses_len = len(self.val_losses)
        if train_losses_len > val_losses_len:  # training collapsed after train epoch & before val epoch
            self.train_losses = self.train_losses[:val_losses_len]
        elif val_losses_len > train_losses_len:
            raise Exception  # then sth. is really off
        print('-' * 20)
