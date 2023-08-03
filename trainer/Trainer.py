import os
import sys

import wandb
from wandb.wandb_run import Run as WandbRun

sys.path.append(os.getcwd())

from data_utils import torch_data

from trainer.options import parse_args
from trainer.config import load_JsonConfig
from nets.init_model import init_model

import torch
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import random
import logging
import time
import shutil

from typing import Callable, Dict, Optional, List

def prn_obj(obj):
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))


LossDict = Dict[str, float]
OnStepEndCallable = Callable[[int, LossDict], None]


class Trainer():
    from trainer.callbacks import TrainerCallback

    use_wandb: bool

    def __init__(self, callbacks: Optional[List[TrainerCallback]] = None) -> None:
        self.callbacks = callbacks or []
        parser = parse_args()
        self.args = parser.parse_args()
        self.config = load_JsonConfig(self.args.config_file)
        
        os.environ['smplx_npz_path']=self.config.smplx_npz_path
        os.environ['extra_joint_path']=self.config.extra_joint_path
        os.environ['j14_regressor_path']=self.config.j14_regressor_path

        self.use_wandb = self.args.use_wandb
        # torch.set_default_dtype(torch.float64)
        # wandb_run = wandb.init(project=f's2g_sweep')

        # if self.args.use_wandb:
        #     print('starting wandb sweep agent...')
        #     wandb_key = 'e3d537403fce5c8a99893c2cbe20a8d49a79358d'
        #     os.environ['WANDB_API_KEY'] = wandb_key
        #
        #     default_config=dict(w_b=1,w_h=10)
        #     wandb.init(config=default_config)
        #     self.config.param.w_b=wandb.config.w_b
        #     self.config.param.w_h=wandb.config.w_h
        #     self.config.Train.epochs=30

        # if self.args.use_wandb:
        #     print('starting wandb sweep agent...')
        #     wandb_key = 'e3d537403fce5c8a99893c2cbe20a8d49a79358d'
        #     os.environ['WANDB_API_KEY'] = wandb_key
        #
        #     wandb.init(config=self.args, project="s2g_sweep")
        #     # wandb.config.update(self.args)
        #
        #     self.config.param.w_b=self.args.w_b
        #     self.config.param.w_h=self.args.w_h
        #     self.config.Train.epochs=30

        self.device = torch.device(self.args.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.setup_seed(self.args.seed)
        self.set_train_dir()

        shutil.copy(self.args.config_file, self.train_dir)

        self.generator = init_model(self.config.Model.model_name, self.args, self.config)
        self.init_dataloader()
        self.start_epoch = 0
        self.global_steps = 0
        if self.args.resume:
            self.resume()
        # self.init_optimizer()

    @classmethod
    def create_default(cls):
        from trainer.callbacks import WandbLogger, ModelCheckpoint

        trainer = cls()
        if trainer.use_wandb:
            wandb_run = wandb.init(
                project="gfts",
                config=trainer.config,
                dir=os.getenv("WANDB_DIR", None),
            )
            wandb_logger_cb = WandbLogger.create_from_config(trainer.config, wandb_run)
            model_checkpoint_cb = ModelCheckpoint.create_from_config(trainer.config, wandb_run)
            trainer.register_callback(wandb_logger_cb, model_checkpoint_cb)
        else:
            model_checkpoint_cb = ModelCheckpoint.create_from_config(trainer.config)
            trainer.register_callback(model_checkpoint_cb)
        return trainer

    def register_callback(self, *callbacks: TrainerCallback):
        self.callbacks.extend(callbacks)

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def set_train_dir(self):
        time_stamp = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        train_dir = os.path.join(os.getcwd(), self.args.save_dir, os.path.normpath(
            time_stamp + '-' + self.args.exp_name + '-' + self.config.Log.name))
        # train_dir= os.path.join(os.getcwd(), self.args.save_dir, os.path.normpath(time_stamp+'-'+self.args.exp_name+'-'+time.strftime("%H:%M:%S")))
        os.makedirs(train_dir, exist_ok=True)
        log_file=os.path.join(train_dir, 'train.log')

        fmt="%(asctime)s-%(lineno)d-%(message)s"
        logging.basicConfig(
            stream=sys.stdout, level=logging.INFO,format=fmt, datefmt='%m/%d %I:%M:%S %p'
        )
        fh=logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)
        self.train_dir = train_dir

    def resume(self):
        print('resume from a previous ckpt')
        ckpt = torch.load(self.args.pretrained_pth)
        self.generator.load_state_dict(ckpt['generator'])
        self.start_epoch = ckpt['epoch']
        self.global_steps = ckpt['global_steps']
        self.generator.global_step = self.global_steps


    def init_dataloader(self):
        if 'freeMo' in self.config.Model.model_name:
            if self.config.Data.data_root.endswith('.csv'):
                raise NotImplementedError
            else:
                data_class = torch_data
            
            self.train_set = data_class(
                data_root=self.config.Data.data_root,
                speakers=self.args.speakers,
                split='train',
                limbscaling=self.config.Data.pose.augmentation,
                normalization=self.config.Data.pose.normalization,
                norm_method=self.config.Data.pose.norm_method,
                split_trans_zero=True,
                num_pre_frames=self.config.Data.pose.pre_pose_length,
                num_frames=self.config.Data.pose.generate_length,
                aud_feat_win_size=self.config.Data.aud.aud_feat_win_size,
                aud_feat_dim=self.config.Data.aud.aud_feat_dim,
                feat_method=self.config.Data.aud.feat_method,
                context_info=self.config.Data.aud.context_info
            )

            if self.config.Data.pose.normalization:
                self.norm_stats = (self.train_set.data_mean, self.train_set.data_std)
                save_file = os.path.join(self.train_dir, 'norm_stats.npy')
                np.save(save_file, self.norm_stats, allow_pickle=True)

            self.train_set.get_dataset()
            self.trans_set = self.train_set.trans_dataset
            self.zero_set = self.train_set.zero_dataset

            self.trans_loader = data.DataLoader(self.trans_set, batch_size=self.config.DataLoader.batch_size, shuffle=True, num_workers=self.config.DataLoader.num_workers, drop_last=True) 
            self.zero_loader = data.DataLoader(self.zero_set, batch_size=self.config.DataLoader.batch_size, shuffle=True, num_workers=self.config.DataLoader.num_workers, drop_last=True)
        elif 'smplx' in self.config.Model.model_name or 's2g' in self.config.Model.model_name:
            data_class = torch_data

            self.train_set = data_class(
                data_root=self.config.Data.data_root,
                speakers=self.args.speakers,
                split='train',
                limbscaling=self.config.Data.pose.augmentation,
                normalization=self.config.Data.pose.normalization,
                norm_method=self.config.Data.pose.norm_method,
                split_trans_zero=False,
                num_pre_frames=self.config.Data.pose.pre_pose_length,
                num_frames=self.config.Data.pose.generate_length,
                num_generate_length=self.config.Data.pose.generate_length,
                aud_feat_win_size=self.config.Data.aud.aud_feat_win_size,
                aud_feat_dim=self.config.Data.aud.aud_feat_dim,
                feat_method=self.config.Data.aud.feat_method,
                context_info=self.config.Data.aud.context_info,
                smplx=True,
                audio_sr=22000,
                convert_to_6d=self.config.Data.pose.convert_to_6d,
                expression=self.config.Data.pose.expression,
                config=self.config
            )
            if self.config.Data.pose.normalization:
                self.norm_stats = (self.train_set.data_mean, self.train_set.data_std)
                save_file = os.path.join(self.train_dir, 'norm_stats.npy')
                np.save(save_file, self.norm_stats, allow_pickle=True)
            self.train_set.get_dataset()
            self.train_loader = data.DataLoader(self.train_set.all_dataset,
                                                batch_size=self.config.DataLoader.batch_size, shuffle=True,
                                                num_workers=self.config.DataLoader.num_workers, drop_last=True)
        else:
            data_class = torch_data

            self.train_set = data_class(
                data_root=self.config.Data.data_root,
                speakers=self.args.speakers,
                split='train',
                limbscaling=self.config.Data.pose.augmentation,
                normalization=self.config.Data.pose.normalization,
                norm_method=self.config.Data.pose.norm_method,
                split_trans_zero=False,
                num_pre_frames=self.config.Data.pose.pre_pose_length,
                num_frames=self.config.Data.pose.generate_length,
                aud_feat_win_size=self.config.Data.aud.aud_feat_win_size,
                aud_feat_dim=self.config.Data.aud.aud_feat_dim,
                feat_method=self.config.Data.aud.feat_method,
                context_info=self.config.Data.aud.context_info,
                config=self.config
            )

            if self.config.Data.pose.normalization:
                self.norm_stats = (self.train_set.data_mean, self.train_set.data_std)
                save_file = os.path.join(self.train_dir, 'norm_stats.npy')
                np.save(save_file, self.norm_stats, allow_pickle=True)

            self.train_set.get_dataset()

            self.train_loader = data.DataLoader(self.train_set.all_dataset, batch_size=self.config.DataLoader.batch_size, shuffle=True, num_workers=self.config.DataLoader.num_workers, drop_last=True)

    def init_optimizer(self):
        pass

    def print_func(self, loss_dict, steps):
        info_str = ['global_steps:%d'%(self.global_steps)]
        info_str += ['%s:%.4f'%(key, loss_dict[key]/steps) for key in list(loss_dict.keys())]
        logging.info(','.join(info_str))
    
    def save_model(self, epoch, wandb_run: Optional[WandbRun] = None, save_as_best=False):
        # if 'vq' in self.config.Model.model_name:
        #     state_dict = {
        #         'g_body': self.g_body.state_dict(),
        #         'g_hand': self.g_hand.state_dict(),
        #         'epoch': epoch,
        #         'global_steps': self.global_steps
        #     }
        # else:
        state_dict = {
            'generator': self.generator.state_dict(),
            'epoch': epoch,
            'global_steps': self.global_steps
        }
        name_suffix = "best" if save_as_best else epoch
        save_name = os.path.join(self.train_dir, f'ckpt-{name_suffix}.pth')
        torch.save(state_dict, save_name)
        if wandb_run:
            wandb_run.save(save_name)

    def train_epoch(self, epoch) -> LossDict:
        epoch_loss_dict = {} #最好是追踪每个epoch的loss变换
        epoch_steps = 0
        if 'freeMo' in self.config.Model.model_name:
            for bat in zip(self.trans_loader, self.zero_loader):
                self.global_steps += 1
                epoch_steps += 1
                _, loss_dict = self.generator(bat)
                
                if epoch_loss_dict:#非空
                    for key in list(loss_dict.keys()):
                        epoch_loss_dict[key] += loss_dict[key]
                else:
                    for key in list(loss_dict.keys()):
                        epoch_loss_dict[key] = loss_dict[key]

                if self.global_steps % self.config.Log.print_every == 0:
                    self.print_func(epoch_loss_dict, epoch_steps)

                self.on_step_end(loss_dict, self.global_steps)
        else:
            # self.config.Model.model_name==smplx_S2G
            for bat in self.train_loader:
                # if epoch_steps == 1000:
                #     break
                self.global_steps += 1
                epoch_steps += 1
                bat['epoch'] = epoch

                _, loss_dict = self.generator(bat)
                if epoch_loss_dict:#非空
                    for key in list(loss_dict.keys()):
                        epoch_loss_dict[key] += loss_dict[key]
                else:
                    for key in list(loss_dict.keys()):
                        epoch_loss_dict[key] = loss_dict[key]
                if self.global_steps % self.config.Log.print_every == 0:
                    self.print_func(epoch_loss_dict, epoch_steps)

                self.on_step_end(loss_dict)

            return epoch_loss_dict

    def train(self):
        logging.info('start_training')

        self.on_train_begin()
        self.total_loss_dict = {}
        for epoch in range(self.start_epoch, self.config.Train.epochs):
            logging.info('epoch:%d'%(epoch))
            epoch_losses = self.train_epoch(epoch)

            self.on_epoch_end(epoch_losses, epoch)

    def on_train_begin(self):
        for cb in self.callbacks:
            cb.on_train_begin(self)

    def on_epoch_end(self, epoch_losses: LossDict, epoch: int):
        for cb in self.callbacks:
            cb.on_epoch_end(self, epoch_losses, epoch)

    def on_step_end(self, step_losses: LossDict, step: int):
        for cb in self.callbacks:
            cb.on_step_end(self, step_losses, step)
