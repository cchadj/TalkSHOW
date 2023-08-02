from torch import Tensor

import utils.optim
from utils.optim import ScheduledOptim
from vqgan.vqmodules.gan_models import setup_vq_transformer, calc_vq_loss_gestformer, VQModelTransformer

from nets.layers import *
from nets.base import TrainWrapperBaseClass
from data_utils.lower_body import c_index, c_index_3d, c_index_6d


class TrainWrapper(TrainWrapperBaseClass):
    '''
    a wrapper receving a batch from data_utils and calculate loss
    '''
    g_body: VQModelTransformer
    g_body_optimizer: ScheduledOptim

    g_hand: VQModelTransformer
    g_hand: ScheduledOptim

    g: VQModelTransformer
    g_optimizer: ScheduledOptim

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device(self.args.device)
        self.global_step = 0

        self.convert_to_6d = self.config.Data.pose.convert_to_6d
        self.expression = self.config.Data.pose.expression
        self.epoch = 0
        self.init_params()
        self.num_classes = 4
        self.composition = self.config.Model.composition
        self.bh_model = self.config.Model.bh_model

        self.audio = False
        self.discriminator = None
        self.discriminator_optimizer = None

        if self.composition:
            b_config = config.as_dict()
            b_config["transformer_config"]["in_dim"] = 39

            b_generator, b_optimizer, _ = setup_vq_transformer(args, b_config, device=self.device)
            self.g_body = b_generator
            self.g_body_optimizer = b_optimizer

            h_config = config.as_dict()
            h_config["transformer_config"]["in_dim"] = 90
            h_generator, h_optimizer, _ = setup_vq_transformer(args, h_config, device=self.device)
            self.g_hand = h_generator
            self.g_hand_optimizer = h_optimizer
        else:
            generator, optimizer, _ = setup_vq_transformer(args, config.as_dict(), device=self.device)
            self.g = generator
            self.g_optimizer = optimizer

        if self.convert_to_6d:
            self.c_index = c_index_6d
        else:
            self.c_index = c_index_3d

        # if torch.cuda.device_count() > 1:
            # self.generator = torch.nn.DataParallel(self.generator, device_ids=[0, 1])
        super().__init__(args, config)

    def __call__(self, bat: dict):
        self.global_step += 1

        poses = bat['poses'].to(self.device).to(torch.float32)
        poses = poses[:, self.c_index, :]
        gt_poses = poses.permute(0, 2, 1)

        loss = 0
        loss_dict = {}
        if self.composition:
            b_poses = gt_poses[..., :self.each_dim[1]]
            h_poses = gt_poses[..., self.each_dim[1]:]
            loss_dict, loss = self.vq_train(b_poses[:, :], self.g_body_optimizer, self.g_body, loss_dict, loss, "b")
            loss_dict, loss = self.vq_train(h_poses[:, :], self.g_hand_optimizer, self.g_hand, loss_dict, loss, "h")
        else:
            loss_dict, loss = self.vq_train(gt_poses[:, :], self.g_optimizer, self.g, loss_dict, loss, "g")

        total_loss = None
        return total_loss, loss_dict

    def vq_train(self, gt: Tensor, optimizer: utils.optim.ScheduledOptim, model: VQModelTransformer, accumulated_losses: dict, total_loss, name: str, pre=None):
        # e_q_loss, x_recon = model(gt)
        x_recon, e_q_loss = model(gt)
        loss, loss_dict = self.get_loss(pred_poses=x_recon, gt_poses=gt, e_q_loss=e_q_loss, pre=pre)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step_and_update_lr()

        for key in list(loss_dict.keys()):
            accumulated_losses[name + key] = loss_dict.get(key, 0).item()
        return accumulated_losses, total_loss

    def init_optimizer(self):
        # already setup by setup_vq_transform
        pass

    def init_params(self):
        if self.config.Data.pose.convert_to_6d:
            scale = 2
        else:
            scale = 1

        global_orient = round(0 * scale)
        leye_pose = reye_pose = round(0 * scale)
        jaw_pose = round(0 * scale)
        body_pose = round((63 - 24) * scale)
        left_hand_pose = right_hand_pose = round(45 * scale)
        if self.expression:
            expression = 100
        else:
            expression = 0

        b_j = 0
        jaw_dim = jaw_pose
        b_e = b_j + jaw_dim
        eye_dim = leye_pose + reye_pose
        b_b = b_e + eye_dim
        body_dim = global_orient + body_pose
        b_h = b_b + body_dim
        hand_dim = left_hand_pose + right_hand_pose
        b_f = b_h + hand_dim
        face_dim = expression

        self.dim_list = [b_j, b_e, b_b, b_h, b_f]
        self.full_dim = jaw_dim + eye_dim + body_dim + hand_dim
        self.pose = int(self.full_dim / round(3 * scale))
        self.each_dim = [jaw_dim, eye_dim + body_dim, hand_dim, face_dim]

    def get_loss(self,
                 pred_poses,
                 gt_poses,
                 e_q_loss,
                 pre=None
                 ):
        loss_dict = {}


        rec_loss = torch.mean(torch.abs(pred_poses - gt_poses))
        v_pr = pred_poses[:, 1:] - pred_poses[:, :-1]
        v_gt = gt_poses[:, 1:] - gt_poses[:, :-1]
        velocity_loss = torch.mean(torch.abs(v_pr - v_gt))

        if pre is None:
            f0_vel = 0
        else:
            v0_pr = pred_poses[:, 0] - pre[:, -1]
            v0_gt = gt_poses[:, 0] - pre[:, -1]
            f0_vel = torch.mean(torch.abs(v0_pr - v0_gt))

        gen_loss = rec_loss + e_q_loss + velocity_loss + f0_vel

        loss_dict['rec_loss'] = rec_loss
        loss_dict['velocity_loss'] = velocity_loss
        # loss_dict['e_q_loss'] = e_q_loss
        if pre is not None:
            loss_dict['f0_vel'] = f0_vel

        return gen_loss, loss_dict

    def state_dict(self):
        if self.composition:
            model_state = {
                'g_body': self.g_body.state_dict(),
                'g_body_optim': self.g_body_optimizer.state_dict(),
                'g_hand': self.g_hand.state_dict(),
                'g_hand_optim': self.g_hand_optimizer.state_dict(),
                'discriminator': self.discriminator.state_dict() if self.discriminator is not None else None,
                'discriminator_optim': self.discriminator_optimizer.state_dict() if self.discriminator is not None else None
            }
        else:
            model_state = {
                'g': self.g.state_dict(),
                'g_optim': self.g_optimizer.state_dict(),
                'discriminator': self.discriminator.state_dict() if self.discriminator is not None else None,
                'discriminator_optim': self.discriminator_optimizer.state_dict() if self.discriminator is not None else None
            }
        return model_state

    def load_state_dict(self, state_dict):
        if self.composition:
            self.g_body.load_state_dict(state_dict['g_body'])
            self.g_body_optimizer.load_state_dict(state_dict['g_body_optim'])
            self.g_hand.load_state_dict(state_dict['g_hand'])
            self.g_hand_optimizer.load_state_dict(state_dict['g_hand_optim'])
        else:
            self.g.load_state_dict(state_dict['g'])
            self.g_optimizer.load_state_dict(state_dict['g_optim'])
