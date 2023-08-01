from vqgan.vqmodules.gan_models import setup_vq_transformer, calc_vq_loss_gestformer, VQModelTransformer

from nets.layers import *
from nets.base import TrainWrapperBaseClass
from data_utils.lower_body import c_index, c_index_3d, c_index_6d


class TrainWrapper(TrainWrapperBaseClass):
    '''
    a wrapper receving a batch from data_utils and calculate loss
    '''
    generator: VQModelTransformer

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

        generator, g_optimizer, start_epoch = setup_vq_transformer(args, config.as_dict(), device=self.device)
        self.generator = generator
        self.generator_optimizer = g_optimizer

        if self.convert_to_6d:
            self.c_index = c_index_6d
        else:
            self.c_index = c_index_3d

        # if torch.cuda.device_count() > 1:
            # self.generator = torch.nn.DataParallel(self.generator, device_ids=[0, 1])
        super().__init__(args, config)

    def init_optimizer(self):
        # already setup by setup_vq_transform
        pass

    def state_dict(self):
        model_state = {
            'generator': self.generator.state_dict(),
            'generator_optim': self.generator_optimizer.state_dict(),
            'audioencoder': self.audioencoder.state_dict() if self.audio else None,
            'audioencoder_optim': self.audioencoder_optimizer.state_dict() if self.audio else None,
            'discriminator': self.discriminator.state_dict() if self.discriminator is not None else None,
            'discriminator_optim': self.discriminator_optimizer.state_dict() if self.discriminator is not None else None
        }
        return model_state

    def load_state_dict(self, state_dict):

        from collections import OrderedDict
        new_state_dict = OrderedDict()  # create new OrderedDict that does not contain `module.`
        for k, v in state_dict.items():
            sub_dict = OrderedDict()
            if v is not None:
                for k1, v1 in v.items():
                    name = k1.replace('module.', '')
                    sub_dict[name] = v1
            new_state_dict[k] = sub_dict
        state_dict = new_state_dict
        if 'generator' in state_dict:
            self.generator.load_state_dict(state_dict['generator'])
        else:
            self.generator.load_state_dict(state_dict)

        if 'generator_optim' in state_dict and self.generator_optimizer is not None:
            self.generator_optimizer.load_state_dict(state_dict['generator_optim'])

        if self.discriminator is not None:
            self.discriminator.load_state_dict(state_dict['discriminator'])

            if 'discriminator_optim' in state_dict and self.discriminator_optimizer is not None:
                self.discriminator_optimizer.load_state_dict(state_dict['discriminator_optim'])

        if 'audioencoder' in state_dict and self.audioencoder is not None:
            self.audioencoder.load_state_dict(state_dict['audioencoder'])

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

    def __call__(self, bat: dict):
        self.global_step += 1

        poses = bat['poses'].to(self.device).to(torch.float32)
        poses = poses[:, self.c_index, :]
        gt_poses = poses.permute(0, 2, 1)
        b_poses = gt_poses[..., :self.each_dim[1]]

        loss = 0
        loss_dict = {}
        loss_dict, loss = self.vq_train(b_poses[:, :], 'b', self.generator, loss_dict, loss)

        total_loss = None
        return total_loss, loss_dict

    def vq_train(self, gt, name, model, dict, total_loss, pre=None):
        # e_q_loss, x_recon = model(gt)
        x_recon, e_q_loss = model(gt)
        loss, loss_dict = self.get_loss(pred_poses=x_recon, gt_poses=gt, e_q_loss=e_q_loss, pre=pre)

        optimizer = self.generator_optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step_and_update_lr()

        for key in list(loss_dict.keys()):
            dict[name + key] = loss_dict.get(key, 0).item()
        return dict, total_loss

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
