import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from more_itertools import first, take
from itertools import islice
from torch import Tensor

from data_utils.lower_body import part2full
from scripts.demo import get_vertices
from scripts.diversity import init_dataloader
from scripts.render_smpl import load_smpl, create_smplx_model
from torch_utils import match_sequence_length
from trainer.config import load_JsonConfig
from trainer.options import parse_args
from nets.init_model import init_model
from nets import s2g_body_vq
from visualise.rendering import RenderTool
from enum import Enum
from typing import Union
from tqdm import tqdm


class EncoderType(Enum):
    GESTFORMER = "gestformer"
    TALKSHOW = "talkshow"


def main():
    parser = ArgumentParser()
    parser.add_argument("smpl_files", type=Path, nargs="*")
    parser.add_argument("--reconstruction-model-path", type=Path, default=Path("./experiments/ckpt-best.pth"))
    parser = parse_args(parser)
    args = parser.parse_args()

    reconstruction_model_path: Path = args.reconstruction_model_path
    device = torch.device(args.device)

    # print('init smlpx model...')
    # smplx_model = create_smplx_model("./visualise", device)

    print("load smpl files...")
    predicted_result_list = load_smpl(*args.smpl_files)
    # betas = torch.zeros([1, 300], dtype=torch.float64).to(device)

    # print("calculate vertices...")
    # vertices_list, _ = get_vertices(smplx_model, betas, result_list, exp=True)

    config = load_JsonConfig(args.config_file)
    if config.Model.model_name == "gestformer_encoder":
        encoder_type = EncoderType.GESTFORMER
    else:
        encoder_type = EncoderType.TALKSHOW

    print('load reconstruction model...')
    model = init_model(
        config.Model.model_name, args, config
    )
    model_path = reconstruction_model_path
    model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    model.g_body.load_state_dict(model_ckpt['generator']['g_body'])
    model.g_hand.load_state_dict(model_ckpt['generator']['g_hand'])
    model.g_body.eval()
    model.g_hand.eval()

    infer_set, infer_loader, _ = init_dataloader(
        config.Data.data_root,
        ["conan"],
        args,
        config,
    )

    smplx_model = create_smplx_model()
    betas = torch.zeros([1, 300], dtype=torch.float64).to(device)

    # for some reason, TalkShow can encode sequences of different lengths, but Gestformer can only encode sequences of
    # the same length as the training data. This is a problem because the training data is 300 frames long, but the test data
    # is 100 frames long. So we need to truncate the training data to 100 frames. This is done in the following line:
    n_sequences_to_reconstruct = 10
    n_sequences_to_skip = 2
    loader = islice(infer_loader, n_sequences_to_skip, n_sequences_to_skip + n_sequences_to_reconstruct)
    for bat in tqdm(loader, total=n_sequences_to_reconstruct):
        ground_truth_poses = bat['poses'].to(torch.float32).to(device)
        pred_poses = reconstruct(model, ground_truth_poses, encoder_type=encoder_type)
        pred_poses = pred_poses.squeeze()
        if encoder_type == EncoderType.TALKSHOW:
            pred_poses = pred_poses.transpose(1, 0)

        pred_seq_length = len(pred_poses)
        face_pose = match_sequence_length(default_face_pose(), pred_seq_length, axis=0)
        jaw_pose = match_sequence_length(default_jaw_pose(), pred_seq_length, axis=0)

        pred = torch.cat([jaw_pose, pred_poses, face_pose], dim=-1)
        predicted_poses = part2full(pred, stand=False)

        gt_poses = ground_truth_poses[:, model.c_index, :]
        gt_poses = gt_poses.squeeze().transpose(1, 0)
        gt_poses = match_sequence_length(gt_poses, pred_seq_length, axis=0)
        gt_poses = torch.cat([jaw_pose, gt_poses, face_pose], dim=-1)
        ground_truth_poses = part2full(gt_poses, stand=False)

        # currently the generation sequence length, based on body_vq_v2.json, is 88 so we need to pad the sequence
        # to 300 frames which is what part2full() expects

        # predicted_result_list = [predicted_poses]
        # ground_truth_result_list = [ground_truth_poses]
        results_list = [predicted_poses, ground_truth_poses]

        # predicted_vertices_list, _ = get_vertices(smplx_model, betas, predicted_result_list, config.Data.pose.expression)
        # ground_truth_vertices_list, _ = get_vertices(smplx_model, betas, ground_truth_result_list, config.Data.pose.expression)
        vertices_list, _ = get_vertices(smplx_model, betas, results_list, config.Data.pose.expression)

        # result_list = [ground_truth_poses.squeeze().transpoze(1, 0)]
        # ground_truth_vertices_list, _ = get_vertices(smplx_model, betas, result_list, config.Data.pose.expression)

        # copy cur_wav_file to out_dir so that it can be used by render_sequences()
        # also change the name as render_sequences() creates a file with the same name and overwrites the original
        # so I just add a suffix to the name so that the original is not overwritten and can be used by the next iteration

        audio_file = bat["aud_file"][0]
        out_dir = Path("out")
        out_dir.mkdir(exist_ok=True, parents=True)
        rendertool = RenderTool(str(out_dir))
        rendertool._render_sequences(str(audio_file), vertices_list, stand=False, face=False, whole_body=False)
        # rendertool._render_sequences(str(cur_wav_file), predicted_vertices_list, stand=False, face=False, whole_body=False)
        # rendertool._render_sequences(str(cur_wav_file), ground_truth_vertices_list, stand=False, face=False, whole_body=False)




def reconstruct(model: s2g_body_vq, poses: Tensor, encoder_type: Union[str, EncoderType] = EncoderType.TALKSHOW):
    poses = poses[:, model.c_index, :]
    gt_poses = poses.permute(0, 2, 1)
    # if encoder_type == EncoderType.GESTFORMER:
    #     gt_poses = gt_poses[:, :88]

    b_poses, h_poses = gt_poses[..., :model.each_dim[1]], gt_poses[..., model.each_dim[1]:]

    encoder_type = EncoderType(encoder_type)
    if encoder_type == EncoderType.GESTFORMER:
        b_poses_seqs = get_non_overlapping_subsequences(b_poses, 88)
        h_poses_seqs = get_non_overlapping_subsequences(h_poses, 88)
        pred_poses = []
        for b_seq, h_seq in zip(b_poses_seqs, h_poses_seqs):
            encoded_b_poses = model.g_body.module.encode(b_seq)[0]
            encoded_h_poses = model.g_hand.module.encode(h_seq)[0]

            body = model.g_body.module.decode(encoded_b_poses)
            hand = model.g_hand.module.decode(encoded_h_poses)

            pred_poses.append(torch.cat([body, hand], dim=-1).detach().cpu())
        pred_poses = torch.cat(pred_poses, dim=1)[:, :b_poses.shape[1]]
    else:
        _, pred_body = model.g_body(gt_poses=b_poses, id=1)
        _, pred_hand = model.g_hand(gt_poses=h_poses, id=1)

        pred_poses = torch.cat([pred_body, pred_hand], dim=1).detach().cpu()

    return pred_poses


def get_non_overlapping_subsequences(tensor, subseq_length):
    _, seq_length, n_feats = tensor.shape
    num_subsequences = seq_length // subseq_length
    remainder = seq_length % subseq_length

    # Get the non-overlapping subsequences
    subsequences = list(torch.split(tensor[:, :num_subsequences * subseq_length, :], subseq_length, dim=1))

    if remainder > 0:
        # If there's a remainder, pad the last subsequence with repeated values
        last_frame = tensor[:, -remainder, :].repeat(1, subseq_length - remainder, 1)

        # last_frame = tensor[:, -1, :].unsqueeze(1).repeat(1, subseq_length - remainder, 1)
        remainder_subseq = torch.cat((tensor[:, -remainder:, :], last_frame), dim=1)
        subsequences.append(remainder_subseq)

    return tuple(subsequences)


# def get_non_overlapping_subsequences(sequence, subseq_length=88):
#     _, seq_length, n_feats = sequence.shape
#     num_subsequences = seq_length // subseq_length
#     remainder = seq_length % subseq_length
#
#     # Reshape the sequence into non-overlapping subsequences
#     subsequences = sequence[:, :num_subsequences * subseq_length, :].reshape(1, num_subsequences, subseq_length,
#                                                                              n_feats)
#
#     # If there's a remainder, pad the last subsequence with repeated values
#     if remainder > 0:
#         repeated_values = sequence[:, -1:, :]
#         padding = np.repeat(repeated_values, subseq_length - remainder, axis=1)
#         last_subsequence_padded = np.concatenate((sequence[:, -remainder:, :], padding), axis=1)
#         last_subsequence_padded = last_subsequence_padded.reshape(1, 1, subseq_length, n_feats)
#         subsequences = np.concatenate((subsequences, last_subsequence_padded), axis=1)
#
#     return subsequences


def default_jaw_pose(seq_length=300) -> Tensor:
    jaw = torch.load("data/default-poses/jaw.pt")[0, :][None, :].repeat([seq_length, 1])
    return jaw


def default_face_pose(seq_length=300) -> Tensor:
    face = torch.load("data/default-poses/face.pt")[0, :][None, :].repeat([seq_length, 1])
    return face


if __name__ == "__main__":
    main()
