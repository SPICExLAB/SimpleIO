"""
Evaluate predicted body pose against GT xsens_pose per sequence.

Inputs:
  - net_output.pickle written by inference_motion.py for the pose model.
    Each entry has keys 'ts' (downsampled timestamps) and 'net_pose'
    (B*T, 69) root-translated joint positions.
  - Same dataset configuration used for training/inference.

Metric:
  - MPJPE (mean per-joint position error, m): mean over (time, 23 joints)
    of the Euclidean distance between predicted and GT joint position.
  - Per-joint MPJPE for inspection.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import os
import json
import glob
import pickle
import argparse

import numpy as np
import torch
from pyhocon import ConfigFactory

from datasets import SeqDataset
from utils import CPU_Unpickler

NUM_JOINTS = 23


def compute_mpjpe(pred_pose_69, gt_pose_69):
    """Both inputs are (T, 69). Returns (mpjpe_scalar, per_joint_array_23)."""
    T = pred_pose_69.shape[0]
    pred = pred_pose_69.view(T, NUM_JOINTS, 3)
    targ = gt_pose_69.view(T, NUM_JOINTS, 3)
    per_joint_t = (pred - targ).norm(dim=-1)        # (T, 23)
    per_joint = per_joint_t.mean(dim=0)             # (23,)
    mpjpe = per_joint.mean().item()                 # scalar
    return mpjpe, per_joint.cpu().numpy()


def align_pose_to_timestamps(net_pose, vel_ts, gt_ts):
    """Pick the GT samples that align with vel_ts (the model's downsampled grid).
    Assumes vel_ts is a subset of gt_ts (which is true here since both come from
    the same uniform 50Hz grid). Returns indices into gt_ts and the matched
    predicted pose tensor.
    """
    vel_ts_flat = vel_ts.reshape(-1)
    indices = torch.cat([torch.where(gt_ts == t)[0] for t in vel_ts_flat]).to(torch.long)
    return indices, net_pose[:vel_ts_flat.shape[0]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="experiments/nymeria_pose_predict")
    parser.add_argument("--dataconf", type=str, default="configs/nymeria_pose_predict.conf")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seqlen", type=int, default=1000)
    parser.add_argument("--savedir", type=str, default="./result/loss_result_pose_predict")
    args = parser.parse_args()
    print("\n", args, "\n")

    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.eval

    net_result_path = os.path.join(args.exp, 'net_output.pickle')
    if not os.path.isfile(net_result_path):
        raise FileNotFoundError(f"Unable to load the network result: {net_result_path}")
    with open(net_result_path, 'rb') as handle:
        inference_state_load = CPU_Unpickler(handle).load()

    os.makedirs(args.savedir, exist_ok=True)

    all_results = []
    for data_conf in dataset_conf.data_list:
        if isinstance(data_conf.data_drive, list) and len(data_conf.data_drive) == 0:
            data_drive = sorted(
                os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(data_conf["data_root"], "*.pkl"))
            )
        else:
            data_drive = list(data_conf.data_drive)

        for data_name in data_drive:
            if data_name not in inference_state_load:
                print(f"  skip {data_name} (not in net_output)")
                continue

            dataset = SeqDataset(
                data_conf.data_root, data_name, args.device,
                name=data_conf.name, duration=args.seqlen, step_size=args.seqlen,
                drop_last=False, conf=dataset_conf,
            )
            gt_pose = dataset.seq.data.get('pose', None)
            if gt_pose is None:
                print(f"  skip {data_name} (no GT pose in dataset)")
                continue

            inference_state = inference_state_load[data_name]
            net_pose = inference_state['net_pose']                # (T_down, 69)
            vel_ts   = inference_state['ts']                       # (T_down, 1) or (T_down,)
            if vel_ts.ndim == 2:
                vel_ts = vel_ts[:, 0]

            gt_ts = dataset.data['time']
            indices, pred_aligned = align_pose_to_timestamps(net_pose, vel_ts, gt_ts)
            gt_aligned = gt_pose[indices]                          # (T_down, 69)

            mpjpe, per_joint = compute_mpjpe(pred_aligned.cpu(), gt_aligned.cpu())

            result = {
                'name': data_name,
                'MPJPE_m': mpjpe,
                'MPJPE_cm': mpjpe * 100,
                'per_joint_cm': (per_joint * 100).round(3).tolist(),
                'duration_s': float(gt_ts[-1] - gt_ts[0]),
                'num_eval_frames': int(pred_aligned.shape[0]),
            }
            all_results.append(result)
            print(f"  {data_name}: MPJPE = {mpjpe*100:.2f} cm  ({pred_aligned.shape[0]} frames)")

    file_path = os.path.join(args.savedir, "result.json")
    with open(file_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    if all_results:
        mpjpe_vals = np.array([r['MPJPE_cm'] for r in all_results])
        per_joint_stack = np.array([r['per_joint_cm'] for r in all_results])
        print()
        print("=" * 60)
        print(f"{len(all_results)} sequences")
        print(f"MPJPE: mean {mpjpe_vals.mean():.2f} cm, "
              f"median {np.median(mpjpe_vals):.2f} cm, "
              f"max {mpjpe_vals.max():.2f} cm")
        print()
        per_joint_mean = per_joint_stack.mean(axis=0)
        print("Per-joint error (cm, averaged over sequences):")
        for j in range(NUM_JOINTS):
            print(f"  joint {j:2d}: {per_joint_mean[j]:6.2f}")
    print(f"\nsaved {file_path}")
