# Visualize predicted vs ground-truth velocity and trajectory per sequence.
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pyhocon import ConfigFactory
from datasets import SeqDataset

from utils import CPU_Unpickler, interp_xyz
from velocity_integrator import Velocity_Integrator, integrate_pos


def plot_sequence(out_path, ts, pred_vel, gt_vel, pred_traj, gt_traj):
    err = pred_traj - gt_traj
    pos_err = np.linalg.norm(err, axis=-1)
    ate = float(np.sqrt(np.mean(pos_err ** 2)))
    ate_v = float(np.sqrt(np.mean(err[:, 1] ** 2)))
    ate_h = float(np.sqrt(np.mean(np.linalg.norm(err[:, [0, 2]], axis=-1) ** 2)))
    final_err = float(np.linalg.norm(err[-1]))
    total_dist = float(np.linalg.norm(np.diff(gt_traj, axis=0), axis=-1).sum())
    drift_pct = 100.0 * final_err / total_dist if total_dist > 0 else float('nan')

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0])

    vel_axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for i, name in enumerate(['x', 'y', 'z']):
        vel_axes[i].plot(ts, gt_vel[:, i], label='gt', linewidth=1.0)
        vel_axes[i].plot(ts, pred_vel[:, i], label='pred', linewidth=1.0, alpha=0.8)
        vel_axes[i].set_ylabel(f'v{name} [m/s]')
        vel_axes[i].grid(True, alpha=0.3)
    vel_axes[0].legend(loc='upper right')
    vel_axes[-1].set_xlabel('time [s]')
    for ax in vel_axes[:-1]:
        ax.tick_params(labelbottom=False)

    traj_ax = fig.add_subplot(gs[:, 1])
    traj_ax.plot(gt_traj[:, 0], gt_traj[:, 2], label='gt', linewidth=1.0)
    traj_ax.plot(pred_traj[:, 0], pred_traj[:, 2], label='pred', linewidth=1.0, alpha=0.8)
    traj_ax.scatter(gt_traj[0, 0], gt_traj[0, 2], c='g', s=40, marker='o', label='start', zorder=5)
    traj_ax.scatter(gt_traj[-1, 0], gt_traj[-1, 2], c='r', s=40, marker='x', label='gt end', zorder=5)
    traj_ax.scatter(pred_traj[-1, 0], pred_traj[-1, 2], c='b', s=40, marker='x', label='pred end', zorder=5)
    traj_ax.set_xlabel('x [m]')
    traj_ax.set_ylabel('z [m]')
    traj_ax.set_aspect('equal', adjustable='datalim')
    traj_ax.grid(True, alpha=0.3)
    traj_ax.legend(loc='upper left')

    metrics = (
        f"ATE:    {ate:.2f} m\n"
        f"ATE_V:  {ate_v:.2f} m\n"
        f"ATE_H:  {ate_h:.2f} m\n"
        f"Drift:  {drift_pct:.2f}%"
    )
    traj_ax.text(0.98, 0.02, metrics, transform=traj_ax.transAxes,
                 ha='right', va='bottom', fontsize=14, family='monospace',
                 bbox=dict(facecolor='white', edgecolor='#bbbbbb', boxstyle='round,pad=0.5'))

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument("--exp", type=str, default="experiments/nymeria", help="Path for AirIO netoutput")
    parser.add_argument("--seqlen", type=int, default=1000, help="the length of the segment")
    parser.add_argument("--dataconf", type=str, default="configs/nymeria.conf", help="the configuration of the dataset")
    parser.add_argument("--savedir", type=str, default="./result/visualizations", help="Directory where the plots will be saved")

    args = parser.parse_args()
    print(("\n" * 3) + str(args) + ("\n" * 3))
    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.eval

    net_result_path = os.path.join(args.exp, 'net_output.pickle')
    if not os.path.isfile(net_result_path):
        raise FileNotFoundError(f"Unable to load the network result: {net_result_path}")
    with open(net_result_path, 'rb') as handle:
        inference_state_load = CPU_Unpickler(handle).load()

    folder = args.savedir
    os.makedirs(folder, exist_ok=True)

    for data_conf in dataset_conf.data_list:
        if isinstance(data_conf.data_drive, list) and len(data_conf.data_drive) == 0:
            data_drive = sorted(
                os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(data_conf["data_root"], "*.pkl"))
            )
        else:
            data_drive = list(data_conf.data_drive)

        for data_name in data_drive:
            print(f"\n=== {data_name} ===")
            dataset = SeqDataset(
                data_conf.data_root, data_name, args.device,
                name=data_conf.name, duration=args.seqlen, step_size=args.seqlen,
                drop_last=False, conf=dataset_conf,
            )
            init = dataset.get_init_value()

            inference_state = inference_state_load[data_name]
            gt_ts = dataset.data['time']
            vel_ts = inference_state['ts']

            if "coordinate" in dataset_conf.keys() and dataset_conf["coordinate"] == "body_coord":
                rotation = dataset.data['gt_orientation']
                net_vel = interp_xyz(gt_ts, vel_ts[:, 0], inference_state['net_vel']).to(rotation.dtype)
                net_vel = rotation * net_vel
            else:
                net_vel = interp_xyz(gt_ts, vel_ts[:, 0], inference_state['net_vel']).to(dataset.data['velocity'].dtype)

            dt = gt_ts[1:] - gt_ts[:-1]
            data_inte = {"vel": net_vel, "dt": dt}
            integrator_vel = Velocity_Integrator(init['pos']).to(args.device)
            inf_outstate = integrate_pos(integrator_vel, data_inte, init, dataset, device=args.device)

            ts = gt_ts.cpu().numpy()
            pred_vel = net_vel.detach().cpu().numpy()
            gt_vel = dataset.data['velocity'].cpu().numpy()
            pred_traj = inf_outstate['poses'][0].detach().cpu().numpy()
            gt_traj = inf_outstate['poses_gt'][1:].cpu().numpy()

            out_path = os.path.join(folder, f"{data_name}.png")
            plot_sequence(out_path, ts, pred_vel, gt_vel, pred_traj, gt_traj)
            print(f"  saved {os.path.basename(out_path)}")
