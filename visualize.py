import argparse
import os
import pickle

import matplotlib.pyplot as plt
import pypose as pp
import torch
from pyhocon import ConfigFactory

from datasets.nymeria_dataset import Nymeria


def integrate_body_velocity(vel_body, rot_quat, ts, init_pos):
    """Body-frame velocity -> world-frame trajectory."""
    vel_world = pp.SO3(rot_quat) @ vel_body
    dt = torch.diff(ts, prepend=ts[:1])
    return init_pos + torch.cumsum(vel_world * dt[:, None], dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="nymeria_small.conf")
    parser.add_argument("--seq", type=str, default=None, help="sequence name (default: first one in pickle)")
    parser.add_argument("--out", type=str, default="trajectory.png")
    args = parser.parse_args()

    conf = ConfigFactory.parse_file(args.config)

    with open(os.path.join(conf.general.exp_dir, "net_output.pickle"), "rb") as f:
        net_out = pickle.load(f)
    seq_name = args.seq if args.seq is not None else next(iter(net_out))
    print(f"Visualizing: {seq_name}")

    pred = net_out[seq_name]
    pred_vel = pred["net_vel"].cpu().float()
    pred_ts = pred["ts"].squeeze(-1).cpu().float()

    data_root = conf.eval.data_list[0]["data_root"]
    seq = Nymeria(data_root=data_root, data_name=seq_name)
    gt_time = seq.data["time"].float()
    gt_pos = seq.data["gt_translation"]
    gt_quat = seq.data["gt_orientation"].tensor()
    gt_vel_body = seq.data["velocity"]

    idx = torch.searchsorted(gt_time, pred_ts).clamp(max=len(gt_time) - 1)
    pred_traj = integrate_body_velocity(pred_vel, gt_quat[idx], pred_ts, gt_pos[idx[0]])
    gt_vel_at_pred = gt_vel_body[idx]
    t_rel = pred_ts - pred_ts[0]

    fig = plt.figure(figsize=(9, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.35)

    ax_traj = fig.add_subplot(gs[0])
    ax_traj.plot(gt_pos[:, 0], gt_pos[:, 2], label="GT", linewidth=2)
    ax_traj.plot(pred_traj[:, 0], pred_traj[:, 2], label="Pred", linewidth=2, alpha=0.8)
    ax_traj.scatter(gt_pos[0, 0], gt_pos[0, 2], color="green", s=80, label="start", zorder=5)
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("z [m]")
    ax_traj.set_title(f"Trajectory  —  {seq_name}", fontsize=10)
    ax_traj.legend(loc="best")
    ax_traj.grid(alpha=0.3)

    axes_vel = [fig.add_subplot(gs[i + 1]) for i in range(3)]
    for ax, comp, label in zip(axes_vel, range(3), ["x", "y", "z"]):
        ax.plot(t_rel, gt_vel_at_pred[:, comp], label="GT", linewidth=1.8)
        ax.plot(t_rel, pred_vel[:, comp], label="Pred", linewidth=1.4, alpha=0.85)
        ax.set_ylabel(f"v_{label} [m/s]")
        ax.grid(alpha=0.3)
        if ax is axes_vel[0]:
            ax.legend(loc="upper right")
            ax.set_title("Body-frame velocity", fontsize=10)
    axes_vel[-1].set_xlabel("time [s]")
    for ax in axes_vel[:-1]:
        ax.tick_params(labelbottom=False)

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.out}")
