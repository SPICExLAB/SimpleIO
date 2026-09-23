"""
Render an mp4 comparing predicted pose (left) vs GT pose (right) for the
first N seconds of a sequence, using a trained PoseNetMotionwithRot.

Loads the trained model from a checkpoint, runs inference on the sequence,
interpolates the (downsampled) predictions back to 50 Hz, and animates the
two skeletons side-by-side.
"""

import argparse
import os
import pickle

import numpy as np
import torch
import torch.utils.data as Data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pyhocon import ConfigFactory

from datasets.dataset_motion import SeqeuncesMotionDataset
from datasets.dataset_utils import collate_fcs
from model.code import PoseNetMotionwithRot
from utils import move_to, so3_log


# Xsens MVN 23-segment hierarchy: (child, parent). 0 = pelvis, 6 = head.
BONES = [
    (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5),     # spine + head
    (7, 4), (8, 7), (9, 8), (10, 9),                    # right arm
    (11, 4), (12, 11), (13, 12), (14, 13),              # left arm
    (15, 0), (16, 15), (17, 16), (18, 17),              # right leg
    (19, 0), (20, 19), (21, 20), (22, 21),              # left leg
]
POSE_HZ = 50
NUM_JOINTS = 23


def remap(p):
    """(x_left, y_up, z_fwd) -> matplotlib display (x, z_fwd, y_up)."""
    return p[..., 0], p[..., 2], p[..., 1]


def run_inference(net, seq_name, conf, device):
    """Run model on `seq_name`, return (pred_pose_69, pred_timestamps) at the
    model's downsampled rate, concatenated across windows."""
    ds_conf = conf.eval
    ds_conf["mode"] = "infevaluate"
    ds_conf["exp_dir"] = conf.general.exp_dir
    ds_conf.data_list[0]["window_size"] = 250
    ds_conf.data_list[0]["step_size"] = 250
    dataset = SeqeuncesMotionDataset(
        data_set_config=ds_conf,
        data_path=seq_name,
        data_root=ds_conf.data_list[0]["data_root"],
    )
    loader = Data.DataLoader(dataset, batch_size=1, shuffle=False,
                             collate_fn=collate_fcs["motion"], drop_last=False)
    pred_poses, pred_ts = [], []
    with torch.no_grad():
        for data, _, label in loader:
            data, label = move_to([data, label], device)
            rot = so3_log(label["gt_rot"][:, :-1, :])
            out = net(data, rot)                                   # (1, T_d, 69)
            ts = net.get_label(data["ts"][..., None])[..., 0]      # (1, T_d)
            pred_poses.append(out["net_pose"][0].cpu())
            pred_ts.append(ts[0].cpu())
    pred_poses = torch.cat(pred_poses, dim=0).numpy()              # (T_total, 69)
    pred_ts = torch.cat(pred_ts, dim=0).numpy()                    # (T_total,)
    return pred_poses, pred_ts


def interp_pose_to(times, src_t, src_pose):
    """Linearly interpolate src_pose (T_src, 69) along src_t (T_src,) onto
    `times` (T_dst,). Returns (T_dst, 69)."""
    T_dst = times.shape[0]
    out = np.empty((T_dst, src_pose.shape[1]), dtype=src_pose.dtype)
    for c in range(src_pose.shape[1]):
        out[:, c] = np.interp(times, src_t, src_pose[:, c])
    return out


def gt_pose_for_sequence(seq_name, data_root):
    """Load GT root-translated joint positions for the sequence.
    Timestamps are kept in absolute SLAM time so they align with the
    pred_ts coming back from the model (also in absolute time).
    """
    pkl = os.path.join(data_root, seq_name + ".pkl")
    with open(pkl, "rb") as f:
        raw = pickle.load(f)
    xp = raw["gt_data"]["xsens_pose"]                              # (N, 23, 4, 4)
    joint_pos = xp[..., :3, 3]                                      # (N, 23, 3)
    root_pos = joint_pos[:, 0:1, :]
    gt = (joint_pos - root_pos).reshape(joint_pos.shape[0], -1).float().numpy()
    gt_t = raw["imu_data"]["time"].cpu().numpy()                   # absolute, not zeroed
    return gt, gt_t


def render_animation(out_path, pred_poses_50hz, gt_poses, duration_s,
                     pose_hz=POSE_HZ, render_stride=2, seq_name=""):
    """Both inputs are (T, 69) on the same 50Hz grid, root-translated."""
    n_max = min(int(pose_hz * duration_s), pred_poses_50hz.shape[0], gt_poses.shape[0])
    P_pred = pred_poses_50hz[:n_max:render_stride].reshape(-1, NUM_JOINTS, 3)
    P_gt = gt_poses[:n_max:render_stride].reshape(-1, NUM_JOINTS, 3)
    T = P_pred.shape[0]
    fps = pose_hz // render_stride
    print(f"Rendering {T} frames ({T / fps:.1f}s of video at {fps} fps)")

    fig = plt.figure(figsize=(12, 6), dpi=80)
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    titles = ["predicted", "ground truth"]
    LIM = 1.2
    for ax, title in zip(axes, titles):
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
        ax.set_xlabel("X"); ax.set_ylabel("Z (fwd)"); ax.set_zlabel("Y (up)")
        ax.set_title(title)
        ax.view_init(elev=10, azim=-70)

    scats, lines = [], []
    for ax, color in zip(axes, ("tab:orange", "tab:red")):
        scats.append(ax.scatter([], [], [], s=18, c=color))
        lines.append([ax.plot([], [], [], "-", lw=2, color="tab:blue")[0] for _ in BONES])

    suptitle = fig.suptitle(seq_name, fontsize=9)
    time_text = fig.text(0.5, 0.93, "", ha="center", fontsize=11)

    def update(i):
        for panel_idx, P in enumerate((P_pred, P_gt)):
            pts = P[i]
            xs, ys, zs = remap(pts)
            scats[panel_idx]._offsets3d = (xs, ys, zs)
            for ln, (c, p) in zip(lines[panel_idx], BONES):
                seg = np.stack([pts[p], pts[c]], axis=0)
                xs_b, ys_b, zs_b = remap(seg)
                ln.set_data(xs_b, ys_b)
                ln.set_3d_properties(zs_b)
        t_actual = i * render_stride
        time_text.set_text(f"t = {t_actual / pose_hz:.2f}s   frame {t_actual}/{n_max}")
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    print(f"Saving to {out_path} ...")
    anim.save(out_path, writer=writer)
    print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/nymeria_pose_predict.conf")
    parser.add_argument("--ckpt", type=str,
                        default="experiments/nymeria_pose_predict/ckpt/best_model.ckpt")
    parser.add_argument("--seq", type=str,
                        default="20230607_s0_james_johnson_act3_ifj2gc_cpfbody_imu50hz_pose50hz")
    parser.add_argument("--duration", type=int, default=300, help="seconds to animate")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", type=str, default="predicted_vs_gt_pose.mp4")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    conf = ConfigFactory.parse_file(args.config)

    # Load model
    print(f"Loading {args.ckpt}")
    net = PoseNetMotionwithRot(conf.train).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    print(f"  loaded epoch {ckpt.get('epoch', '?')}")

    # Predict
    print(f"Running inference on {args.seq}")
    pred_poses, pred_ts = run_inference(net, args.seq, conf, device)
    print(f"  predicted {pred_poses.shape[0]} downsampled frames "
          f"(t = {pred_ts[0]:.2f}..{pred_ts[-1]:.2f}s)")

    # GT
    data_root = conf.eval.data_list[0]["data_root"]
    gt_poses, gt_ts = gt_pose_for_sequence(args.seq, data_root)
    print(f"  GT has {gt_poses.shape[0]} frames at 50Hz")

    # Interpolate predicted onto GT timestamps so the animation is at 50Hz
    pred_50hz = interp_pose_to(gt_ts, pred_ts, pred_poses)
    print(f"  interpolated pred to {pred_50hz.shape[0]} frames at 50Hz")

    render_animation(args.out, pred_50hz, gt_poses, args.duration,
                     pose_hz=POSE_HZ, render_stride=2, seq_name=args.seq)
