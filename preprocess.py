#!/usr/bin/env python3
"""Nymeria preprocessing -> CPF-aligned world-frame pickles.

Output schema:
    <seq>/recording_head/{data/motion.vrs, mps/slam/{closed_loop_trajectory.csv, online_calibration.jsonl}}
    -> <out_dir>/<seq>_cpfbody_<hz>hz.pkl
"""
import argparse
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from projectaria_tools.core import data_provider
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm


# ---------- native-rate stream loaders ----------

def load_imu_calibrations(calib_path):
    """Return {'imu-right': cal, 'imu-left': cal} or {None, None} if missing."""
    out = {'imu-right': None, 'imu-left': None}
    if not calib_path.exists():
        return out
    with open(calib_path) as f:
        for line in f:
            for ic in json.loads(line).get('ImuCalibrations', []):
                if ic['Label'] not in out:
                    continue
                qw, (qx, qy, qz) = ic['T_Device_Imu']['UnitQuaternion'][0], ic['T_Device_Imu']['UnitQuaternion'][1]
                out[ic['Label']] = {
                    'R': Rotation.from_quat([qx, qy, qz, qw]).as_matrix(),
                    'gyro_bias': np.array(ic['Gyroscope']['Bias']['Offset'], dtype=np.float64),
                    'accel_bias': np.array(ic['Accelerometer']['Bias']['Offset'], dtype=np.float64),
                }
    return out


def load_imu(provider, label, calib):
    """Return (timestamps_s, accel, gyro) calibrated into device frame at native rate."""
    sid = provider.get_stream_id_from_label(label)
    n = provider.get_num_data(sid)
    ts = np.empty(n, np.float64)
    accel = np.empty((n, 3), np.float64)
    gyro = np.empty((n, 3), np.float64)
    for i in tqdm(range(n), desc=f'load {label}', leave=False):
        s = provider.get_imu_data_by_index(sid, i)
        ts[i] = s.capture_timestamp_ns * 1e-9
        accel[i] = s.accel_msec2
        gyro[i] = s.gyro_radsec
    if calib is not None:
        # Right-multiply by R.T applies R to each column-vector row: v_dev = R @ (raw - bias)
        accel = (accel - calib['accel_bias']) @ calib['R'].T
        gyro = (gyro - calib['gyro_bias']) @ calib['R'].T
    return ts, accel, gyro


def load_mag(provider):
    sid = provider.get_stream_id_from_label('mag0')
    cal = provider.get_sensor_calibration(sid).magnetometer_calibration()
    n = provider.get_num_data(sid)
    ts = np.empty(n, np.float64)
    mag = np.empty((n, 3), np.float64)
    for i in range(n):
        s = provider.get_magnetometer_data_by_index(sid, i)
        ts[i] = s.capture_timestamp_ns * 1e-9
        mag[i] = cal.raw_to_rectified(np.array(s.mag_tesla).reshape(3, 1)).ravel()
    return ts, mag * 1e6  # tesla -> microtesla


def load_baro(provider):
    sid = provider.get_stream_id_from_label('baro0')
    cal = provider.get_sensor_calibration(sid).barometer_calibration()
    n = provider.get_num_data(sid)
    ts = np.empty(n, np.float64)
    pressure_kpa = np.empty(n, np.float64)
    temperature = np.empty(n, np.float64)
    for i in range(n):
        s = provider.get_barometer_data_by_index(sid, i)
        ts[i] = s.capture_timestamp_ns * 1e-9
        pressure_kpa[i] = cal.raw_to_rectified(s.pressure) * 1e-3
        temperature[i] = s.temperature
    altitude_m = 44330.0 * (1.0 - (pressure_kpa / 101.325) ** (1 / 5.255))
    return ts, pressure_kpa, altitude_m, temperature


def load_body_poses(seq_path, provider, t_grid_abs):
    """Sample Xsens body poses onto the device-time grid, aligned into Aria SLAM world.

    Returns (body_poses, valid_mask): body_poses is (n, J, 4, 4); NaN-filled outside the
    body's valid timespan. Returns (None, None) if body data or nymeria is missing.
    """
    if not (seq_path / 'body' / 'xdata.npz').exists():
        print(f"  no body/xdata.npz under {seq_path}; skipping body poses")
        return None, None
    try:
        from nymeria.data_provider import NymeriaDataProvider
        from nymeria.definitions import BodyModel
        from nymeria.xsens_constants import XSensConstants
        from projectaria_tools.core.sensor_data import TimeDomain
        nymeria_dp = NymeriaDataProvider(sequence_rootdir=seq_path, load_observer=False,
                                         load_wrist=False, load_bbox=False, body_model=BodyModel.MOMENTUM)
    except Exception as e:
        print(f"  body loading unavailable ({e}); skipping body poses")
        return None, None

    body_dp = nymeria_dp.body_dp
    head_idx, J = XSensConstants.part_names.index("Head"), XSensConstants.num_parts
    xs_q  = body_dp.xsens_data[XSensConstants.k_part_qWXYZ]
    xs_t  = body_dp.xsens_data[XSensConstants.k_part_tXYZ]
    xs_us = body_dp.xsens_data[XSensConstants.k_timestamps_us]

    timecodes_ns = np.array([provider.convert_from_device_time_to_timecode_ns(int(t * 1e9)) for t in t_grid_abs])
    t_lo, t_hi = nymeria_dp.timespan_ns
    valid_mask = (timecodes_ns >= t_lo) & (timecodes_ns <= t_hi)
    if not valid_mask.any():
        print("  grid does not overlap body data timespan; skipping body poses")
        return None, None

    # For each grid sample, realign XSens-world into Aria-world via the head joint and
    # express every XSens joint pose in Aria world:
    #     T_aria_xsensworld = T_aria_ariahead @ T_ariahead_xsenshead @ T_xsenshead_xsensworld
    #     body_poses[i, j]  = T_aria_xsensworld @ T_xsensworld_xsensjoint_j
    body_poses = np.full((len(t_grid_abs), J, 4, 4), np.nan, dtype=np.float64)
    for i in tqdm(np.where(valid_mask)[0], desc='load body', leave=False):
        t_ns = int(timecodes_ns[i])
        aria_head_pose          = nymeria_dp.recording_head.get_pose(t_ns, TimeDomain.TIME_CODE)[0].transform_world_device.to_matrix()
        xsens_head_to_aria_head = nymeria_dp.T_Hd_Hx(t_ns).to_matrix()                  # rigid handeye offset
        idx = int(np.argmin(np.abs(xs_us - t_ns / 1000)))
        joints_in_xsens_world   = body_dp.qt_to_se3(xs_q[idx], xs_t[idx])               # list of SE3, per joint
        xsens_world_to_aria     = aria_head_pose @ xsens_head_to_aria_head @ joints_in_xsens_world[head_idx].inverse().to_matrix()
        body_poses[i] = xsens_world_to_aria @ np.stack([T.to_matrix() for T in joints_in_xsens_world])
    return body_poses, valid_mask


# ---------- interpolation helpers ----------

def interp_cols(t_dst, t_src, x_src):
    """Per-column linear interp; works for (N,) and (N,K)."""
    if x_src.ndim == 1:
        return np.interp(t_dst, t_src, x_src)
    out = np.empty((len(t_dst), x_src.shape[1]), dtype=x_src.dtype)
    for k in range(x_src.shape[1]):
        out[:, k] = np.interp(t_dst, t_src, x_src[:, k])
    return out


def quat_wxyz_to_xyzw(q):
    return np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)


# ---------- main per-sequence pipeline ----------

def process_sequence(seq_path: Path, out_dir: Path, skip_seconds: float = 30.0, sample_hz: int = 200):
    out_path = out_dir / f"{seq_path.name}_cpfbody_{sample_hz}hz.pkl"
    if out_path.exists():
        print(f"skip (exists): {out_path}")
        return out_path

    vrs = seq_path / 'recording_head' / 'data' / 'motion.vrs'
    traj_csv = seq_path / 'recording_head' / 'mps' / 'slam' / 'closed_loop_trajectory.csv'
    calib_jsonl = seq_path / 'recording_head' / 'mps' / 'slam' / 'online_calibration.jsonl'
    if not vrs.exists() or not traj_csv.exists():
        print(f"missing required files in {seq_path}")
        return None

    provider = data_provider.create_vrs_data_provider(str(vrs))
    R_dev_cpf_frames = provider.get_device_calibration().get_transform_device_cpf().to_matrix()[:3, :3] # map IMU to CPF
    R_dev_cpf_vec = R_dev_cpf_frames.T  # vector transform = transpose of frame transform

    cals = load_imu_calibrations(calib_jsonl)
    tR, accR_dev, gyrR_dev = load_imu(provider, 'imu-right', cals['imu-right'])
    tL, accL_dev, gyrL_dev = load_imu(provider, 'imu-left', cals['imu-left'])
    tM, mag = load_mag(provider)
    tB, pressure, altitude, temperature = load_baro(provider)

    df = pd.read_csv(traj_csv)
    t_traj = df['tracking_timestamp_us'].to_numpy() / 1e6
    order = np.argsort(t_traj)
    t_traj = t_traj[order]
    q_traj_xyzw = quat_wxyz_to_xyzw(df[['qw_world_device', 'qx_world_device',
                                        'qy_world_device', 'qz_world_device']].to_numpy()[order])
    pos_world = df[['tx_world_device', 'ty_world_device', 'tz_world_device']].to_numpy()[order]
    vel_cols = [f'device_linear_velocity_{a}_device' for a in 'xyz']
    if all(c in df.columns for c in vel_cols):
        vel_dev = df[vel_cols].to_numpy()[order]
    else:
        # Fall back to finite differences in world frame, rotated into device frame
        vw = np.gradient(pos_world, axis=0) / np.gradient(t_traj)[:, None]
        R_dw = Rotation.from_quat(q_traj_xyzw).as_matrix()
        vel_dev = np.einsum('nji,nj->ni', R_dw, vw)  # R.T @ v per sample

    # ---- CPF-aligned world frame at reference time (first IMU + skip) ----
    # find reference time after skip_seconds (e.g., 30)
    t_ref = tR[0] + skip_seconds
    ref_idx = int(np.argmin(np.abs(t_traj - t_ref)))
    reference_time = float(t_traj[ref_idx])

    # find head orientation at reference time
    R_dev_world_ref = Rotation.from_quat(q_traj_xyzw[ref_idx]).as_matrix()
    R_cpf_world_ref = R_dev_world_ref @ R_dev_cpf_frames

    # forward = horizontal direction the wearer was facing at reference time
    fwd = R_cpf_world_ref[:, 2].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd) + 1e-8

    # up = gravity; left = perpendicular to both (right-hand rule)
    up   = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, fwd)
    left /= np.linalg.norm(left)

    # stack axes into the cpfworld -> SLAM-world rotation (use .T for the reverse)
    R_world_to_cpfworld = np.column_stack([left, up, fwd])

    # origin = wearer's position at reference time
    reference_position = pos_world[ref_idx].astype(np.float64)

    # ---- Build single uniform target grid; t=0 anchored at first right-IMU sample post-skip ----
    start_idx = int(np.searchsorted(tR, t_ref, side='left'))
    t0_abs = float(tR[start_idx])
    t_end_abs = min(tR[-1], tL[-1], tM[-1], tB[-1], t_traj[-1])
    t_grid_zero = np.arange(0.0, t_end_abs - t0_abs, 1.0 / sample_hz)
    t_grid_abs = t_grid_zero + t0_abs
    n = len(t_grid_zero)
    print(f"{seq_path.name}: {n} samples @ {sample_hz} Hz")

    # ---- Single-pass interpolation of every native stream onto the uniform grid ----
    accR_dev_g = interp_cols(t_grid_abs, tR, accR_dev)
    gyrR_dev_g = interp_cols(t_grid_abs, tR, gyrR_dev)
    accL_dev_g = interp_cols(t_grid_abs, tL, accL_dev)
    gyrL_dev_g = interp_cols(t_grid_abs, tL, gyrL_dev)
    mag_g = interp_cols(t_grid_abs, tM, mag)
    pressure_g = interp_cols(t_grid_abs, tB, pressure)
    altitude_g = interp_cols(t_grid_abs, tB, altitude)
    temperature_g = interp_cols(t_grid_abs, tB, temperature)
    pos_world_g = interp_cols(t_grid_abs, t_traj, pos_world)
    vel_dev_g = interp_cols(t_grid_abs, t_traj, vel_dev)
    q_dev_world_g = Slerp(t_traj, Rotation.from_quat(q_traj_xyzw))(np.clip(t_grid_abs, t_traj[0], t_traj[-1])).as_quat()

    # ---- Vectorized rotations into CPF / CPF-world ----
    # Vectors: v @ R_dev_cpf_vec.T == R_dev_cpf_vec @ v  (for rows treated as column vectors)
    accR_cpf = accR_dev_g @ R_dev_cpf_vec.T
    gyrR_cpf = gyrR_dev_g @ R_dev_cpf_vec.T
    accL_cpf = accL_dev_g @ R_dev_cpf_vec.T
    gyrL_cpf = gyrL_dev_g @ R_dev_cpf_vec.T
    vel_cpf = vel_dev_g @ R_dev_cpf_vec.T

    R_dev_world_g = Rotation.from_quat(q_dev_world_g).as_matrix()         # (n,3,3)
    R_cpf_world_g = R_dev_world_g @ R_dev_cpf_frames                       # (n,3,3)
    R_cpf_cpfworld_g = R_world_to_cpfworld.T @ R_cpf_world_g               # (n,3,3)
    quat_cpfworld = Rotation.from_matrix(R_cpf_cpfworld_g).as_quat()       # xyzw
    pos_cpfworld = (pos_world_g - reference_position) @ R_world_to_cpfworld

    # ---- Xsens body poses: load at grid times and transform world -> cpfworld ----
    body_poses_world, body_mask = load_body_poses(seq_path, provider, t_grid_abs)
    if body_poses_world is not None:
        T_cw = np.eye(4)
        T_cw[:3, :3] = R_world_to_cpfworld.T
        T_cw[:3, 3]  = -R_world_to_cpfworld.T @ reference_position
        body_poses_cpfworld = T_cw @ body_poses_world  # broadcasts: (n, J, 4, 4)
    else:
        body_poses_cpfworld = None

    # ---- Validation: integrate body-frame velocity in CPF-world, compare to position ----
    vel_cpfworld = np.einsum('nij,nj->ni', R_cpf_cpfworld_g, vel_cpf)
    dt = np.diff(t_grid_zero)
    integrated = np.empty_like(pos_cpfworld)
    integrated[0] = pos_cpfworld[0]
    integrated[1:] = pos_cpfworld[0] + np.cumsum(vel_cpfworld[:-1] * dt[:, None], axis=0)
    err = np.linalg.norm(integrated - pos_cpfworld, axis=1)
    total_dist = float(np.sum(np.linalg.norm(np.diff(pos_cpfworld, axis=0), axis=1)))
    duration = float(t_grid_zero[-1] - t_grid_zero[0])
    validation = {
        'max_error': float(err.max()),
        'mean_error': float(err.mean()),
        'final_error': float(err[-1]),
        'drift_percentage': float(err[-1] / total_dist * 100) if total_dist > 0 else 0.0,
        'total_distance': total_dist,
        'duration': duration,
    }
    print(f"  drift: {validation['drift_percentage']:.3f}%  max_err: {validation['max_error']:.3f}m")

    # ---- Pack to target schema ----
    f32 = lambda a: np.asarray(a, np.float32)
    tens = lambda a: torch.tensor(np.asarray(a, np.float32))
    output = {
        'sequence_name': seq_path.name,
        'duration_seconds': duration,
        'num_samples': int(n),
        'imu_sampling_rate_hz': float(sample_hz),
        'coordinate_frames': {
            'imu_right': 'CPF (body frame)',
            'imu_left_cpf': 'CPF (body frame) - same as right',
            'imu_left_body': 'Left IMU own body frame (after calibration, before CPF transform)',
            'velocity': 'CPF (body frame)',
            'orientation': 'CPF→CPF-world quaternions',
            'position': 'CPF-world frame',
        },
        'reference_time': reference_time,
        'reference_position': f32(reference_position),
        'R_device_to_CPF_frames': f32(R_dev_cpf_frames),
        'R_device_to_CPF_vectors': f32(R_dev_cpf_vec),
        'R_world_to_cpfworld': f32(R_world_to_cpfworld),
        'imu_data': {
            'time': tens(t_grid_zero),
            'accel': tens(accR_cpf),
            'gyro': tens(gyrR_cpf),
            'accel_left_cpf': tens(accL_cpf),
            'gyro_left_cpf': tens(gyrL_cpf),
            'accel_left_body': tens(accL_dev_g),
            'gyro_left_body': tens(gyrL_dev_g),
            'mag': tens(mag_g),
            'pressure': tens(pressure_g),
            'altitude': tens(altitude_g),
            'temperature': tens(temperature_g),
        },
        'gt_data': {
            'time': tens(t_grid_zero),
            'orientation': tens(quat_cpfworld),
            'position': tens(pos_cpfworld),
            'velocity': tens(vel_cpf),
        },
        'validation': validation,
    }
    if body_poses_cpfworld is not None:
        output['gt_data']['xsens_pose'] = tens(body_poses_cpfworld)            # (n, J, 4, 4)
        output['gt_data']['xsens_valid_mask'] = torch.tensor(body_mask, dtype=torch.bool)
        output['coordinate_frames']['xsens_pose'] = 'CPF-world frame (per-joint 4x4)'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"saved: {out_path}")
    return out_path


def _job(args):
    return process_sequence(*args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path', help='Single Nymeria sequence directory or a parent containing many.')
    p.add_argument('-o', '--output-dir', default='output_final')
    p.add_argument('--skip-seconds', type=float, default=30.0)
    p.add_argument('--sample-hz', type=int, default=200)
    p.add_argument('--max', type=int)
    p.add_argument('--parallel', type=int, default=1)
    args = p.parse_args()

    in_path = Path(args.path)
    out_dir = Path(args.output_dir)
    if (in_path / 'recording_head' / 'data' / 'motion.vrs').exists():
        seqs = [in_path]
    else:
        seqs = sorted(d for d in in_path.iterdir() if d.is_dir() and not d.name.startswith('.'))
    if args.max:
        seqs = seqs[:args.max]
    print(f"processing {len(seqs)} sequence(s) -> {out_dir}")

    jobs = [(s, out_dir, args.skip_seconds, args.sample_hz) for s in seqs]
    if args.parallel > 1 and len(jobs) > 1:
        with Pool(args.parallel) as pool:
            for _ in tqdm(pool.imap(_job, jobs), total=len(jobs)):
                pass
    else:
        for j in jobs:
            _job(j)


if __name__ == '__main__':
    main()
