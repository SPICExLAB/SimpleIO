# `preprocess.py` — Nymeria preprocessing refresher

What this script does, why, and the coordinate-frame story that makes it confusing.

## What it does (one sentence)

Per Nymeria sequence, read raw multi-rate sensor data + SLAM trajectory, transform everything into useful coordinate frames, resample to a uniform 200 Hz grid, and write a `<seq>_cpfbody_200hz.pkl` that the dataloader consumes.

```
<seq>/recording_head/{data/motion.vrs, mps/slam/{closed_loop_trajectory.csv, online_calibration.jsonl}}
    -> <out_dir>/<seq>_cpfbody_200hz.pkl
```

Run with:
```
python preprocess.py <seq_dir_or_parent> -o <out_dir> --sample-hz 200 [--skip-seconds 30]
```

## Coordinate frames in play

Five frames. The trickiest concept is that there are **two body frames** (IMU chip, device) **and two world frames** (SLAM-world, cpfworld) — plus CPF (the head-anatomical body frame the model trains in).

| Frame | Type | Moves with wearer? | What lives in it |
|---|---|---|---|
| IMU-chip | body | yes | raw `accel_msec2`, `gyro_radsec` straight from sensor |
| Device | body | yes | Aria glasses' canonical body frame |
| **CPF** | body | yes | head-anatomical (X=left, Y=up, Z=forward through nose) |
| SLAM-world | world | no | trajectory CSV's positions and quaternions; Z-up |
| **cpfworld** | world | no | what we build; Y-up Z-forward, matches CPF convention |

The two frames in **bold** are what the dataloader uses. Everything else is plumbing.

## Two independent transformation chains

The script does **two** rotations, doing two different jobs on two different parts of the data:

```
IMU side (model inputs):
  IMU-chip frame  --[per-IMU calib R_Device_Imu]-->  device frame
  device frame    --[Aria's R_dev_cpf_vec]------->  CPF
  (then linear-interpolated onto the 200 Hz grid)

Trajectory side (model GT):
  SLAM-world position  --[subtract reference_position, then rotate]-->  cpfworld position
  SLAM-world orientation (R_dev_world)  --[compose with R_dev_cpf, then with R_world_to_cpfworld.T]-->  R_cpf_cpfworld
  (then Slerp/linearly interpolated onto the 200 Hz grid)
```

These two chains never interact. The IMU never enters any world frame; the trajectory never enters any body frame except CPF momentarily during composition.

## Why cpfworld exists (the conceptual reason)

SLAM-world is per-recording arbitrary: origin is wherever SLAM happened to start, yaw is wherever the wearer was facing then. Training against raw SLAM-world targets would force the model to learn that recordings starting at `(12, -4)` facing northeast are equivalent to recordings starting at `(0.3, 8)` facing west.

cpfworld erases that arbitrariness. Two specific choices:

1. **Origin = wearer's position at t = first_IMU + 30s.** Every recording starts at `(0, 0, 0)`.
2. **Axes = gravity-up (Y) and wearer's-horizontal-facing-at-t=30s (Z).** Every recording starts with the wearer facing along `+Z`.

Bonus: cpfworld uses Y-up Z-forward to match CPF's convention. This means at t=30s, `R_cpf_cpfworld` is the identity. The model never has to learn to undo a constant offset rotation.

(SLAM is gravity-aligned too — Z-up. We borrow its gravity direction when building cpfworld's Y axis.)

## How `R_world_to_cpfworld` is constructed

[preprocess.py:148-181](preprocess.py#L148-L181). Step-by-step:

1. Find the trajectory row closest to `t = first_IMU_timestamp + 30s`. Call its quaternion `q_ref` and its position `reference_position`.
2. Convert `q_ref` to a matrix `R_dev_world_ref` (maps device → SLAM-world).
3. Compose with `R_dev_cpf_frames` (Aria's device→CPF) to get `R_cpf_world_ref`. Its columns are CPF axes expressed in SLAM-world.
4. Take column 2 — CPF's +Z, the gaze direction — in SLAM-world coordinates. Project onto the horizontal plane (`fwd[2] = 0`) and renormalize. This is cpfworld's +Z (forward).
5. Set `up = (0, 0, 1)`, which in SLAM-world is the gravity-anti-parallel direction. This becomes cpfworld's +Y.
6. `left = up × fwd` gives a right-handed orthogonal X axis.
7. Stack `[left, up, fwd]` as columns → `R_world_to_cpfworld`.

The matrix's columns are cpfworld's basis vectors written in SLAM-world. It maps **cpfworld → SLAM-world**.

### Naming gotcha

`R_world_to_cpfworld` is **misnamed**: despite the name, it maps cpfworld → world. The transpose `R_world_to_cpfworld.T` is what goes SLAM-world → cpfworld. Inherited from the original 1127-line script; left as-is for compatibility.

## What `R_world_to_cpfworld` is used for

Three places in the script:

1. **Orientation** ([preprocess.py:214](preprocess.py#L214)): `R_cpf_cpfworld = R_world_to_cpfworld.T @ R_cpf_world`. Swaps the world-side of the rotation chain from SLAM-world to cpfworld.
2. **Position** ([preprocess.py:216](preprocess.py#L216)): `pos_cpfworld = (pos_world - reference_position) @ R_world_to_cpfworld`. Subtract the new origin, then re-express the displacement in cpfworld.
3. **Saved in the pickle** for downstream debugging or inverse mapping.

Note: row-vector `v @ M` is equivalent to column-vector `M.T @ v`, which is why position uses `@ R_world_to_cpfworld` without an explicit transpose. The math is the same.

## Orientation: what gets stored

`gt_data['orientation']` is `R_cpf_cpfworld` per timestamp, as `(qx, qy, qz, qw)`. That is:

> "Rotation that maps a vector expressed in CPF body frame to the same vector expressed in cpfworld."

It's body→world. At t=30s it's the identity by construction. As the wearer moves their head, it encodes the cumulative rotation away from that initial pose.

### Using it downstream

- Body-frame vector to world: `v_cpfworld = R_cpf_cpfworld @ v_cpf`.
- World-frame vector to body: `v_cpf = R_cpf_cpfworld.T @ v_cpfworld`.
- Gravity in body frame: `g_cpf = R_cpf_cpfworld.T @ (0, -9.81, 0)`. Used by the dataloader at [datasets/nymeria_dataset.py:67](datasets/nymeria_dataset.py#L67) when `remove_g: True`.

## Output schema

```
sequence_name: str
duration_seconds: float
num_samples: int                       # length of every time-series array
sampling_rate_hz: float                # 200
coordinate_frames: dict[str, str]      # documentation of which frame each field uses
reference_time: float                  # absolute SLAM time at t=0 of the output grid
reference_position: (3,) float32       # in SLAM-world; subtracted to put the wearer at origin
R_device_to_CPF_frames: (3,3) float32  # Aria T_device_cpf
R_device_to_CPF_vectors: (3,3) float32 # transpose of above (for rotating IMU vectors)
R_world_to_cpfworld: (3,3) float32     # cpfworld -> SLAM-world (see naming gotcha)

imu_data: dict of torch.float32 tensors
    time:             (N,)             # zeroed at 0, step = 1/sample_hz
    accel:            (N, 3)           # right IMU in CPF body frame
    gyro:             (N, 3)           # right IMU in CPF body frame
    accel_left_cpf:   (N, 3)           # left IMU in CPF
    gyro_left_cpf:    (N, 3)
    accel_left_body:  (N, 3)           # left IMU still in device frame (post-calibration)
    gyro_left_body:   (N, 3)
    mag:              (N, 3)           # microtesla
    pressure:         (N,)             # kPa
    altitude:         (N,)             # meters, derived from pressure via standard atmosphere
    temperature:      (N,)

gt_data: dict of torch.float32 tensors
    time:             (N,)
    orientation:      (N, 4)           # R_cpf_cpfworld as (qx, qy, qz, qw)
    position:         (N, 3)           # in cpfworld; starts at (0, 0, 0)
    velocity:         (N, 3)           # in CPF body frame (NOT cpfworld)

validation: dict
    max_error, mean_error, final_error, drift_percentage, total_distance, duration
```

Note: `gt_data['velocity']` is in CPF body, not cpfworld. To get cpfworld velocity, do `R_cpf_cpfworld @ velocity`.

## Resampling notes

Every native-rate stream is interpolated **once**, directly onto the uniform 200 Hz grid:

- Vectors and scalars: linear interpolation (`np.interp`).
- Quaternions (orientation only): **Slerp**, along the great-circle arc on the unit-quaternion sphere. Component-wise linear interp would chord across the sphere and not preserve constant angular velocity.

The grid spans the intersection of all five streams' time windows after the 30s skip, anchored so `t = 0` at the first right-IMU sample post-skip.

## Validation: the integration drift check

At the end of `process_sequence`, we rotate `velocity_cpf` into cpfworld using the per-sample `R_cpf_cpfworld`, integrate forward in time, and compare to `position_cpfworld`. A correct rotation chain produces ~0.06% drift over 220 m of walking (~14 cm final error). Anything significantly higher means a transposed matrix somewhere.

This is the cheapest sanity check that catches the most common bugs: wrong direction on a frame transform, missed transpose, wrong quaternion ordering.

## Common gotchas

- **Quaternion order:** Aria CSV is `(qw, qx, qy, qz)`. scipy is `(qx, qy, qz, qw)`. The script converts at [preprocess.py:103-104](preprocess.py#L103-L104).
- **R_device_to_CPF_frames vs vectors:** the "frames" version maps cpf-vector → device-vector (despite the name); the "vectors" version is its transpose and maps device-vector → cpf-vector. Used in different chain positions.
- **`R_world_to_cpfworld` naming:** as noted above, this matrix maps cpfworld → world, not world → cpfworld. Use `.T` to go the documented direction.
- **Velocity is body-frame:** the CSV reports `device_linear_velocity_*_device` in device frame, not world. We rotate it to CPF, not to cpfworld. Downstream consumers that want world velocity must apply `R_cpf_cpfworld`.
- **Gravity:** cpfworld is Y-up, so gravity is `(0, -9.81, 0)` in cpfworld, not `(0, 0, -9.81)`.

## File layout

```
preprocess.py
    load_imu_calibrations           # parses per-IMU R_Device_Imu and biases from JSONL
    load_imu                        # native-rate IMU stream; applies calibration into device frame
    load_mag                        # native-rate magnetometer; calibrated, in microtesla
    load_baro                       # native-rate barometer; pressure, derived altitude, temperature
    interp_cols, quat_wxyz_to_xyzw  # small helpers
    process_sequence                # the main pipeline (one sequence -> one pickle)
    main                            # CLI, sequence discovery, optional Pool parallelism
```

Total: ~300 lines. Replaces a 1127-line predecessor (`preprocess_nymeria_mag_baro_2imu_both.py`) and produces a structurally identical pickle with slightly better numerical accuracy (single-pass interpolation, proper Slerp for orientation).
