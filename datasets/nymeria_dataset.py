import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pypose as pp

from datasets.dataset import Sequence


class Nymeria(Sequence):
    def __init__(
        self, 
        data_root: str,
        data_name: str,
        remove_g: bool = False,
        target_hz: int = 50,
        gravity: float = 9.81,
        **kwargs
    ):
        """
        Nymeria dataset class.

        time: (N,)
        acc: (N, 3)
        gyro: (N, 3)
        gt_orientation: (N, 4)
        gt_translation: (N, 3)
        velocity: (N, 3)
        """

        super().__init__()

        # Load pickle file 
        pkl_file = os.path.join(data_root, data_name + '.pkl')
        with open(pkl_file, 'rb') as f:
            raw = pickle.load(f)

        # Optionally downsample
        data_sampling_rate_hz = raw['imu_sampling_rate_hz']
        downsampling_rate = max(1, round(data_sampling_rate_hz / target_hz))
        for key in ["time", "accel", "gyro", "orientation", "position", "velocity"]:
            if key in raw.get('imu_data', {}):
                raw['imu_data'][key] = raw['imu_data'][key][::downsampling_rate]
            if key in raw.get('gt_data', {}):
                raw['gt_data'][key] = raw['gt_data'][key][::downsampling_rate]

        # Package IMU data 
        self.data = {}
        self.data['time'] = raw['imu_data']['time']
        self.data['acc'] = raw['imu_data']['accel']
        self.data['gyro'] = raw['imu_data']['gyro']

        # Remove gravity from accelerometer
        gravity_vec = torch.tensor([0, -gravity, 0], dtype=self.data['acc'].dtype)
        gravity_vec = gravity_vec.expand(self.data['acc'].shape[0], -1)

        # Compute dt
        time = self.data['time']
        dt = torch.zeros_like(time)
        dt[1:] = time[1:] - time[:-1]
        dt[0] = dt[1]
        self.data['dt'] = dt        

        if remove_g:
            self.data['acc'] = self.data['acc'] + self.data["gt_orientation"].Inv() @ gravity_vec

        # Package ground truth data 
        orientation = raw['gt_data']['orientation']
        self.data['gt_orientation'] = pp.SO3(orientation) if orientation.shape[-1] == 4 else orientation
        self.data['gt_translation'] = raw['gt_data']['position']
        self.data['velocity'] = raw['gt_data']['velocity'] # body-frame velocity

        # Mask (all valid)
        self.data['mask'] = torch.ones(len(time), dtype=torch.bool)        

    def get_length(self):
        return len(self.data['time'])


if __name__ == "__main__":
    data_root = "/data/datasets/Nymeria_Processed/test/"
    data_name = "20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz"

    dataset = Nymeria(
        data_root=data_root,
        data_name=data_name,
        target_hz=50,
    )

    print(f"\n=== Dataset Summary ===")
    print(f"Length: {dataset.get_length()} frames")
    print(f"\nData fields:")
    for k, v in dataset.data.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {type(v)}")