"""Export a trained PoseNetMotionwithRot checkpoint to ONNX."""

import argparse
import torch
import torch.nn as nn
from pyhocon import ConfigFactory

from model.code import PoseNetMotionwithRot


class PoseNetONNX(nn.Module):
    """Flat-tensor wrapper for ONNX export. Takes (acc, gyro, rot), returns pose."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, acc, gyro, rot):
        return self.net({"acc": acc, "gyro": gyro}, rot)["net_pose"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nymeria_pose_predict.conf")
    parser.add_argument("--ckpt", default="experiments/nymeria_pose_predict/ckpt/best_model.ckpt")
    parser.add_argument("--out", default="pose_model.onnx")
    parser.add_argument("--window", type=int, default=250,
                        help="IMU window length used for the dummy export tensor")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    conf = ConfigFactory.parse_file(args.config)
    net = PoseNetMotionwithRot(conf.train)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    wrapper = PoseNetONNX(net).eval()

    B, W = 1, args.window
    acc  = torch.randn(B, W, 3)
    gyro = torch.randn(B, W, 3)
    rot  = torch.randn(B, W, 3)

    torch.onnx.export(
        wrapper, (acc, gyro, rot), args.out,
        input_names=["acc", "gyro", "rot"], output_names=["pose"],
        dynamic_axes={
            "acc":  {0: "batch", 1: "time"},
            "gyro": {0: "batch", 1: "time"},
            "rot":  {0: "batch", 1: "time"},
            "pose": {0: "batch", 1: "time_out"},
        },
        opset_version=args.opset,
    )
    print(f"Exported -> {args.out}")

    # Numerical parity check with onnxruntime (if available)
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        onnx_out, = sess.run(None, {"acc": acc.numpy(), "gyro": gyro.numpy(), "rot": rot.numpy()})
        with torch.no_grad():
            torch_out = wrapper(acc, gyro, rot).numpy()
        print(f"output shape: {onnx_out.shape}    "
              f"|onnx - torch|_max = {np.abs(onnx_out - torch_out).max():.2e}")
    except ImportError:
        print("(install onnxruntime to numerically verify)")
