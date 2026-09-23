import os
import sys
import glob
import torch

import torch.utils.data as Data
import argparse
import pickle

import tqdm
from pyhocon import ConfigFactory

from datasets import collate_fcs, SeqeuncesMotionDataset
from model.code import (
    CodeNetMotion,
    CodeNetMotionwithRot,
    CodeNetMotionwithRotPose,
    PoseNetMotionwithRot,
)
from utils import move_to, save_state, so3_log

NETWORKS = {
    "code":            CodeNetMotion,
    "codewithrot":     CodeNetMotionwithRot,
    "codewithrotpose": CodeNetMotionwithRotPose,
    "posenetwithrot":  PoseNetMotionwithRot,
}


def inference(network, loader, confs):
    '''
    Correction inference
    save the corrections generated from the network.
    '''
    network.eval()
    evaluate_states = {}
    with torch.no_grad():
        inte_state = None
        for data, _, label in tqdm.tqdm(loader):
            data, label = move_to([data, label],  confs.device)
            rot = so3_log(label['gt_rot'][:, :-1, :])
            inte_state = network.forward(data, rot)
            inte_state['ts'] = network.get_label(data['ts'][...,None])[0]
            save_state(evaluate_states, inte_state)
           
        for k, v in evaluate_states.items():    
            evaluate_states[k] = torch.cat(v,  dim=-2)
    return evaluate_states

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/EuRoC/motion_body.conf', help='config file path')
    parser.add_argument('--load', type=str, default=None, help='path for specific model check point, Default is the best model')
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size.')
    parser.add_argument('--seqlen', type=int, default=250, help='window size for chunked inference')
    parser.add_argument('--whole', default=False, action="store_true", help='estimate the whole seq in one shot (default: chunked)')


    args = parser.parse_args(); print(args)
    conf = ConfigFactory.parse_file(args.config)
    conf.train.device = args.device
    conf['device'] = args.device
    dataset_conf = conf.eval

    network_name = conf.train.get("network", "codewithrot")
    if network_name not in NETWORKS:
        raise ValueError(f"Unknown network '{network_name}'. Available: {list(NETWORKS)}")
    print(f"Using network: {network_name}")
    network = NETWORKS[network_name](conf.train).to(args.device)
    save_folder = os.path.join(conf.general.exp_dir, "evaluate")
    os.makedirs(save_folder, exist_ok=True)

    if args.load is None:
        ckpt_path = os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt")
    else:
        ckpt_path = os.path.join(conf.general.exp_dir, "ckpt", args.load)

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device),weights_only=True)
        print("loaded state dict %s in epoch %i"%(ckpt_path, checkpoint["epoch"]))
        network.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise KeyError(f"No model loaded {ckpt_path}")
        sys.exit()
        
    collate_fn = collate_fcs['motion']

    cov_result, rmse = [], []
    net_out_result = {}
    evals = {}
    dataset_conf.data_list[0]["window_size"] = args.seqlen
    dataset_conf.data_list[0]["step_size"] = args.seqlen
    for data_conf in dataset_conf.data_list:
        if isinstance(data_conf.data_drive, list) and len(data_conf.data_drive) == 0:
            paths = sorted(
                os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(data_conf["data_root"], "*.pkl"))
            )
        else:
            paths = list(data_conf.data_drive)
        for path in paths:
            dataset_conf["mode"] = "inference" if args.whole else "infevaluate"
            dataset_conf["exp_dir"] = conf.general.exp_dir
            eval_dataset = SeqeuncesMotionDataset(data_set_config=dataset_conf, data_path=path, data_root=data_conf["data_root"])
            eval_loader = Data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, 
                                            shuffle=False, collate_fn=collate_fn, drop_last = False)
            inference_state = inference(network=network, loader = eval_loader, confs=conf.train)
            if isinstance(network, PoseNetMotionwithRot):
                inference_state['net_pose'] = inference_state['net_pose'][0]  # TODO: batch size != 1
            else:
                if not "cov" in inference_state.keys():
                    inference_state["cov"] = torch.zeros_like(inference_state["net_vel"])
                inference_state['net_vel'] = inference_state['net_vel'][0]  # TODO: batch size != 1
            net_out_result[path] = inference_state

    net_result_path = os.path.join(conf.general.exp_dir, 'net_output.pickle')
    print("save netout, ", net_result_path)
    with open(net_result_path, 'wb') as handle:
        pickle.dump(net_out_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
   