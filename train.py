import argparse
import os
import torch
import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pyhocon import ConfigFactory
from tqdm import tqdm
import pypose as pp

from datasets.dataset_motion import SeqeuncesMotionDataset
from model.code import (
    CodeNetMotion,
    CodeNetMotionwithRot,
    CodeNetMotionwithRotPose,
    PoseNetMotionwithRot,
)
from model.losses import get_motion_loss, get_motion_RMSE, get_pose_loss, get_pose_RMSE
from datasets.dataset_utils import collate_fcs
from utils import move_to, save_state, cat_state, so3_log, save_ckpt


NETWORKS = {
    "code":            CodeNetMotion,
    "codewithrot":     CodeNetMotionwithRot,
    "codewithrotpose": CodeNetMotionwithRotPose,
    "posenetwithrot":  PoseNetMotionwithRot,
}


def _is_pose_net(network):
    return isinstance(network, PoseNetMotionwithRot)


def _forward(network, data, label):
    """Call network with rotation if it needs it, otherwise without."""
    if isinstance(network, CodeNetMotionwithRot):
        rot = so3_log(label['gt_rot'][:, :-1, :])
        return network(data, rot)
    return network(data)


def _gt_label(network, data, label):
    """Pick the regression target for the model."""
    if _is_pose_net(network):
        return network.get_label(data['pose'])
    return network.get_label(label['gt_vel'])


def train(network, loader, confs, epoch, optimizer):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses
    """    
    network.train()
    losses, pred_cov = 0, 0
    is_pose = _is_pose_net(network)
    train_loss_fn = get_pose_loss if is_pose else get_motion_loss
    track_cov = (not is_pose) and confs.propcov

    t_range = tqdm(loader)
    for i, (data, _, label) in enumerate(t_range):
        data, label = move_to([data, label], confs.device)

        inte_state = _forward(network, data, label)
        gt_label = _gt_label(network, data, label)
        loss_state = train_loss_fn(inte_state, gt_label, confs)

        losses += loss_state["loss"].item()
        if track_cov:
            pred_cov += loss_state["cov_loss"].mean().item()

        t_range.set_description(
            f"training epoch: %03d, loss: %.06f" % (epoch, loss_state["loss"])
        )
        t_range.refresh()

        optimizer.zero_grad()
        loss_state["loss"].backward()
        if confs.get("gradient_clip", None) is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), confs.gradient_clip)
        optimizer.step()

    return {"loss": losses / (i + 1), "cov": pred_cov / (i + 1)}


def test(network, loader, confs):
    """Test network on validation set."""
    network.eval()
    losses, pred_cov = 0, 0
    is_pose = _is_pose_net(network)
    eval_loss_fn = get_pose_RMSE if is_pose else get_motion_RMSE
    track_cov = (not is_pose) and confs.propcov

    with torch.no_grad():
        t_range = tqdm(loader)
        for i, (data, _, label) in enumerate(t_range):
            data, label = move_to([data, label], confs.device)

            inte_state = _forward(network, data, label)
            gt_label = _gt_label(network, data, label)
            loss_state = eval_loss_fn(inte_state, gt_label, confs)

            losses += loss_state["loss"].item()
            if track_cov:
                pred_cov += loss_state["cov_loss"].mean().item()
                cov_loss_value = torch.sqrt(loss_state['cov_loss'])
            else:
                cov_loss_value = 0

            t_range.set_description(
                "testing loss: %.06f, cov: %.06f, error: %.06f" % (
                    losses / (i + 1), cov_loss_value, loss_state['dist']
                )
            )
            t_range.refresh()

    return {"loss": losses / (i + 1), "cov": pred_cov / (i + 1)}


def evaluate(network, loader, confs):
    """Evaluate network and collect predictions."""
    network.eval()
    evaluate_states, loss_states, labels = {}, {}, {}
    pred_cov = []
    is_pose = _is_pose_net(network)
    eval_loss_fn = get_pose_RMSE if is_pose else get_motion_RMSE
    track_cov = (not is_pose) and confs.propcov

    with torch.no_grad():
        for i, (data, _, label) in enumerate(tqdm(loader)):
            data, label = move_to([data, label], confs.device)

            inte_state = _forward(network, data, label)
            gt_label = _gt_label(network, data, label)
            loss_state = eval_loss_fn(inte_state, gt_label, confs)

            save_state(loss_states, loss_state)
            save_state(evaluate_states, inte_state)
            save_state(labels, label)

            if "cov" in inte_state and inte_state["cov"] is not None:
                pred_cov.append(inte_state["cov"])

        # Aggregate results
        if track_cov:
            cov = torch.cat(pred_cov, dim=-2)
        else:
            cov = torch.tensor(0.0, device=confs.device)

        for k, v in loss_states.items():
            if k != "cov_loss" or track_cov:
                loss_states[k] = torch.stack(v, dim=0)

        cat_state(evaluate_states)
        cat_state(labels)

        label_str = "pose MPJPE" if is_pose else "vel loss"
        print(f"evaluating: {label_str} {loss_states['loss'].mean():.6f}, cov {cov.mean():.6f}")

    return {
        "evaluate": evaluate_states,
        "evaluate_cov": cov,
        "loss": loss_states,
        "labels": labels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    conf = ConfigFactory.parse_file(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = SeqeuncesMotionDataset(data_set_config=conf.train)
    test_dataset = SeqeuncesMotionDataset(data_set_config=conf.test)
    eval_dataset = SeqeuncesMotionDataset(data_set_config=conf.eval)

    # Create collate functions
    collate_fn_train, collate_fn_test = collate_fcs["base"], collate_fcs["base"]    

    # Create dataloaders
    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=conf.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn_train,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = Data.DataLoader(
        test_dataset,
        batch_size=conf.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn_test,
        num_workers=4,
        pin_memory=True,
    )
    eval_loader = Data.DataLoader(
        dataset=eval_dataset,
        batch_size=conf.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn_test,
        drop_last=True,
    )    


    # Load model 
    network_name = conf.train.get("network", "codewithrot")
    if network_name not in NETWORKS:
        raise ValueError(f"Unknown network '{network_name}'. Available: {list(NETWORKS)}")
    print(f"Using network: {network_name}")
    model = NETWORKS[network_name](conf.train).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=conf.train.lr, 
        weight_decay=conf.train.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=conf.train.factor,
        patience=conf.train.patience,
        min_lr=conf.train.min_lr,
    )


    # Training 
    start_epoch = 0
    best_loss = float('inf')
    ckpt_dir = os.path.join(conf.general.exp_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\nStarting training for {conf.train.max_epochs} epochs...")
    for epoch in range(start_epoch, conf.train.max_epochs):
        print(f"\nEpoch {epoch + 1}/{conf.train.max_epochs}")

        train_loss = train(model, train_loader, conf.train, epoch, optimizer)
        test_loss = test(model, test_loader, conf.train)        

        print(f"  Train loss: {train_loss['loss']:.6f}")
        print(f"  Test loss:  {test_loss['loss']:.6f}")

        if epoch % conf.train.eval_freq == conf.train.eval_freq - 1: # doesn't trigger on first epoch (weird hack)
            eval_state = evaluate(network=model, loader=eval_loader, confs=conf.train)
            print(f"  Eval loss: {eval_state['loss']['loss'].mean():.6f}")

        scheduler.step(test_loss['loss'])

        # Save checkpoint
        is_best = test_loss['loss'] < best_loss
        if is_best:
            best_loss = test_loss['loss']
        save_ckpt(model, optimizer, scheduler, epoch, best_loss, conf, is_best)

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
