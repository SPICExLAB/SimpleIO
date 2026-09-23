import torch
from .loss_func import loss_fc_list, diag_ln_cov_loss
from utils import report_hasNan
import numpy as np

def motion_loss_(fc, pred, targ):
    dist = pred - targ
    loss = fc(dist)
    return loss, dist

def get_motion_loss(inte_state, label, confs):
    ## The state loss for evaluation
    loss, cov_loss = 0, {}
    loss_fc = loss_fc_list[confs.loss]
    
    vel_loss, vel_dist = motion_loss_(loss_fc, inte_state['net_vel'],label)

    # Apply the covariance loss
    if confs.propcov:
        #velocity covariance.
        cov = inte_state['cov']
        cov_loss = cov.mean()

        if "covaug" in confs and confs["covaug"] is True:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist, cov)
        else:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist.detach(), cov)
    loss += confs.weight * vel_loss
    return {'loss':loss, 'cov_loss':cov_loss}


def get_motion_RMSE(inte_state, label, confs):
    '''
    get the RMSE of the last state in one segment
    '''
    def _RMSE(x):
        return torch.sqrt((x.norm(dim=-1)**2).mean())
    cov_loss = 0
    dist = (inte_state['net_vel'] - label)
    dist = torch.mean(dist,dim=-2)
    loss = _RMSE(dist)[None,...]

    if confs.propcov:
        #velocity covariance.
        cov = inte_state['cov']
        cov_loss = cov.mean()

    return {'loss': loss,
            'dist': dist.norm(dim=-1).mean(),
            'cov_loss': cov_loss}


def get_pose_loss(inte_state, label, confs):
    """MSE on flattened joint positions (B, T, 69)."""
    dist = inte_state['net_pose'] - label
    loss = dist.pow(2).mean()
    return {'loss': confs.weight * loss, 'cov_loss': torch.zeros((), device=loss.device)}


def get_pose_RMSE(inte_state, label, confs):
    """MPJPE on a (B, T, 23, 3) view of joint positions, in meters.

    Returned 'loss' is the per-window MPJPE (mean over batch+time+joints).
    """
    pred = inte_state['net_pose']
    B, T = pred.shape[:2]
    pred_xyz = pred.view(B, T, -1, 3)
    targ_xyz = label.view(B, T, -1, 3)
    per_joint_err = (pred_xyz - targ_xyz).norm(dim=-1)        # (B, T, 23)
    mpjpe = per_joint_err.mean()                              # scalar
    return {'loss': mpjpe[None, ...],
            'dist': mpjpe,
            'cov_loss': torch.zeros((), device=mpjpe.device)}