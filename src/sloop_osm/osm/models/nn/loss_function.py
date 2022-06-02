import torch
import torch.nn as nn
import math
from sloop.datasets.dataloader import FdFoRefAngle

def clamp_angle_tensor(angle):
    """Given an arbitrary angle (tensor), return an equivalent angle
    in the sense of the semantics of the frame of reference
    that is within 0 to 180 degrees (in radians)"""
    angle = torch.abs(angle)
    angle = (angle % (math.pi*2))
    for i, a in enumerate(angle):
        if a > math.pi:
            a = math.pi*2 - a
        angle[i] = a
    return angle

def clamp_angle(angle):
    """Given an arbitrary angle (float), return an equivalent angle
    in the sense of the semantics of the frame of reference
    that is within 0 to 180 degrees (in radians)"""
    angle = abs(angle)
    angle = (angle % (math.pi*2))
    if angle > math.pi:
        angle = math.pi*2 - angle
    return angle

class FoRefLoss(nn.Module):
    def __init__(self, dataset, device="cpu", reduction="sum"):
        super(FoRefLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.dataset = dataset
        self.device = device

    def forward(self, prediction, label):
        pred_foref_angle = self.dataset.rescale(FdFoRefAngle.NAME, prediction[:,2])
        label_foref_angle = self.dataset.rescale(FdFoRefAngle.NAME, label[:,2])
        angle_diff = clamp_angle_tensor(pred_foref_angle - label_foref_angle)
        # angle diff could be normalized by dividing by pi since it's within 0-pi
        angle_diff = (angle_diff / math.pi).reshape(prediction.shape[0], -1)
        zeros = torch.zeros(angle_diff.shape).to(self.device)

        pred_vector = torch.cat([prediction[:,:2], angle_diff], 1)
        label_vector = torch.cat([label[:,:2], zeros], 1)
        return self.mse_loss(pred_vector, label_vector)


class FoRefAngleLoss(nn.Module):
    def __init__(self, dataset, device="cpu", reduction="mean"):
        super(FoRefAngleLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.dataset = dataset
        self.device = device

    def forward(self, prediction, label):
        pred_foref_angle = self.dataset.rescale(FdFoRefAngle.NAME, prediction[:,0])
        label_foref_angle = self.dataset.rescale(FdFoRefAngle.NAME, label[:,0])
        angle_diff = clamp_angle_tensor(pred_foref_angle - label_foref_angle)
        # angle diff could be normalized by dividing by pi since it's within 0-pi
        angle_diff = (angle_diff / math.pi).reshape(prediction.shape[0], -1)
        zeros = torch.zeros(angle_diff.shape).to(self.device)
        return self.mse_loss(angle_diff, zeros)
