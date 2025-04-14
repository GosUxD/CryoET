from torch import nn as nn
import os
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import dilation
from scipy.spatial import distance
from pycm import ConfusionMatrix
from pycm.output import table_print, stat_print
from pycm.params import SUMMARY_CLASS, SUMMARY_OVERALL
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import torch
import numpy as np
import pandas as pd
import sys


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


# Dice loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, args=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        # flatten label and prediction tensors
        targets = F.one_hot(targets, num_classes=outputs.size(1)).permute(0, 4, 1, 2, 3)
        outputs = F.softmax(outputs, dim=1, dtype=torch.float32)
        
        outputs = flatten(outputs)
        targets = flatten(targets)

        intersection = (outputs * targets).sum(-1)
    
        dice = (2. * intersection + self.smooth) / (outputs.sum(-1) + targets.sum(-1) + self.smooth)
        return 1 - dice.mean()

dice = DiceLoss()

class TverskyLoss(nn.Module):
    """
    Tversky Loss = 1 - Tversky Index
    Tversky Index = TP / (TP + alpha * FN + beta * FP)
    alpha + beta = 1 is common but not required.
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, outputs, targets):
        # Flatten (C, N*D*H*W) if 3D; for 2D it would be (C, N*H*W)
        # but let's assume 3D for consistency with your Dice code
        targets = F.one_hot(targets, num_classes=outputs.size(1)).permute(0, 4, 1, 2, 3)
        outputs = F.softmax(outputs, dim=1, dtype=torch.float32)
        
        outputs = flatten(outputs)
        targets = flatten(targets)

        # True Positives, False Positives & False Negatives
        TP = (outputs * targets).sum(dim=-1)
        FP = (outputs * (1 - targets)).sum(dim=-1)
        FN = ((1 - outputs) * targets).sum(dim=-1)

        # Tversky Index
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FN + self.beta * FP + self.smooth
        )

        # Tversky Loss
        tversky_loss = 1 - tversky_index.mean()
        return tversky_loss
    
tversky = TverskyLoss()

def combined_loss_dice_CE(logits, targets):
    #class_weights= (('apo-ferritin', 62400), ('beta-galactosidase', 3080), ('ribosome', 1800), ('thyroglobulin', 10100), ('virus-like-particle', 8400), ('beta-amylase', 4130))
    #weights = torch.tensor((1, 624, 30, 18, 101, 84), dtype=torch.float).to(outputs.device)
    weights = torch.tensor((1, 2, 3, 2, 3, 2), dtype=torch.float).to(logits.device)
        
    with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision
        ce_loss = F.cross_entropy(logits.float(), targets)#, weights) #disabled weighing
    dice_loss = dice(logits, targets)
    
    return dice_loss * 0.5 + ce_loss * 0.5, ce_loss, dice_loss


def seg_metrics(y_pred, y_true, smooth=1e-7, isTrain=True, threshold=0.5, use_sigmoid=False):
    y_true = F.one_hot(y_true.long(), num_classes=y_pred.size(1)).permute(0, 4, 1, 2, 3)
    y_pred = torch.where(y_pred < threshold, torch.zeros(1).cuda(), torch.ones(1).cuda())

    #flatten label and prediction tensors
    y_pred = flatten(y_pred)
    y_true = flatten(y_true)
        
    tp = (y_true * y_pred).sum(-1)
    fp = ((1 - y_true) * y_pred).sum(-1)
    fn = (y_true * (1 - y_pred)).sum(-1)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    iou = (tp + smooth) / (tp + fn + fp + smooth)    
    f1 = 2 * (precision*recall) / (precision + recall + smooth)

    mean_precision = precision.mean()
    mean_recall = recall.mean()
    mean_iou = iou.mean()
    mean_f1 = f1.mean()

    # for training, ouput mean metrics
    if isTrain:
        return mean_precision.item(), mean_recall.item(), mean_iou.item(), mean_f1.item()
    # for testing, output metrics array by class with threshold
    else:
        return precision.detach().cpu().numpy(), recall.detach().cpu().numpy(), iou.detach().cpu().numpy(), f1.detach().cpu().numpy()




def kaggle_compute_metrics(reference_points, reference_radius, candidate_points):
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0, [], []

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles, [], []

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)    
    
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    
    matches_within_threshold = set(matches_within_threshold)
    #print(f"Within threshold: {len(matches_within_threshold)}", matches_within_threshold)
    tp_coordinates = []
    fp_coordinates = []
    for idx, match in enumerate(raw_matches):
        if match == []:
            fp_coordinates.append(candidate_points[idx])
        else:
            tp_coordinates.append(candidate_points[idx])    

    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn, tp_coordinates, fp_coordinates


def score(solution: pd.DataFrame,
        submission: pd.DataFrame,
        distance_multiplier: float,
        beta: int) -> float:
    '''
    F_beta
      - a true positive occurs when
         - (a) the predicted location is within a threshold of the particle radius, and
         - (b) the correct `particle_type` is specified
      - raw results (TP, FP, FN) are aggregated across all experiments for each particle type
      - f_beta is calculated for each particle type
      - individual f_beta scores are weighted by particle type for final score
    '''

    particle_radius = {
        'apo-ferritin': 60,
        'beta-amylase': 65,
        'beta-galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus-like-particle': 135,
    }

    weights = {
        'apo-ferritin': 1, #1
        'beta-amylase': 0, #0
        'beta-galactosidase': 2, #2
        'ribosome': 1, #1
        'thyroglobulin': 2, #2
        'virus-like-particle': 1, #1
    }

    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}

    # Filter submission to only contain experiments found in the solution split
    split_experiments = set(solution['experiment'].unique())
    submission = submission.loc[submission['experiment'].isin(split_experiments)]

    assert solution.duplicated(subset=['experiment', 'x', 'y', 'z']).sum() == 0
    assert particle_radius.keys() == weights.keys()

    results = {}
    for particle_type in solution['particle_type'].unique():
        results[particle_type] = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
        }
    fp_coordinates = [[], [], [], [], []]
    tp_coordinates = [[], [], [], [], []]

    for experiment in split_experiments:
        for idx, particle_type in enumerate(solution['particle_type'].unique()):
            reference_radius = particle_radius[particle_type]
            select = (solution['experiment'] == experiment) & (solution['particle_type'] == particle_type)
            reference_points = solution.loc[select, ['x', 'y', 'z']].values

            select = (submission['experiment'] == experiment) & (submission['particle_type'] == particle_type)
            candidate_points = submission.loc[select, ['x', 'y', 'z']].values

            if len(reference_points) == 0:
                reference_points = np.array([])
                reference_radius = 1

            if len(candidate_points) == 0:
                candidate_points = np.array([])

            tp, fp, fn, tp_coors, fp_coors = kaggle_compute_metrics(reference_points, reference_radius, candidate_points)
            if len(tp_coors) > 0:
                tp_coordinates[idx].extend(tp_coors)
            if len(fp_coors) > 0:
                fp_coordinates[idx].extend(fp_coors)
            
            results[particle_type]['total_tp'] += tp
            results[particle_type]['total_fp'] += fp
            results[particle_type]['total_fn'] += fn

    aggregate_fbeta = 0.0
    for particle_type, totals in results.items():
        tp = totals['total_tp']
        fp = totals['total_fp']
        fn = totals['total_fn']
        
        print(f'{particle_type}: TP={tp}, FP={fp}, FN={fn}')

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        aggregate_fbeta += fbeta * weights.get(particle_type, 1.0)

    if weights:
        aggregate_fbeta = aggregate_fbeta / sum(weights.values())
    else:
        aggregate_fbeta = aggregate_fbeta / len(results)
    return aggregate_fbeta, tp_coordinates, fp_coordinates




















