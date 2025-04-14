from utils.utils import visualize_centroids
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics.loss import seg_metrics, score
from torch.amp import autocast
from postprocess.postprocess import _nms_v2, morphological_filter_3d
from utils.utils import set_seed, prepare_submission_df, prepare_truths_df, de_dup, merge_points_by_confidence
from postprocess.postprocess import find_connected_component_with_confidence, find_centroid_with_confidence
import matplotlib.pyplot as plt
from decouple import config
import numpy as np
import torch
import importlib
import argparse
import sys
from copy import copy
import os

NEPTUNE_API_TOKEN = config('NEPTUNE_API_TOKEN')

BASEDIR= '.'
for DIRNAME in 'configs data models postprocess metrics'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')

parser = argparse.ArgumentParser()
parser.add_argument("-C", "--config", help='Config File', type=str, default='cfg_inference')
parser.add_argument("-G", "--gpu", help="GPU#", type=str, default='1')
parser.add_argument("--comment", help="Comment", type=str, default='')


parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)
cfg.comment = parser_args.comment
os.environ["CUDA_VISIBLE_DEVICES"] = parser_args.gpu

if len(other_args) > 1:
    other_args = {k.replace('-',''):v for k, v in zip(other_args[1::2], other_args[2::2])}
    
    for key in other_args:
        if key in cfg.__dict__:
            
            print(f"Overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}")
            cfg_type = type(cfg.__dict__[key])
            if cfg_type == bool:
                cfg.__dict__[key] = other_args[key] == 'True'
            elif cfg_type == type(None):
                cfg.__dict__[key] = other_args[key]
            else:
                cfg.__dict__[key] = cfg_type(other_args[key])
                

cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
if cfg.seed < 0:
    cfg.seed = np.random.randint(1_000_000) 
#set_seed(cfg.seed)   
#print(f"Seed : {cfg.seed}")

CryoDataset = importlib.import_module(cfg.dataset).CryoDataset
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device
CryoResUNet3D = importlib.import_module(cfg.model).CryoResUNet3D
MCDSUnet = importlib.import_module('mcds_3dunet').MCDSUnet

beta_scores = []
statistics = []
for fold in range(cfg.n_folds):
    fold = fold + cfg.fold_offset

    cfg.val_tomograms = [cfg.all_tomograms[fold]]
    cfg.test_tomograms = [cfg.all_tomograms[fold]]
    print(f"Fold: {fold} \nValidation on: {cfg.val_tomograms}")

    test_dataset = CryoDataset(cfg, 'test')
    test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size_eval,
            pin_memory=cfg.pin_memory,
        )
    weights = f'logs/checkpoints/CRYOET-420/swa_epoch_{99}_fold{fold}.pth'
    cfg.feature_maps = [24,48,72,106]
    cfg.use_scSE = True
    checkpoint = torch.load(weights, weights_only=True)
    new_state_dict = {}
    for k, v in checkpoint["model"].items():
        new_k = k.replace("module.", "")  # Remove the prefix
        new_k = new_k.replace("basic_", "basic_module.")  # Remove the prefix
        new_k = new_k.replace("_orig_mod.", "")  # Remove the prefix
        new_state_dict[new_k] = v
        
    model = CryoResUNet3D(cfg).to(cfg.device)
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    weights = f'logs/checkpoints/CRYOET-271/epoch_{149}_fold{fold}.pth'
    cfg.feature_maps = [24,48,72,106]
    cfg.use_scSE = False
    checkpoint = torch.load(weights, weights_only=True)
    new_state_dict = {}
    for k, v in checkpoint["model"].items():
        new_k = k.replace("_orig_mod.", "")  # Remove the prefix
        new_state_dict[new_k] = v
        
    model2 = CryoResUNet3D(cfg).to(cfg.device)
    model2.load_state_dict(new_state_dict, strict=True)
    model2.eval()
    
    weights = f'logs/checkpoints/CRYOET-364/epoch_{49}_fold{fold}.pth'
    cfg.feature_maps = [24,48,72]
    cfg.use_scSE = True
    checkpoint = torch.load(weights, weights_only=True)
    new_state_dict = {}
    for k, v in checkpoint["model"].items():
        new_k = k.replace("_orig_mod.", "")  # Remove the prefix
        new_state_dict[new_k] = v
        
    model3 = CryoResUNet3D(cfg).to(cfg.device)
    model3.load_state_dict(new_state_dict, strict=True)
    model3.eval()
    
    weights = f'logs/checkpoints/CRYOET-421/swa_epoch_{49}_fold{fold}.pth'
    cfg.feature_maps = [24,48,72]
    cfg.use_scSE = True
    checkpoint = torch.load(weights, weights_only=True)
    new_state_dict = {}
    for k, v in checkpoint["model"].items():
        new_k = k.replace("module.", "")  # Remove the prefix
        new_k = new_k.replace("basic_", "basic_module.")  # Remove the prefix
        new_k = new_k.replace("_orig_mod.", "")  # Remove the prefix
        new_state_dict[new_k] = v
        
    model4 = CryoResUNet3D(cfg).to(cfg.device)
    model4.load_state_dict(new_state_dict, strict=False)
    model4.eval()
    
    experiment_predictions = []
    precisions, recalls, f1_scores, ious = [], [], [], []
    apo_ferritin, beta_galactosidase, ribosome, thyroglobulin, virus_like_particle = [], [], [], [], []
    torch.set_grad_enabled(False)    
    for index, data in enumerate(tqdm(test_dataloader)):
            data = batch_to_device(data, cfg.device)                        
            img_flip = torch.flip(data['image'], [-1])
            img_flip2 = torch.flip(data['image'], [-2])
            img_flip3 = torch.flip(data['image'], [-3])
        
            if cfg.mixed_precision:
                with autocast('cuda'):
                    seg_output = model(data['image'])
                    seg_flip = model(img_flip)
                    seg_flip2 = model(img_flip2)
                    
                    seg2_output = model2(data['image'])
                    seg2_flip = model2(img_flip)
                    seg2_flip2 = model2(img_flip2)
                    
                    seg3_output = model3(data['image'])
                    seg3_flip = model3(img_flip)
                    seg3_flip2 = model3(img_flip2)
                    
                    seg4_output = model4(data['image'])
                    seg4_flip = model4(img_flip)
                    seg4_flip2 = model4(img_flip2)

            else:
                seg_output = model(data['image'])
            
            seg_output = torch.nn.functional.softmax(seg_output, dim=1)
            seg_flip = torch.nn.functional.softmax(seg_flip, dim=1)
            seg_flip = torch.flip(seg_flip, [-1])
            seg_flip2 = torch.nn.functional.softmax(seg_flip2, dim=1)
            seg_flip2 = torch.flip(seg_flip2, [-2])
            
            seg2_output = torch.nn.functional.softmax(seg2_output, dim=1)
            seg2_flip = torch.nn.functional.softmax(seg2_flip, dim=1)
            seg2_flip = torch.flip(seg2_flip, [-1])
            seg2_flip2 = torch.nn.functional.softmax(seg2_flip2, dim=1)
            seg2_flip2 = torch.flip(seg2_flip2, [-2])
            
            seg3_output = torch.nn.functional.softmax(seg3_output, dim=1)
            seg3_flip = torch.nn.functional.softmax(seg3_flip, dim=1)
            seg3_flip = torch.flip(seg3_flip, [-1])
            seg3_flip2 = torch.nn.functional.softmax(seg3_flip2, dim=1)
            seg3_flip2 = torch.flip(seg3_flip2, [-2])
            
            seg4_output = torch.nn.functional.softmax(seg4_output, dim=1)
            seg4_flip = torch.nn.functional.softmax(seg4_flip, dim=1)
            seg4_flip = torch.flip(seg4_flip, [-1])
            seg4_flip2 = torch.nn.functional.softmax(seg4_flip2, dim=1)
            seg4_flip2 = torch.flip(seg4_flip2, [-2])
            
            seg_output = (seg_output + seg_flip + seg_flip2) / 3
            seg2_output = (seg2_output + seg2_flip + seg2_flip2) / 3
            seg3_output = (seg3_output + seg3_flip + seg3_flip2) / 3
            seg4_output = (seg4_output + seg4_flip + seg4_flip2) / 3
            
            seg_output = (seg_output + seg2_output + seg3_output + seg4_output) / 4
            
            tomogram = data['tomogram'][0]
            component, confidences = find_connected_component_with_confidence(seg_output[:, 1:], 
                                                threshold=[0.35, 0.3, 0.5, 0.3, 0.5], max_radius=25)
            centroids = find_centroid_with_confidence(component, data['position'], cfg=cfg, tomogram=tomogram, confidences_batched=confidences)
            for centroid in centroids:
                if centroid['class'] == 1:
                    if centroid['volume'] > 10:
                        apo_ferritin.append(centroid)
                elif centroid['class'] == 2:
                    if centroid['volume'] > 10:
                        beta_galactosidase.append(centroid)
                elif centroid['class'] == 3:
                    if centroid['volume'] > 10:
                        ribosome.append(centroid)
                elif centroid['class'] == 4:
                    if centroid['volume'] > 10:
                        thyroglobulin.append(centroid)
                elif centroid['class'] == 5:
                    if centroid['volume'] > 10:
                        virus_like_particle.append(centroid)
                    
            #visualize_predictions(data['label'][0].cpu().numpy(), seg_output[0].cpu().numpy())
                    
            precision, recall, f1_score, iou = seg_metrics(seg_output, data['label'])
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            ious.append(iou)
        
        
    apo_ferritin_deduped = merge_points_by_confidence(apo_ferritin, 4, 1, tomogram)
    beta_galactosidase_deduped = merge_points_by_confidence(beta_galactosidase, 6, 2, tomogram)
    ribosome_deduped = merge_points_by_confidence(ribosome, 12, 3, tomogram)
    thyroglobulin_deduped = merge_points_by_confidence(thyroglobulin, 10, 4, tomogram)
    virus_like_particle_deduped = merge_points_by_confidence(virus_like_particle, 9, 5, tomogram)
    
        
    deduped = np.concatenate([apo_ferritin_deduped,beta_galactosidase_deduped,ribosome_deduped,
                thyroglobulin_deduped,virus_like_particle_deduped], axis=0)

    submission = prepare_submission_df(deduped, cfg)
    truths = prepare_truths_df(cfg)
    #save submission and truths to disk
    submission.to_csv('logs/submission.csv', index=False)
    truths.to_csv('logs/truths.csv', index=False)
    beta_score, TP_coordinate, FP_coordinates = score(solution=truths, submission=submission, distance_multiplier=0.5, beta=4)
    beta_scores.append(beta_score)
    print(f'Beta Score: {beta_score}')

    # for i in range(5):
    #     visualize_centroids(TP_coordinate[i], cfg.test_tomograms[0])

    #visualize_centroids(FP_coordinates[1], cfg.test_tomograms[0])

    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1_score = np.mean(f1_scores)
    iou = np.mean(ious)
    statistics.append([precision, recall, f1_score, iou])

print(f'Beta Scores: {beta_scores}')
print(f'Mean Beta Score: *****{np.mean(beta_scores)}*****')
        
    




