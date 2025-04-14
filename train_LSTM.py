from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics.loss import DiceLoss, seg_metrics, combined_loss_dice_CE, score, TverskyLoss
from torch.cuda.amp import autocast, GradScaler
from postprocess.postprocess import _nms_v2, find_connected_component, find_centroid
from postprocess.postprocess import find_centroid_with_confidence, find_connected_component_with_confidence
from utils.utils import set_seed, initialize_weights, prepare_submission_df, prepare_truths_df, de_dup, visualize_predictions
from utils.utils import merge_points_by_confidence
from utils.awp import AdvWeightPerturb
from neptune.utils import stringify_unsupported
import neptune
import matplotlib.pyplot as plt
from decouple import config
import glob
import numpy as np
import transformers
import torch
import torch.nn.functional as F
import importlib
import argparse
import sys
from copy import copy
import os
import gc
from utils.optimizers import Lookahead

    
NEPTUNE_API_TOKEN = config('NEPTUNE_API_TOKEN')

BASEDIR= '.'
for DIRNAME in 'configs data models postprocess metrics'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')

parser = argparse.ArgumentParser()
parser.add_argument("-C", "--config", help='Config File', type=str, default='cfg_LSTM')
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
set_seed(cfg.seed)   

CryoDataset = importlib.import_module(cfg.dataset).CryoDataset
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device
RCNN = importlib.import_module(cfg.model).RCNN

neptune_api_token=NEPTUNE_API_TOKEN
if cfg.logging:
    # Start neptune
    if cfg.neptune_project == "common/quickstarts":
        neptune_api_token=neptune.ANONYMOUS_API_TOKEN
    else:
        neptune_api_token=NEPTUNE_API_TOKEN
    
    neptune_run = neptune.init_run(
            project=cfg.neptune_project,
            tags="baseline",
            mode="async",
            api_token=neptune_api_token,
            #capture_stdout=False,
            #capture_stderr=False,
            source_files=['configs/*.py', 'data/*.py', 'metrics/*.py', 
                'models/*.py', 'postprocess/postprocess.py', 'utils/utils.py', '*.py'],
            description=cfg.comment,
            #git_ref=False
        )
    print(f"Neptune system id : {neptune_run._sys_id}")
    print(f"Neptune URL       : {neptune_run.get_url()}")
    neptune_run["cfg"] = stringify_unsupported(cfg.__dict__)
    os.mkdir(f'logs/checkpoints/{neptune_run._sys_id}')

for fold in range(cfg.n_folds):
    cfg.fold = fold + cfg.fold_offset
    
    cfg.train_tomograms = [tom for tom in cfg.all_tomograms if tom != cfg.all_tomograms[cfg.fold]]
    cfg.val_tomograms = [cfg.all_tomograms[cfg.fold]]
    cfg.test_tomograms = [cfg.all_tomograms[cfg.fold]]
    
    print(f"Seed: {cfg.seed}, Fold: {cfg.fold} \n Training on: {cfg.train_tomograms} \n Validation on: {cfg.val_tomograms}")

    train_dataset = CryoDataset(cfg, 'train')
    train_dataloader = DataLoader(
            train_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

    val_dataset = CryoDataset(cfg, cfg.eval)
    val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=cfg.batch_size_eval,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

    model = RCNN(cfg).to(cfg.device)
    #weights = f'logs/checkpoints/CRYOET-217/epoch_149_fold{cfg.fold}.pth'

    # checkpoint = torch.load(weights)
    # new_state_dict = {}
    # for k, v in checkpoint["model"].items():
    #     new_k = k.replace("_orig_mod.", "")  # Remove the prefix
    #     new_state_dict[new_k] = v
        
    #model.load_state_dict(new_state_dict, strict=True)
    model = torch.compile(model, disable=not cfg.compile)
    initialize_weights(model)
    dice_loss = DiceLoss()
    tversky_loss = TverskyLoss()
    total_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    #optimizer.load_state_dict(torch.load(f'logs/checkpoints/CRYOET-217/epoch_149_fold{fold}.pth')['optimizer'])
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup * (total_steps //cfg.batch_size),
        num_training_steps=cfg.epochs * (total_steps //cfg.batch_size),
        num_cycles=0.5
    )
    awp = AdvWeightPerturb(model, delta=0.01, eps=1e-6, use_mixed_precision=cfg.mixed_precision, cfg=cfg)


    scaler = GradScaler()
    optimizer.zero_grad()

    cfg.curr_step = 0
    i = 0

    for epoch in range(cfg.epochs):
        cfg.curr_epoch = epoch  
        progress_bar = tqdm(range(len(train_dataloader))[:], desc=f'Train epoch {epoch}', ascii=' >=')
        tr_it = iter(train_dataloader)
        ce_losses = []
        dice_losses = []
        gc.collect()
        
        precisions = []
        recalls = []
        f1_scores = []
        ious = []
        
        model.train()
        torch.set_grad_enabled(True)  
        for itr in progress_bar:
            cfg.curr_step += cfg.batch_size
            data = next(tr_it)
            data = batch_to_device(data, cfg.device)
            
            if cfg.awp and epoch > cfg.awp_start:
                loss, ce_loss, dice_loss = awp.train_step(optimizer, combined_loss_dice_CE, data['image'], data['label'])
                ce_losses.append(ce_loss.item())
                dice_losses.append(dice_loss.item())
            else:
                if cfg.mixed_precision:
                    with autocast():
                        seg_output = model(data['image'])
                else:
                    seg_output = model(data['image']) 
                        
                loss, ce_loss, dice_loss = combined_loss_dice_CE(seg_output, data['label'])
                
                if cfg.mixed_precision:
                    scaler.scale(loss).backward()            
                    if cfg.clip_grad > 0:
                            scaler.unscale_(optimizer)                          
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    loss.backward()
                    if cfg.clip_grad > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()
                
            ce_losses.append(ce_loss.item())
            dice_losses.append(dice_loss.item())
            
            if scheduler is not None:
                scheduler.step()
            
            if cfg.logging:
                    neptune_run[f"train{cfg.fold}/dice_loss"].log(value=(dice_loss.item()), step=cfg.curr_step)
                    neptune_run[f"train{cfg.fold}/ce_loss"].log(value=(ce_loss.item()), step=cfg.curr_step)
                    neptune_run[f"lr{cfg.fold}"].log(value=optimizer.param_groups[0]['lr'], step=cfg.curr_step)
                
        print(f'CE Loss: {np.mean(ce_losses)}, Dice Loss: {np.mean(dice_losses)}')
        
        if epoch >= 0:

            apo_ferritin, beta_galactosidase, ribosome, thyroglobulin, virus_like_particle = [], [], [], [], []
            if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
                model.eval()
                torch.set_grad_enabled(False)    
                for index, data in enumerate(tqdm(val_dataloader, desc=f'Val epoch {epoch}', ascii=' >=')):
                    data = batch_to_device(data, cfg.device)

                    if cfg.mixed_precision:
                        with autocast():
                            seg_output = model(data['image']) 
                    else:
                        seg_output = model(data['image'])
                                
                    seg_output = torch.nn.functional.softmax(seg_output, dim=1)
                    
                    tomogram = data['tomogram'][0]
                    #seg_output = F.one_hot(data['label'].long(), num_classes=cfg.num_classes).permute(0, 4, 1, 2, 3).float()
                    component, confidences = find_connected_component_with_confidence(seg_output[:, 1:], max_radius=25)
                    centroids = find_centroid_with_confidence(component, data['position'], cfg=cfg, tomogram=tomogram, confidences_batched=confidences)
                    for centroid in centroids:
                        if centroid['class'] == 1:
                            apo_ferritin.append(centroid)
                        elif centroid['class'] == 2:
                            beta_galactosidase.append(centroid)
                        elif centroid['class'] == 3:
                            ribosome.append(centroid)
                        elif centroid['class'] == 4:
                            thyroglobulin.append(centroid)
                        elif centroid['class'] == 5:
                            virus_like_particle.append(centroid)
                            
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
                    
                deduped = np.array(apo_ferritin_deduped)
                deduped = np.concatenate((deduped, beta_galactosidase_deduped), axis=0)
                deduped = np.concatenate((deduped, ribosome_deduped), axis=0)
                deduped = np.concatenate((deduped, thyroglobulin_deduped), axis=0)
                deduped = np.concatenate((deduped, virus_like_particle_deduped), axis=0)
                
                submission = prepare_submission_df(deduped, cfg)
                truths = prepare_truths_df(cfg)
                #save submission and truths to disk
                submission.to_csv('logs/submission.csv', index=False)
                truths.to_csv('logs/truths.csv', index=False)
                beta_score, tp_coords, fp_coords = score(solution=truths, submission=submission, distance_multiplier=0.5, beta=4)
                print(f'Beta Score: {beta_score}')
                precision = np.mean(precisions)
                recall = np.mean(recalls)
                f1_score = np.mean(f1_scores)
                iou = np.mean(ious)
                
                if cfg.logging:
                    neptune_run[f"val{cfg.fold}/precision"].log(value=precision, step=cfg.curr_step)
                    neptune_run[f"val{cfg.fold}/recall"].log(value=recall, step=cfg.curr_step)
                    neptune_run[f"val{cfg.fold}/f1_score"].log(value=f1_score, step=cfg.curr_step)
                    neptune_run[f"val{cfg.fold}/iou"].log(value=iou, step=cfg.curr_step)
                    neptune_run[f"val{cfg.fold}/beta_score"].log(value=beta_score, step=cfg.curr_step)  
                
                print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}, IoU: {iou}')
                if cfg.logging and epoch >= 0:
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, f'logs/checkpoints/{neptune_run._sys_id}/epoch_{epoch}_fold{cfg.fold}.pth')

    if cfg.logging:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, f'logs/checkpoints/{neptune_run._sys_id}/epoch_{epoch}_fold{cfg.fold}.pth')
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, 'logs/checkpoints/model_debug.pth')
    

        
        




