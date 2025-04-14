from skimage.morphology import binary_opening, disk, erosion, dilation
from skimage.measure import label, regionprops
import numpy as np
from scipy.spatial import KDTree
import numpy as np
import json
import zarr
from matplotlib import pyplot as plt
from matplotlib import cm  # For colormaps
import os
import random
import torch
import pandas as pd
import sys


class BNUpdateWrapper:
    def __init__(self, dataloader, key='image'):
        self.dataloader = dataloader
        self.key = key

    def __iter__(self):
        for batch in self.dataloader:
            yield batch[self.key]

    def __len__(self):
        return len(self.dataloader)

def unfreeze_layers(epoch, model):
    """ Gradually unfreezes layers based on epoch number. """
    if epoch == 2:  # After 5 epochs, unfreeze mid-level features
        for name, param in model.named_parameters():
            if "encoders.2" in name:  
                param.requires_grad = True
    elif epoch == 5:  # After 10 epochs, fully unfreeze the model
        for param in model.parameters():
            param.requires_grad = True


def set_seed(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False #initial
    torch.backends.cudnn.deterministic = True #initial
    
    os.environ['TORCHDYNAMO_REPORT_GUARD_FAILURES']='1'

def gaussian3D(shape, sigma=1):
    l, m, n = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-l:l + 1, -m:m + 1, -n:n + 1]
    sigma = (sigma - 1.) / 2.
    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    return h

def load_coordinates_xyz(experiment, particle):
    with open(f'/home/daniel/AI/Projects/3.CryoET/1.CompetitionModel/datamount/train/overlay/{experiment}/Picks/{particle}.json') as f:
        coordinates = []
        data = json.load(f)['points']
        for item in data:
            coordinates.append(item['location'])
        return coordinates
    
def load_coordinates_json(experiment, particle):
    with open(f'/home/daniel/AI/Projects/3.CryoET/1.CompetitionModel/datamount/train/overlay/{experiment}/Picks/{particle}.json') as f:
        return json.load(f)
    
def load_labels(tomogram):
    label = np.load(f'/home/daniel/AI/Projects/3.CryoET/1.CompetitionModel/datamount/train/truths/{tomogram}.npy')
    return label

def load_tomogram(tomogram, type='denoised'):    
    zarr_file = zarr.open(f'/home/daniel/AI/Projects/3.CryoET/1.CompetitionModel/datamount/train/static/{tomogram}/VoxelSpacing10.000/{type}.zarr', mode='r')
    return zarr_file['0']

def load_tomogram_all_types(tomogram):
    tomograms = []
    for type in ['denoised','ctfdeconvolved','isonetcorrected','wbp']:
        zarr_file = zarr.open(f'/home/daniel/AI/Projects/3.CryoET/1.CompetitionModel/datamount/train/static/{tomogram}/VoxelSpacing10.000/{type}.zarr', mode='r')
        tomograms.append(zarr_file['0'])
    return tomograms

def prepare_datapoints(cfg, mode):
    train_datapoints = []        
    for tomogram in cfg.train_tomograms if mode == 'train' else cfg.val_tomograms:
        for particle in cfg.particles:
            datapoints = load_coordinates_json(tomogram, particle)
            for datapoint in datapoints['points']:
                point = {"x": datapoint['location']['x'], 
                            "y": datapoint['location']['y'], 
                            "z": datapoint['location']['z'], 
                            "tomogram": tomogram, 
                            "radius": cfg.particle_radius[particle]}
                train_datapoints.append(point)

                    
    return train_datapoints




def visualize_3d(data):
    data = np.transpose(data, (2, 0, 1))
    volume_shape = data.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    
    cmap = cm.gray  # Grayscale colormap
    norm = plt.Normalize(vmin=0, vmax=1)  # Normalize values to [0, 1]
    facecolors = cmap(norm(data))
    
    # Plot the 3D voxels
    ax.voxels(data, facecolors=facecolors, edgecolor='k', linewidth=0.5)
    
    # Label the axes
    ax.set_xlabel('Width (X)')
    ax.set_zlabel('Height (Y)')
    ax.set_ylabel('Depth (Z)')
    
    print(data.shape)
    box_aspect = (volume_shape[0], volume_shape[1], volume_shape[2])  # (width, height, depth)
    ax.set_box_aspect(box_aspect)
    
    ax.view_init(azim=-80, elev=30)
    #plt.savefig('logs/labels_sanity/3d.png')
    plt.show()
    

def normalize(x):
    lower, upper = np.percentile(x, (0.5, 99.5))
    x = np.clip(x, lower, upper)
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return x

def z_score_normalize(tomogram):
    mean = np.mean(tomogram)
    std = np.std(tomogram)
    return (tomogram - mean) / std

def z_score_normalize_clip(x):
    lower, upper = np.percentile(x, (0.5, 99.5))
    x = np.clip(x, lower, upper)
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std

import torch.nn as nn

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def prepare_submission_df(predictions, cfg):
    df = pd.DataFrame(columns=['experiment','particle_type','x','y','z'])
    rows = []
    
    for pred in predictions:
        try:
            x, y, z, cls = pred['x'] * 10.012, pred['y'] * 10.012, pred['z'] * 10.012, pred['class']
            experiment = pred['tomogram']
            particle_type = cfg.index_to_particle[int(cls)]
            rows.append([experiment, particle_type, x, y, z])
        except Exception as e:
            print("EXCEPTION", e)
    df = pd.DataFrame(rows, columns=['experiment','particle_type','x','y','z'])
    
    return df

def de_dup(pred, radius):
    if len(pred) == 0:
        return []
    mini_dist = radius
    indexs = []
    areas = np.array([p['prob'] for p in pred])
    pred = np.array(pred)
    
    pred_final_ = [[p['z'], p['y'], p['x']] for p in pred]
    pred_final_ = np.array(pred_final_)
    for idx, item in enumerate(pred_final_):
        if idx in indexs:
            continue
        d2 = np.linalg.norm(pred_final_ - item, ord=2, axis=1)
        tmp = (d2 < mini_dist)
        tmp[idx] = False
        if len(np.nonzero(tmp)[0]) >= 1:
            temp_idx = np.nonzero(tmp)[0].tolist()
            temp_idx.insert(0, idx)
            for idx_, item in enumerate(temp_idx):
                if item in indexs:
                    del temp_idx[idx_]
            max_idx = np.argmax(areas[temp_idx])
            del temp_idx[max_idx]
            indexs.extend(temp_idx)
            
    pred = np.delete(pred, indexs, axis=0)
        
    return pred

def merge_points_by_confidence(pred, radius, cls, tomogram):
    if len(pred) == 0:
        return []
    
    mini_dist = radius
    merged_points = []
    used_indices = set()
    pred = np.array(pred)
    
    # Extract areas (confidence) and coordinates
    confidences = np.array([p['prob'] for p in pred])
    coordinates = np.array([[p['z'], p['y'], p['x']] for p in pred])
    
    for idx, item in enumerate(coordinates):
        if idx in used_indices:
            continue
        
        # Find points within the radius
        d2 = np.linalg.norm(coordinates - item, ord=2, axis=1)
        group_mask = d2 < mini_dist
        group_indices = np.where(group_mask)[0]
        
        # Mark indices as used
        used_indices.update(group_indices)
        
        # Calculate weighted centroid
        group_coordinates = coordinates[group_indices]
        group_confidences = confidences[group_indices] ** 20
        total_confidence = group_confidences.sum()
        weighted_centroid = np.sum(group_coordinates.T * group_confidences, axis=1) / total_confidence
        # Merge the group into a single point
        merged_point = {
            'z': weighted_centroid[0],
            'y': weighted_centroid[1],
            'x': weighted_centroid[2],
            'class': cls,
            'tomogram': tomogram,
            'prob': confidences[group_indices].max()  # Optionally sum confidences or use max
        }
        merged_points.append(merged_point)
    
    return merged_points


def prepare_truths_df(cfg):
    df = pd.DataFrame(columns=['experiment','particle_type','x','y','z'])
    rows = []
    for tomogram in cfg.val_tomograms:
        for particle in cfg.particles:
            datapoints = load_coordinates_json(tomogram, particle)
            for datapoint in datapoints['points']:
                experiment = tomogram
                particle_type = particle
                z, y, x = datapoint['location']['z'], datapoint['location']['y'], datapoint['location']['x']
                rows.append([experiment, particle_type, x, y, z])
    df = pd.DataFrame(rows, columns=['experiment','particle_type','x','y','z'])    
    
    return df

import matplotlib.patches as patches

def visualize_3d_on_axis(ax, data):
    """
    Draw 3D voxels onto an existing Axes3D (ax).
    """
    # Move your dimensions around if needed (depends on your data shape)
    data = np.transpose(data, (2, 0, 1))  # (Z, X, Y)

    # Build a grayscale colormap, or any other you like
    cmap = cm.gray
    norm = plt.Normalize(vmin=0, vmax=1)
    facecolors = cmap(norm(data))

    # Plot the 3D voxels
    ax.voxels(data, facecolors=facecolors, edgecolor='k', linewidth=0.5)
    
    # Label axes
    ax.set_xlabel('Width (X)')
    ax.set_zlabel('Height (Y)')
    ax.set_ylabel('Depth (Z)')

    # Adjust aspect ratio
    z_dim, x_dim, y_dim = data.shape
    ax.set_box_aspect((x_dim, y_dim, z_dim))  # (X, Y, Z) order in set_box_aspect

    ax.view_init(azim=-80, elev=30)

def create_rectangle(pad_size=18):
    return patches.Rectangle(
    (pad_size, pad_size),   # Bottom-left corner of the rectangle (x, y)
    72 - 2*pad_size,    # Width (72 - 2*18)
    72 - 2*pad_size,    # Height (72 - 2*18)
    linewidth=2, # Thickness of the rectangle border
    edgecolor='red',  # Color of the border
    facecolor='none'  # No fill color
    )

def visualize_predictions(labels, segmentation, pad_size=18):
   
    plt.figure(figsize=(15, 5))
    #make the min and max of the plot
    plt.subplot(1,6,1)
    plt.imshow(labels.max(0), cmap='gray')
    plt.gca().add_patch(create_rectangle(pad_size))
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(1,6,2)
    plt.imshow(segmentation[0].max(0), cmap='gray')
    plt.gca().add_patch(create_rectangle(pad_size))
    plt.title('Predictions')
    plt.axis('off')
    plt.subplot(1,6,3)
    plt.imshow(segmentation[1].max(0), cmap='gray')
    plt.gca().add_patch(create_rectangle(pad_size))
    plt.title('Predictions')
    plt.axis('off')
    plt.subplot(1,6,4)
    plt.imshow(segmentation[2].max(0), cmap='gray')
    plt.gca().add_patch(create_rectangle(pad_size))
    plt.title('Predictions')
    plt.axis('off')
    plt.subplot(1,6,5)
    plt.imshow(segmentation[3].max(0), cmap='gray')
    plt.gca().add_patch(create_rectangle(pad_size))
    plt.title('Predictions')
    plt.axis('off')
    plt.subplot(1,6,6)
    plt.imshow(segmentation[4].max(0), cmap='gray')
    plt.gca().add_patch(create_rectangle(pad_size))
    plt.title('Predictions')
    plt.axis('off')
    
    
    plt.show()
 

def visualize_centroids(centroids, tomogram_name):
    tomogram = load_tomogram(tomogram_name, type='isonetcorrected')
    labels = load_labels(tomogram_name)
    
    for idx, centroid in enumerate(centroids):
        print(f"Analyzing {idx+1}/{len(centroids)}")
        x, y, z = int(centroid[0]//10), int(centroid[1]//10), int(centroid[2]//10)
        delta = 36
        print(f'Centroid: {x, y, z}')
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(tomogram[z, y-delta:y+delta, x-delta:x+delta], cmap='gray')
        plt.title('Tomogram')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(labels[z, y-delta:y+delta, x-delta:x+delta], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        plt.show()   

    
def combine_torch(data, shape, block_size=72, pad_size=18, reverse=False):
    if reverse:
        shape = shape[::-1]
    union_data = torch.zeros(shape, device=data.device)
    step_size = block_size - 2 * pad_size
    block_size = step_size

    for i in range(shape[0] // step_size + (1 if shape[0] % step_size > 0 else 0)):
        for j in range(shape[1] // step_size + (1 if shape[1] % step_size > 0 else 0)):
            for k in range(shape[2] // step_size + (1 if shape[2] % step_size > 0 else 0)):
                if i == shape[0] // step_size + (1 if shape[0] % step_size > 0 else 0) - 1:
                    x = shape[0] - block_size // 2
                else:
                    x = i * step_size + block_size // 2

                if j == shape[1] // step_size + (1 if shape[1] % step_size > 0 else 0) - 1:
                    y = shape[1] - block_size // 2
                else:
                    y = j * step_size + block_size // 2

                if k == shape[2] // step_size + (1 if shape[2] % step_size > 0 else 0) - 1:
                    z = shape[2] - block_size // 2
                else:
                    z = k * step_size + block_size // 2

                union_data[x - block_size // 2: x + block_size // 2,
                y - block_size // 2: y + block_size // 2,
                z - block_size // 2: z + block_size // 2] = data[
                    i * (shape[1] // block_size + (1 if shape[1] % step_size > 0 else 0)) *
                    (shape[2] // block_size + (1 if shape[2] % step_size > 0 else 0)) + j *
                    (shape[2] // block_size + (1 if shape[2] % step_size > 0 else 0)) + k]

    return union_data