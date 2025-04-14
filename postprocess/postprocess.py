import torch
import torch.nn as nn
import sys
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, ball


def _nms_v2(pred, cfg, cls, tomogram, kernel=3, mp_num=5, positions=None, thresh_prob=0.5, thresh_after_nms=0.1):
        #print("NMS: Pred shape", pred.shape)
        cfg = cfg
        pred = torch.where(pred > thresh_prob, 1, 0)
        meanPool = nn.AvgPool3d(kernel, 1, kernel // 2).cuda()
        maxPool = nn.MaxPool3d(kernel, 1, kernel // 2).cuda()
        hmax = pred.clone().float()
        for _ in range(mp_num):
            hmax = meanPool(hmax)
        pred = hmax.clone()
        hmax = maxPool(hmax)
        keep = ((hmax == pred).float()) * ((pred > thresh_after_nms).float())
        #print("NMS: Keep shape", keep.shape)
        coords = keep.nonzero()  # [N, 5]
        #print("NMS: Coords shape", coords.shape)
        if coords.shape[0] > 2000 or coords.shape[0] == 0:
            return None
        #print("Coords", coords)
        discard_range = cfg.pad_size
        coords = coords[coords[:, 1] >= discard_range]
        coords = coords[coords[:, 1] < cfg.block_size - discard_range]
        coords = coords[coords[:, 2] >= discard_range]
        coords = coords[coords[:, 2] < cfg.block_size - discard_range]
        coords = coords[coords[:, 3] >= discard_range]
        coords = coords[coords[:, 3] < cfg.block_size - discard_range]
        
        try:
            h_val = torch.stack(
                [hmax[item[0], item[1], item[2], item[3]] for item in coords], dim=0).unsqueeze(1)
            leftTop_coords = (positions[coords[:, 0]] - (cfg.block_size // 2)) - cfg.pad_size
            leftTop_coords = torch.clamp(leftTop_coords, 0, None)
            coords[:, 1:4] = coords[:, 1:4] + leftTop_coords
            particle_class = torch.ones((len(coords), 1), device=cfg.device) * cls
            
            # preds = torch.cat(
            #     [particle_class, coords[:, 1:2], coords[:, 2:3], coords[:, 3:4], h_val],
            #     dim=1).cpu().numpy().tolist()
            
            pred_final = []
            for idx, pred in enumerate(coords):
                centroid = {'z': pred[1:2].item(), 
                            'y': pred[2:3].item(), 
                            'x': pred[3:4].item(), 
                            'prob': h_val[idx].item(),
                            'class': particle_class[idx].item(),
                            'tomogram': tomogram}
                pred_final.append(centroid)
            return pred_final
        except Exception as e:
            #print(e)
            return None
import torch
import torch.nn.functional as F

def morphological_filter_3d(input_tensor, operation='erosion', kernel_size=3, threshold=0.5, num_passes=1, device='cuda'):
    """
    Apply 3D morphological filtering to remove small particles.
    
    Args:
        input_tensor (torch.Tensor): 3D tensor of probabilities (binary or continuous).
        operation (str): 'erosion' or 'dilation'.
        kernel_size (int): Size of the structuring element.
        threshold (float): Threshold to binarize the input tensor.
        num_passes (int): Number of times to apply the operation.
        device (str): 'cuda' or 'cpu'.
    
    Returns:
        torch.Tensor: Filtered 3D tensor.
    """
    assert operation in ['erosion', 'dilation', 'meanpool'], "Operation must be 'erosion' or 'dilation'."
    assert num_passes > 0, "num_passes must be a positive integer."
    
    # Create a 3D structuring element (kernel)
    struct_elem = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=device)
    # Threshold the input tensor (binarize) and add batch/channel dimensions
    binarized = (input_tensor > threshold).float().unsqueeze(1) # Shape: [N=1, C=1, D, H, W]

    for _ in range(num_passes):
        if operation == 'erosion':
            # Erosion: Use min pooling
            filtered = F.conv3d(binarized, struct_elem, padding=kernel_size // 2, stride=1)
            binarized = (filtered == struct_elem.numel()).float()
        elif operation == 'dilation':
            # Dilation: Use max pooling
            filtered = F.conv3d(binarized, struct_elem, padding=kernel_size // 2, stride=1)
            binarized = (filtered > 0).float()
        elif operation == 'meanpool':
            filtered = F.avg_pool3d(binarized, kernel_size, padding=kernel_size // 2, stride=1)
            binarized = (filtered > 0).float()
    
    # Remove batch and channel dimensions before returning
    return binarized.squeeze(1)
     
        

def find_connected_component(probability, threshold=[0.5,0.5,0.5,0.5,0.5], max_radius = 10):
    device = probability.device
    probability = probability.detach()
    num_particle_type, D, H, W = probability.shape
    mask = probability > torch.tensor(threshold,device=device).reshape(num_particle_type,1,1,1)

    # allocate the output tensors for labels
    out = (torch.arange(D * H * W, device=device, dtype=torch.float32)+1).reshape(1,D, H, W)
    out = out.repeat(num_particle_type,1,1,1)
    out[~mask] = 0

    out = out.reshape(num_particle_type,1,D, H, W)
    mask = mask.reshape(num_particle_type,1,D, H, W)
    for _ in range(max_radius):
        out = F.max_pool3d(out, kernel_size=3, stride=1, padding=1)
        out = torch.mul(out, mask)  # mask using element-wise multiplication
    out = out.reshape(num_particle_type,D,H,W)
    out = out.long()
    componet=[]
    for i in range(num_particle_type):
        u, inverse = torch.unique(out[i], sorted=True, return_inverse=True)
        componet.append(inverse)
    componet = torch.stack(componet)
    
    return componet
    
def find_centroid(componet, positions, cfg, tomogram):
    device = componet.device
    num_particle_type, D, H, W = componet.shape
    count = componet.flatten(1).max(-1)[0]+1
    cumcount = torch.zeros(num_particle_type+1, dtype=torch.int32, device=device)
    cumcount[1:] = torch.cumsum(count, 0)
    componet = componet+cumcount[:-1].reshape(num_particle_type,1,1,1)

    # gridz, gridy, gridx = torch.meshgrid([
    #     torch.arange(0,D,device=device),
    #     torch.arange(0,H,device=device),
    #     torch.arange(0,W,device=device),
    # ],indexing='ij')

    gridz = torch.arange(0, D, device=device).reshape(1,D,1,1).expand(num_particle_type,-1,H,W)
    gridy = torch.arange(0, H, device=device).reshape(1,1,H,1).expand(num_particle_type,D,-1,W)
    gridx = torch.arange(0, W, device=device).reshape(1,1,1,W).expand(num_particle_type,D,H,-1)
    n  = torch.bincount(componet.flatten())
    nx = torch.bincount(componet.flatten(),weights=gridx.flatten())
    ny = torch.bincount(componet.flatten(),weights=gridy.flatten())
    nz = torch.bincount(componet.flatten(),weights=gridz.flatten())
    
    leftTop_coords = (positions - (cfg.block_size // 2)) - cfg.pad_size
    leftTop_coords = torch.clamp(leftTop_coords, 0, None)

    x=nx/n #+ leftTop_coords[2]
    y=ny/n #+ leftTop_coords[1]
    z=nz/n #+ leftTop_coords[0]

    xyz = torch.stack([x,y,z],1).float()
    xyz = torch.split(xyz, count.tolist(), dim=0)
    centroids = [xxyyzz[1:] for xxyyzz in xyz]
    
    pred_final = []
    for particle_class in range(num_particle_type):
        particle_centroids = centroids[particle_class]
        
        for centroid in particle_centroids:
            if centroid[0] < cfg.pad_size or centroid[0] >= cfg.block_size - cfg.pad_size:
                continue
            if centroid[1] < cfg.pad_size or centroid[1] >= cfg.block_size - cfg.pad_size:
                continue
            if centroid[2] < cfg.pad_size or centroid[2] >= cfg.block_size - cfg.pad_size:
                continue
            
            located_centroid = {'z': centroid[2].item() + leftTop_coords[0].item(), 
                            'y': centroid[1].item() + leftTop_coords[1].item(), 
                            'x': centroid[0].item() + leftTop_coords[2].item(), 
                            'class': particle_class + 1,
                            'tomogram': tomogram}
                        
            pred_final.append(located_centroid)
                    
    return pred_final


def find_connected_component_with_confidence(probability, threshold=[0.5,0.5,0.5,0.5,0.5], max_radius=10):
    device = probability.device
    probability = probability.detach()
    B, num_particle_type, D, H, W = probability.shape
    mask = probability > torch.tensor(threshold, device=device).reshape(1, num_particle_type, 1, 1, 1)

    # Allocate the output tensors for labels
    out = (torch.arange(D * H * W, device=device, dtype=torch.float32) + 1).reshape(1, 1, D, H, W)
    out = out.repeat(B, num_particle_type, 1, 1, 1)
    out[~mask] = 0

    out = out.reshape(B * num_particle_type, 1, D, H, W)
    mask = mask.reshape(B * num_particle_type, 1, D, H, W)
    for _ in range(max_radius):
        out = F.max_pool3d(out, kernel_size=3, stride=1, padding=1)
        out = torch.mul(out, mask)
    out = out.reshape(B, num_particle_type, D, H, W)
    out = out.long()
    
    batched_component = []
    batched_confidences = []
    batched_volumes = []
    for b in range(B):
        increment = 0
        component = []
        confidences = []
        volume = []
        for i in range(num_particle_type):
            increment += 1
            u, inverse = torch.unique(out[b, i], sorted=True, return_inverse=True)
            component.append(inverse)
            #Calculate confidence for each unique component
            conf = {}
            for label in u:
                if label == 0:  # Skip background
                    continue
                mask = out[b, i] == label
                mean_confidence = probability[b, i][mask].mean().item()
                volume = mask.sum().item()
                conf[increment] = {"prob": mean_confidence, "volume": volume}
                increment += 1
            confidences.append(conf)
        component = torch.stack(component)
        
        batched_confidences.append(confidences)
        batched_component.append(component)
        
    component_out = torch.stack(batched_component)
    return component_out, batched_confidences


def find_centroid_with_confidence(components_batched, positions_batched, cfg, tomogram, confidences_batched):
    device = components_batched.device
    B, num_particle_type, D, H, W = components_batched.shape
    # Create coordinate grids for centroids
    gridz = torch.arange(0, D, device=device).reshape(1, D, 1, 1).expand(num_particle_type, -1, H, W)
    gridy = torch.arange(0, H, device=device).reshape(1, 1, H, 1).expand(num_particle_type, D, -1, W)
    gridx = torch.arange(0, W, device=device).reshape(1, 1, 1, W).expand(num_particle_type, D, H, -1)
    
    pred_final = []
    for b in range(B):
    # Offset connected component labels for unique indexing across particle types
        componet = components_batched[b]
        positions = positions_batched[b]
        confidences = confidences_batched[b]
                
        count = componet.flatten(1).max(-1)[0] + 1
        cumcount = torch.zeros(num_particle_type + 1, dtype=torch.int32, device=device)
        cumcount[1:] = torch.cumsum(count, 0)
        componet = componet + cumcount[:-1].reshape(num_particle_type, 1, 1, 1)
        

        # Compute the sums and voxel counts for centroids
        n = torch.bincount(componet.flatten())
        nx = torch.bincount(componet.flatten(), weights=gridx.flatten())
        ny = torch.bincount(componet.flatten(), weights=gridy.flatten())
        nz = torch.bincount(componet.flatten(), weights=gridz.flatten())

        # Compute global offsets
        leftTop_coords = (positions - (cfg.block_size // 2)) - cfg.pad_size

        # Calculate centroids
        x = nx / n
        y = ny / n
        z = nz / n
        xyz = torch.stack([x, y, z], 1).float()
        xyz = torch.split(xyz, count.tolist(), dim=0)
        centroids = [xxyyzz[1:] for xxyyzz in xyz]  # Ignore label 0 (background)

        for particle_class in range(num_particle_type):
            particle_centroids_per_class = centroids[particle_class]
            for label, centroid in enumerate(particle_centroids_per_class):
                
                # Apply discard range
                if centroid[0] <= cfg.discard_range or centroid[0] >= cfg.block_size - cfg.discard_range:
                    continue
                if centroid[1] <= cfg.discard_range or centroid[1] >= cfg.block_size - cfg.discard_range:
                    continue
                if centroid[2] <= cfg.discard_range or centroid[2] >= cfg.block_size - cfg.discard_range:
                    continue
                
                confidence = confidences[particle_class].get(label + cumcount[particle_class].item() + 1, 1)
                # Adjust for global coordinates
                located_centroid = {
                    'z': centroid[2].item() + leftTop_coords[0].item(),
                    'y': centroid[1].item() + leftTop_coords[1].item(),
                    'x': centroid[0].item() + leftTop_coords[2].item(),
                    'class': particle_class + 1,
                    'prob': confidence['prob'], #TODO REVERT
                    'volume': confidence['volume'],
                    'tomogram': tomogram
                }
                
                pred_final.append(located_centroid)

    return pred_final
