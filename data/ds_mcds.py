from  torch.utils.data import Dataset
from utils.utils import load_tomogram, load_labels, normalize, z_score_normalize, load_coordinates_json
from utils.utils import load_tomogram_all_types
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
import torch.nn.functional as F
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
from utils.augments import bandpass_filter, frequency_noise, missing_wedge, band_dropout, randomize_phase 

def prepare_tomograms(cfg, mode):
    tomograms = []
    if mode == 'train':
        for idx, tomogram_name in enumerate(cfg.train_tomograms):
            tomograms.append(z_score_normalize(np.ascontiguousarray(load_tomogram(tomogram_name))))
    elif mode == 'val':
        for idx, tomogram_name in enumerate(cfg.val_tomograms):
            tomograms.append(z_score_normalize(np.ascontiguousarray(load_tomogram(tomogram_name))))
            
    elif mode == 'test':
        for idx, tomogram_name in enumerate(cfg.test_tomograms):
            tomogram = z_score_normalize(np.ascontiguousarray(load_tomogram(tomogram_name)))
            shape = tomogram.shape
            padded_shape = [i + 2 * cfg.pad_size for i in shape]
            
            temp = np.zeros(padded_shape)
            temp[cfg.pad_size:padded_shape[0] - cfg.pad_size,
                cfg.pad_size:padded_shape[1] - cfg.pad_size,
                cfg.pad_size:padded_shape[2] - cfg.pad_size] = tomogram
            tomograms.append(temp)
    
    return tomograms

def prepare_denoised(cfg, mode):
    denoise_tomograms = []
    if mode == 'train':
        for idx, tomogram_name in enumerate(cfg.train_tomograms):
            denoise_tomograms.append(z_score_normalize(np.ascontiguousarray(load_tomogram(tomogram_name, type='isonetcorrected'))))
    elif mode == 'val':
        for idx, tomogram_name in enumerate(cfg.val_tomograms):
            denoise_tomograms.append(z_score_normalize(np.ascontiguousarray(load_tomogram(tomogram_name, type='isonetcorrected'))))

    elif mode == 'test':
        for idx, tomogram_name in enumerate(cfg.test_tomograms):
            tomogram = z_score_normalize(np.ascontiguousarray(load_tomogram(tomogram_name, type='isonetcorrected')))
            shape = tomogram.shape
            padded_shape = [i + 2 * cfg.pad_size for i in shape]
            
            temp = np.zeros(padded_shape)
            temp[cfg.pad_size:padded_shape[0] - cfg.pad_size,
                cfg.pad_size:padded_shape[1] - cfg.pad_size,
                cfg.pad_size:padded_shape[2] - cfg.pad_size] = tomogram
            denoise_tomograms.append(temp)
    return denoise_tomograms

def prepare_labels(cfg, mode):
    labels = []
    if mode == 'train':
        for tomogram in cfg.train_tomograms:
            labels.append(np.ascontiguousarray(load_labels(tomogram)))
    elif mode == 'val':
        for tomogram in cfg.val_tomograms:
            labels.append(np.ascontiguousarray(load_labels(tomogram)))
    elif mode == 'test':
        for tomogram in cfg.test_tomograms:
            label = np.ascontiguousarray(load_labels(tomogram))
            shape = label.shape
            padded_shape = [i + 2 * cfg.pad_size for i in shape]
            
            temp = np.zeros(padded_shape)
            temp[cfg.pad_size:padded_shape[0] - cfg.pad_size,
                cfg.pad_size:padded_shape[1] - cfg.pad_size,
                cfg.pad_size:padded_shape[2] - cfg.pad_size] = label   
            labels.append(temp)
            
    return labels

def prepare_datapoints(cfg, mode, tomograms=None):
    extracted_datapoints = []  
    if mode == 'train' or mode == 'val':      
        for idx, tomogram in enumerate(cfg.train_tomograms) if mode == 'train' else enumerate(cfg.val_tomograms):
            for particle in cfg.particles:
                datapoints = load_coordinates_json(tomogram, particle)
                for datapoint in datapoints['points']:
                    point = {"x": datapoint['location']['x'], 
                                "y": datapoint['location']['y'], 
                                "z": datapoint['location']['z'], 
                                "tomogram": idx, 
                                "particle": particle,
                                "radius": cfg.particle_radius[particle]}
                    extracted_datapoints.append(point)
                    
                    if cfg.random_blocks and np.random.rand() < cfg.random_block_prob:
                        x = np.random.randint(cfg.shift, 630 - cfg.shift) * 10
                        y = np.random.randint(cfg.shift, 630 - cfg.shift) * 10
                        z = np.random.randint(cfg.shift, 184 - cfg.shift) * 10
                        point = {"x": x, "y": y, "z": z, "tomogram": idx, "radius": 10}
                        extracted_datapoints.append(point)
            
            if cfg.use_mined:            
                mined_coordinates = np.load(f'datamount/train/mined/{tomogram}/mined.npy')
                for mined in mined_coordinates:
                    point = {"x": mined[0], "y": mined[1], "z": mined[2], "tomogram": idx, "radius": 10}
                    extracted_datapoints.append(point)
            
            
        return extracted_datapoints
    
    elif mode == 'test':
        pad_size = cfg.pad_size
        block_size = cfg.block_size
        if cfg.test_use_pad:
            step_size = block_size - 2 * pad_size
        else:
            step_size = int(cfg.shift * 2)
        
        for idx, tomogram in enumerate(tomograms):
            shape = tomogram.shape
            for j in range((shape[0] - 2 * pad_size) // step_size + (
                1 if (shape[0] - 2 * pad_size) % step_size > 0 else 0)):
                for k in range((shape[1] - 2 * pad_size) // step_size + (
                        1 if (shape[1] - 2 * pad_size) % step_size > 0 else 0)):
                    for l in range((shape[2] - 2 * pad_size) // step_size + (
                            1 if (shape[2] - 2 * pad_size) % step_size > 0 else 0)):
                        if j == (shape[0] - 2 * pad_size) // step_size + (
                                1 if (shape[0] - 2 * pad_size) % step_size > 0 else 0) - 1:
                            z = shape[0] - block_size // 2
                        else:
                            z = j * step_size + block_size // 2

                        if k == (shape[1] - 2 * pad_size) // step_size + (
                                1 if (shape[1] - 2 * pad_size) % step_size > 0 else 0) - 1:
                            y = shape[1] - block_size // 2
                        else:
                            y = k * step_size + block_size // 2

                        if l == (shape[2] - 2 * pad_size) // step_size + (
                                1 if (shape[2] - 2 * pad_size) % step_size > 0 else 0) - 1:
                            x = shape[2] - block_size // 2
                        else:
                            x = l * step_size + block_size // 2
                            
                        point = {"x": x, 
                                "y": y, 
                                "z": z,
                                "tomogram": idx}
                        extracted_datapoints.append(point)
        return extracted_datapoints
        

def batch_to_device(batch, device):   
    batch_dict = {key: batch[key].to(device) for key in batch if key != 'tomogram'}
    batch_dict['tomogram'] = batch['tomogram']
    return batch_dict        


class CryoDataset(Dataset):
    def __init__(self, cfg, mode = 'train'):
        self.mode = mode
        self.cfg = cfg
        self.shift = cfg.shift
        self.tomograms = prepare_tomograms(cfg, mode)
        self.denoise_tomograms = prepare_denoised(cfg, mode)
        self.labels = prepare_labels(cfg, mode)
        self.datapoints = prepare_datapoints(cfg, mode, self.tomograms)

                
        patch_size = [cfg.block_size] * 3
        self.st = SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.05),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.95, 1.05),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=0, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.99, p_scale_per_sample=0.1
        )
        self.mt = MirrorTransform(axes=(0, 1, 2))

                    
                    
    def __getitem__(self, index):
        datapoint = self.datapoints[index]
        z_max, y_max, x_max = self.tomograms[datapoint['tomogram']].shape
        
        
        if self.mode == 'train' or self.mode == 'val':
            position = ([int(datapoint['z']//10), int(datapoint['y']//10), int(datapoint['x']//10)])
            point = self.__sample(position, np.array([z_max, y_max, x_max]),
                                    radius=int(datapoint['radius']/10))
        else:
            position = ([int(datapoint['z']), int(datapoint['y']), int(datapoint['x'])])
            point = np.array([position[0], position[1], position[2]]) 
               
        if self.mode == 'train':
            img = self.tomograms[datapoint['tomogram']]
        elif self.mode == 'val' or self.mode == 'test':
            img = self.tomograms[datapoint['tomogram']]
            
        img = img[point[0] - self.shift:point[0] + self.shift,
                      point[1] - self.shift:point[1] + self.shift,
                      point[2] - self.shift:point[2] + self.shift]
        
        
        label = self.labels[datapoint['tomogram']]
        label = label[point[0] - self.shift:point[0] + self.shift,
                      point[1] - self.shift:point[1] + self.shift,
                      point[2] - self.shift:point[2] + self.shift]
        
        denoise = self.denoise_tomograms[datapoint['tomogram']]
        denoise = denoise[point[0] - self.shift:point[0] + self.shift,
                                        point[1] - self.shift:point[1] + self.shift,
                                        point[2] - self.shift:point[2] + self.shift]

        img = torch.tensor(img).float()
        denoise = torch.tensor(denoise).float()
        # fourier_augment_prob = torch.rand(1)
        
        # if self.mode == 'train':
        #     if fourier_augment_prob < 0.2:
        #         img = bandpass_filter(img)
        #     elif fourier_augment_prob < 0.4:
        #         img = frequency_noise(img)
        #     elif fourier_augment_prob < 0.6:
        #         img = missing_wedge(img)
        #     elif fourier_augment_prob < 0.8:
        #         img = band_dropout(img)
        #     elif fourier_augment_prob < 1.0:
        #         img = randomize_phase(img)
        
        
        img = img.numpy().astype(np.float32)
        label = label.astype(np.int64)
        denoise = denoise.numpy().astype(np.float32)
       
        img_label = {
            'data': img.reshape(1, -1, self.shift * 2, self.shift * 2, self.shift * 2),
            'seg': label.reshape(1, -1, self.shift * 2, self.shift * 2, self.shift * 2),
            'denoise': denoise.reshape(1, -1, self.shift * 2, self.shift * 2, self.shift * 2)
            }
        
        if self.mode == 'train':
            if torch.rand(1) < 0.5:
                img_label = self.st(**img_label)  # Apply spatial transform
            else:
                img_label = self.mt(**img_label)  # Apply mirror transform
        
        img = torch.tensor(img_label['data']).float().reshape(-1, self.shift * 2, self.shift * 2, self.shift * 2)
        label = torch.tensor(img_label['seg']).long().reshape(self.shift * 2, self.shift * 2, self.shift * 2)
        denoise = torch.tensor(img_label['denoise']).float().reshape(-1, self.shift * 2, self.shift * 2, self.shift * 2)

        position = torch.tensor(point).float()
        tomogram_name = self.cfg.train_tomograms[datapoint['tomogram']] if self.mode == 'train' else self.cfg.test_tomograms[datapoint['tomogram']]    
        return {
            'image': img, 
            'label': label,
            'denoise': denoise,
            'position':position,
            'tomogram': tomogram_name}
    
    
    def __len__(self):
        return len(self.datapoints)
        
    def __sample(self, point, bound, radius=6):
        # point: z, y, x
        if self.mode == 'train':
            shift_amount = self.shift - radius - self.cfg.pad_size # 6 is arbitrary distance from border
            new_point = point + np.random.randint(-shift_amount,  shift_amount, size=3)
        else:
            new_point = np.array([point[0], point[1], point[2]])
        new_point[new_point < self.shift] = self.shift
        new_point[new_point + self.shift > bound] = bound[new_point + self.shift > bound] - self.shift
        return new_point


    def reconv_centers_update_datapoints(self, candidate_centroids):
        points_to_revisit = []
        for centroid in candidate_centroids:
            shape = self.tomograms[self.cfg.test_tomograms.index(centroid['tomogram'])].shape
            bound = np.array([shape[0], shape[1], shape[2]])
            point = np.array([int(np.round(centroid['z'])), int(np.round(centroid['y'])), int(np.round(centroid['x']))])
            
            point[point < self.shift] = self.shift
            point[point + self.shift > bound] = bound[point + self.shift > bound] - self.shift
            
            point = {"z": point[0],
                     "y": point[1], 
                     "x": point[2], 
                     "tomogram": self.cfg.test_tomograms.index(centroid['tomogram'])}
            
            points_to_revisit.append(point)  
            
        self.datapoints = points_to_revisit