import sys
import importlib.util
import numpy as np
import os
import matplotlib.pyplot as plt
import zarr

spec = importlib.util.spec_from_file_location("utils", os.path.join(os.path.dirname(__file__), '../utils/utils.py'))
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

spec2 = importlib.util.spec_from_file_location("cfg", os.path.join(os.path.dirname(__file__), '../configs/cfg.py'))
cfg = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(cfg)
cfg = cfg.cfg


def generate_labels(tomogram_name):
    tomogram = utils.load_tomogram(tomogram_name)    
    label = np.zeros(tomogram.shape, dtype=np.float32)
    z_max, y_max, x_max = tomogram.shape    
    
    for particle in cfg.particles:
        print(f'Generating labels for {particle} in {tomogram_name}')
        coordinates = utils.load_coordinates_xyz(tomogram_name, particle)
        for idx, coordinate in enumerate(coordinates):
            cls_idx, x, y, z = cfg.particle_to_index[particle], int(coordinate['x'] // 10), int(coordinate['y'] // 10), int(coordinate['z'] // 10)
            #r = int(np.ceil(cfg.particle_radius[particle]/10))
            if particle == 'apo-ferritin':
                r = int((cfg.particle_radius[particle]//10) * 0.75)
            elif particle == 'beta-galactosidase':
                r = int((cfg.particle_radius[particle]//10) * 0.75)
            elif particle == 'ribosome':
                r = int((cfg.particle_radius[particle]//10) * 0.75)
            elif particle == 'thyroglobulin':
                r = int((cfg.particle_radius[particle]//10) * 0.75)
            elif particle == 'virus-like-particle':
                r = int((cfg.particle_radius[particle]//10) * 0.75)
            dim = 2 * r + 1
            template = utils.gaussian3D((dim, dim, dim), dim)
               
            z_start = max(0, z - r)
            z_end =  min(z_max, z + r + 1)
            y_start =  max(0, y - r)
            y_end =  min(y_max, y + r + 1)
            x_start =  max(0, x - r)
            x_end =  min(x_max, x + r + 1)
            
            t_z_start = (r - z) if z - r < 0 else 0
            t_z_end = (r + z_max - z) if z + r + 1 > z_max else 2 * r + 1
            t_y_start = (r - y) if y - r < 0 else 0
            t_y_end = (r + y_max - y) if y + r + 1 > y_max else 2 * r + 1
            t_x_start = (r - x) if x - r < 0 else 0
            t_x_end = (r + x_max - x) if x + r + 1  > x_max else 2 * r + 1
            
            tmp1 = label[z_start:z_end, y_start:y_end, x_start:x_end]
            tmp2 = template[t_z_start:t_z_end, t_y_start:t_y_end, t_x_start:t_x_end]

            larger_index = tmp1 < tmp2
            tmp1[larger_index] = tmp2[larger_index]
            
            tg = 0.60653
            
            tmp1[tmp1 <= tg] = 0
            tmp1 = np.where(tmp1 > 0, cls_idx, 0)
            
            
            label[z_start:z_end, y_start:y_end, x_start:x_end] = tmp1
     
    np.save('/home/daniel/AI/Projects/3.CryoET/1.CompetitionModel/datamount/train/truths/' + tomogram_name, label)       
                    
                    
for tomogram in cfg.all_tomograms:
    generate_labels(tomogram)
