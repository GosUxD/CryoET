from types import SimpleNamespace
import os

cfg = SimpleNamespace(**{})

cfg.neptune_project = "oldscholar/cryoet"
cfg.neptune_connection_mode = "async"
cfg.logging = True
cfg.comment = ''

### DATAPATHS
cfg.data_dir = 'datamount/train/static'
cfg.labels_dir = 'datamount/train/truths'
cfg.output_dir = 'logs/output'
cfg.checkpoints_dir = 'logs/checkpoints'

### PARTICLES AND DICTIONARY STUFF
cfg.particles = ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle',]#  'beta-amylase']
cfg.particle_to_index = {'apo-ferritin': 1,'beta-galactosidase': 2,'ribosome': 3,'thyroglobulin': 4,'virus-like-particle': 5, 'beta-amylase': 6}
cfg.index_to_particle = {v:k for k, v in cfg.particle_to_index.items()}
cfg.particle_radius = {'apo-ferritin': 60, 'beta-galactosidase': 90,'ribosome': 150, 'thyroglobulin': 130, 'virus-like-particle': 135, 'beta-amylase': 65}


#cfg.particles = ['apo-ferritin', 'beta-galactosidase']
#cfg.particle_to_index = {'apo-ferritin': 1, 'beta-galactosidase': 2}#'thyroglobulin': 3, 'virus-like-particle': 4, 'beta-galactosidase': 2}
#cfg.index_to_particle = {v:k for k, v in cfg.particle_to_index.items()}

### DATASET STUFF
cfg.dataset = 'ds2'
cfg.eval = 'test'
cfg.n_folds = 7
cfg.fold_offset = 0
cfg.fold = 0
cfg.all_tomograms = ['TS_5_4', 'TS_6_4', 'TS_6_6', 'TS_69_2', 'TS_73_6', 'TS_86_3', 'TS_99_9']
# cfg.all_tomograms = ['TS_14', 'TS_0', 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'TS_5', 'TS_6', 'TS_7', 'TS_8', 'TS_9',
#                      'TS_10', 'TS_11', 'TS_12', 'TS_13',  'TS_15', 'TS_16', 'TS_17', 'TS_18', 'TS_19', 
#                      'TS_20', 'TS_21', 'TS_22', 'TS_23', 'TS_24', 'TS_25', 'TS_26']


### TRAINING PARAMETERS
cfg.epochs = 100
cfg.eval_epochs = 1
cfg.batch_size = 8
cfg.batch_size_eval = 16
cfg.num_workers = 4
cfg.pin_memory = True
cfg.lr = 1e-3
cfg.weight_decay = 0.05
cfg.clip_grad = 0.
cfg.track_norm = False
cfg.warmup = 0.5
cfg.grad_accumulation = 1
cfg.clip_grad = 0
cfg.test_use_pad = True
cfg.mixed_precision = True
cfg.compile = True
cfg.awp = False
cfg.awp_start = 5
cfg.swa_start = 70


### PROJECT PARAMETERS
cfg.seed = 1994
cfg.num_particles = len(cfg.particles)
cfg.num_classes = cfg.num_particles + 1
cfg.use_IP = True
cfg.use_coord = False
cfg.use_scSE = True
cfg.block_size = 128
cfg.shift = cfg.block_size // 2
cfg.random_blocks = True
cfg.random_block_prob = 1.0
cfg.use_mined = False

### MODEL PARAMETERS
cfg.model = 'cryo_unet' #'debug'
cfg.nms_kernel = 5
cfg.in_channels = 1
cfg.out_channels = cfg.num_classes
cfg.feature_maps = [24, 36, 48]
cfg.norm = 'in'
cfg.activation = 'relu'
cfg.use_attention = False
cfg.dropout_p = 0.2
cfg.pad_size = 15
cfg.discard_range = 15
cfg.step_size = 0

