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
cfg.postprocess = 'postprocess'

### PARTICLES AND DICTIONARY STUFF
cfg.particles = ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle',]#  'beta-amylase']
cfg.particle_to_index = {'apo-ferritin': 1,'beta-galactosidase': 2,'ribosome': 3,'thyroglobulin': 4,'virus-like-particle': 5, 'beta-amylase': 6}
cfg.index_to_particle = {v:k for k, v in cfg.particle_to_index.items()}
cfg.particle_radius = {'apo-ferritin': 60, 'beta-galactosidase': 90,'ribosome': 150, 'thyroglobulin': 130, 'virus-like-particle': 135, 'beta-amylase': 65}


#cfg.particles = ['apo-ferritin', 'beta-galactosidase']
#cfg.particle_to_index = {'apo-ferritin': 1, 'beta-galactosidase': 2}#'thyroglobulin': 3, 'virus-like-particle': 4, 'beta-galactosidase': 2}
#cfg.index_to_particle = {v:k for k, v in cfg.particle_to_index.items()}

### DATASET STUFF
cfg.dataset = 'ds_mcds'
cfg.eval = 'val'
cfg.n_folds = 7
cfg.fold_offset = 0
cfg.fold = 0
cfg.all_tomograms = ['TS_5_4', 'TS_6_4', 'TS_6_6', 'TS_69_2', 'TS_73_6', 'TS_86_3', 'TS_99_9']


### TRAINING PARAMETERS
cfg.epochs = 100
cfg.eval_epochs = 5
cfg.batch_size = 16
cfg.batch_size_eval = 16
cfg.num_workers = 4
cfg.pin_memory = True
cfg.lr = 5e-4
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
cfg.awp_start = 0


### PROJECT PARAMETERS
cfg.seed = 1994
cfg.num_particles = len(cfg.particles)
cfg.num_classes = cfg.num_particles + 1
cfg.use_IP = True
cfg.use_coord = True
cfg.use_scSE = True
cfg.block_size = 72
cfg.shift = cfg.block_size // 2
cfg.random_blocks = False
cfg.random_block_prob = 1.0

### MODEL PARAMETERS
cfg.model = 'mcds_3dunet' #'debug'
cfg.nms_kernel = 5
cfg.in_channels = 1
cfg.out_channels = cfg.num_classes
cfg.n_blocks = 4
cfg.start_filters = 32
cfg.norm = 'bn'
cfg.activation = 'relu'
cfg.use_attention = False
cfg.dropout_p = 0.0
cfg.pad_size = 12
cfg.discard_range = 12
