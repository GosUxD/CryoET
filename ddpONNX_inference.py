from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics.loss import seg_metrics, score
from torch.amp import autocast 
from utils.utils import set_seed, prepare_submission_df, prepare_truths_df, de_dup, merge_points_by_confidence
#from postprocess.postprocess import find_connected_component_with_confidence, find_centroid_with_confidence
import matplotlib.pyplot as plt
from decouple import config
import numpy as np
import torch
import importlib
import argparse
import sys
from copy import copy
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import tensorrt as trt
import pycuda.driver as cuda


def ddp_setup(rank, world_size):
    #os.environ['MASTER_ADDR'] = "localhost"
    #os.environ['MASTER_PORT'] = "12355"
    print(f"Rank: {rank}, World Size: {world_size}")
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


def inference_fn(rank: int, world_size: int):
   
    torch.cuda.set_device(rank)
    device = cuda.Device(rank)
    cuda_context = device.make_context()
    cuda_context.push()

    def load_engine(trt_file_path: str):
        """
        Loads a serialized TensorRT engine (.plan) from disk and returns the engine object.
        """
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(trt_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def allocate_buffers(engine, batch_size=16):
        stream = cuda.Stream()
        input_shape = (batch_size, 1, 72, 72, 72)   # B=16
        output_shape = (batch_size, 6, 72, 72, 72)  # B=16
        
        # Compute buffer sizes
        input_size = int(np.prod(input_shape))      # Force to Python int
        output_size = int(np.prod(output_shape))    # Force to Python int

        h_input  = cuda.pagelocked_empty(input_size,  np.float32)
        h_output = cuda.pagelocked_empty(output_size, np.float32)

        # Allocate device memory
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        # Create list of bindings
        bindings = [int(d_input), int(d_output)]
        
        return {
            "stream": stream,
            "h_input": h_input,
            "h_output": h_output,
            "d_input": d_input,
            "d_output": d_output,
            "bindings": bindings,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "engine": engine
        }

    def do_inference(context, buffers, batch_data, cuda_context):
        cuda_context.push()
        stream   = buffers["stream"]
        h_input  = buffers["h_input"]   # shape = [16*1*72*72*72] floats
        h_output = buffers["h_output"]  # shape = [16*6*72*72*72] floats
        d_input  = buffers["d_input"]
        d_output = buffers["d_output"]

        B = batch_data.shape[0]
        current_input_size = B * 1 * 72 * 72 * 72  # just the portion for this batch

        context.set_input_shape("input", batch_data.shape)  # e.g. (B,1,72,72,72)

        np.copyto(h_input[:current_input_size], batch_data.ravel())
        cuda.memcpy_htod_async(d_input, h_input[:current_input_size], stream)

        context.set_tensor_address("input",  int(d_input))
        context.set_tensor_address("output", int(d_output))
        context.execute_async_v3(stream_handle=stream.handle)

        out_shape = context.get_tensor_shape("output")  # e.g. (B,6,72,72,72)
        out_size  = np.prod(out_shape)

        cuda.memcpy_dtoh_async(
            h_output,
            d_output,
            stream=stream
        )
        stream.synchronize()
        cuda_context.pop()
        return h_output[:out_size].reshape(out_shape)
        
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
    print(f"Seed : {cfg.seed}")

    CryoDataset = importlib.import_module(cfg.dataset).CryoDataset
    postprocess = importlib.import_module(cfg.postprocess)
    
    engine = load_engine(f"logs/output/CRYOET420_fold{fold}.plan")
    context = engine.create_execution_context()
    engine2 = load_engine(f"logs/output/CRYOET271_fold{fold}.plan")
    context2 = engine2.create_execution_context()
    engine3 = load_engine(f"logs/output/CRYOET364_fold{fold}.plan")
    context3 = engine3.create_execution_context()
    
    buffers = allocate_buffers(engine, batch_size=16)  
          
    beta_scores = []
    for fold in range(cfg.n_folds):
        fold = fold + cfg.fold_offset
        cfg.fold = fold
        
        new_group = torch.distributed.new_group(ranks=list(range(world_size)))
        print(f"Fold: {cfg.fold}")
        
        cfg.val_tomograms = [cfg.all_tomograms[fold]]
        cfg.test_tomograms = [cfg.all_tomograms[fold]]
        print(f"Val Tomograms: {cfg.val_tomograms}")
        print(f"Test Tomograms: {cfg.test_tomograms}")

        test_dataset = CryoDataset(cfg, 'test')
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        test_dataloader = DataLoader(
                test_dataset,
                shuffle=False,
                sampler=test_sampler,
                drop_last=False,
                num_workers=cfg.num_workers,
                batch_size=cfg.batch_size_eval,
                pin_memory=True,
            )
        
        
        apo_ferritin, beta_galactosidase, ribosome, thyroglobulin, virus_like_particle = [], [], [], [], []
        torch.set_grad_enabled(False)    
        for index, data in enumerate(tqdm(test_dataloader)):
                input_tensor = data['image'].cpu().numpy() # shape = [B,1,72,72,72]
                img_flip = np.flip(input_tensor, [-1])
                img_flip2 = np.flip(input_tensor, [-2])
                img_flip3 = np.flip(input_tensor, [-3])
                img_flip12 = np.flip(input_tensor, [-1, -2])
                img_flip13 = np.flip(input_tensor, [-1, -3])
                img_flip23 = np.flip(input_tensor, [-2, -3])
                
                seg_output = do_inference(context, buffers, input_tensor)  # shape [B,6,72,72,72]
                seg_output = torch.from_numpy(seg_output).to(cfg.device)
                seg_output = torch.nn.functional.softmax(seg_output, dim=1)
                
                seg_flip = do_inference(context, buffers, img_flip)  # shape [B,6,72,72,72]
                seg_flip = torch.from_numpy(seg_flip).to(cfg.device)
                seg_flip = torch.flip(seg_flip, [-1])
                seg_flip = torch.nn.functional.softmax(seg_flip, dim=1)
                
                seg_flip2 = do_inference(context, buffers, img_flip2)  # shape [B,6,72,72,72]
                seg_flip2 = torch.from_numpy(seg_flip2).to(cfg.device)
                seg_flip2 = torch.flip(seg_flip2, [-2])
                seg_flip2 = torch.nn.functional.softmax(seg_flip2, dim=1)
                
                seg_flip3 = do_inference(context, buffers, img_flip3)  # shape [B,6,72,72,72]
                seg_flip3 = torch.from_numpy(seg_flip3).to(cfg.device)
                seg_flip3 = torch.flip(seg_flip3, [-3])
                seg_flip3 = torch.nn.functional.softmax(seg_flip3, dim=1)
                
                seg_flip12 = do_inference(context, buffers, img_flip12)  # shape [B,6,72,72,72]
                seg_flip12 = torch.from_numpy(seg_flip12).to(cfg.device)
                seg_flip12 = torch.flip(seg_flip12, [-1, -2])
                seg_flip12 = torch.nn.functional.softmax(seg_flip12, dim=1)
                
                seg_flip13 = do_inference(context, buffers, img_flip13)  # shape [B,6,72,72,72]
                seg_flip13 = torch.from_numpy(seg_flip13).to(cfg.device)
                seg_flip13 = torch.flip(seg_flip13, [-1, -3])
                seg_flip13 = torch.nn.functional.softmax(seg_flip13, dim=1)
                
                seg_flip23 = do_inference(context, buffers, img_flip23)  # shape [B,6,72,72,72]
                seg_flip23 = torch.from_numpy(seg_flip23).to(cfg.device)
                seg_flip23 = torch.flip(seg_flip23, [-2, -3])
                seg_flip23 = torch.nn.functional.softmax(seg_flip23, dim=1)
                
                seg2_output = do_inference(context2, buffers, input_tensor)  # shape [B,6,72,72,72]
                seg2_output = torch.from_numpy(seg2_output).to(cfg.device)
                seg2_output = torch.nn.functional.softmax(seg2_output, dim=1)
                
                seg3_output = do_inference(context3, buffers, input_tensor)  # shape [B,6,72,72,72]
                seg3_output = torch.from_numpy(seg3_output).to(cfg.device)
                seg3_output = torch.nn.functional.softmax(seg3_output, dim=1)
                
                seg3_flip = do_inference(context3, buffers, img_flip)  # shape [B,6,72,72,72]
                seg3_flip = torch.from_numpy(seg3_flip).to(cfg.device)
                seg3_flip = torch.flip(seg3_flip, [-1])
                seg3_flip = torch.nn.functional.softmax(seg3_flip, dim=1)
                
                seg3_flip2 = do_inference(context3, buffers, img_flip2)  # shape [B,6,72,72,72]
                seg3_flip2 = torch.from_numpy(seg3_flip2).to(cfg.device)
                seg3_flip2 = torch.flip(seg3_flip2, [-2])
                seg3_flip2 = torch.nn.functional.softmax(seg3_flip2, dim=1)
                
                seg3_flip3 = do_inference(context3, buffers, img_flip3)  # shape [B,6,72,72,72]
                seg3_flip3 = torch.from_numpy(seg3_flip3).to(cfg.device)
                seg3_flip3 = torch.flip(seg3_flip3, [-3])
                seg3_flip3 = torch.nn.functional.softmax(seg3_flip3, dim=1)
                
                seg3_flip12 = do_inference(context3, buffers, img_flip12)  # shape [B,6,72,72,72]
                seg3_flip12 = torch.from_numpy(seg3_flip12).to(cfg.device)
                seg3_flip12 = torch.flip(seg3_flip12, [-1, -2])
                seg3_flip12 = torch.nn.functional.softmax(seg3_flip12, dim=1)
                
                seg3_flip13 = do_inference(context3, buffers, img_flip13)  # shape [B,6,72,72,72]
                seg3_flip13 = torch.from_numpy(seg3_flip13).to(cfg.device)
                seg3_flip13 = torch.flip(seg3_flip13, [-1, -3])
                seg3_flip13 = torch.nn.functional.softmax(seg3_flip13, dim=1)
                
                seg3_flip23 = do_inference(context3, buffers, img_flip23)  # shape [B,6,72,72,72]
                seg3_flip23 = torch.from_numpy(seg3_flip23).to(cfg.device)
                seg3_flip23 = torch.flip(seg3_flip23, [-2, -3])
                seg3_flip23 = torch.nn.functional.softmax(seg3_flip23, dim=1)
                
                seg_output = seg_output + seg_flip + seg_flip2 + seg_flip3 + \
                            seg_flip12 + seg_flip13 + seg_flip23 + seg3_output + \
                            seg3_flip + seg3_flip2 + seg3_flip3 + seg3_flip12 + seg3_flip13 + seg3_flip23

                seg_output = (seg_output / 14) * 0.75 + seg2_output * 0.25

                
                tomogram = data['tomogram'][0]
                component, confidences = postprocess.find_connected_component_with_confidence(seg_output[:, 1:], 
                                                    threshold=[0.35, 0.3, 0.5, 0.3, 0.5], max_radius=25)
                centroids = postprocess.find_centroid_with_confidence(component, data['position'], cfg=cfg, tomogram=tomogram, confidences_batched=confidences)
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
                                 
        dist.barrier(new_group)
                        
        all_apo_ferritin = [None for _ in range(world_size)]
        all_beta_galactosidase = [None for _ in range(world_size)]
        all_ribosome = [None for _ in range(world_size)]
        all_thyroglobulin = [None for _ in range(world_size)]
        all_virus_like_particle = [None for _ in range(world_size)]

        if rank == 0:
            dist.gather_object(apo_ferritin, all_apo_ferritin)
            dist.gather_object(beta_galactosidase, all_beta_galactosidase)
            dist.gather_object(ribosome, all_ribosome)
            dist.gather_object(thyroglobulin, all_thyroglobulin)
            dist.gather_object(virus_like_particle, all_virus_like_particle)
        
        else:
            dist.gather_object(apo_ferritin)
            dist.gather_object(beta_galactosidase)
            dist.gather_object(ribosome)
            dist.gather_object(thyroglobulin)
            dist.gather_object(virus_like_particle)
        
        if rank == 0:
            merged_apo_ferritin = []
            for centroids_list in all_apo_ferritin:
                merged_apo_ferritin.extend(centroids_list)
            
            merged_beta_galactosidase = []
            for centroids_list in all_beta_galactosidase:
                merged_beta_galactosidase.extend(centroids_list)
            
            merged_ribosome = []
            for centroids_list in all_ribosome:
                merged_ribosome.extend(centroids_list)
            
            merged_thyroglobulin = []
            for centroids_list in all_thyroglobulin:
                merged_thyroglobulin.extend(centroids_list)
            
            merged_virus_like_particle = []
            for centroids_list in all_virus_like_particle:
                merged_virus_like_particle.extend(centroids_list)
    
            
            apo_ferritin_deduped = merge_points_by_confidence(merged_apo_ferritin, 4, 1, tomogram)
            beta_galactosidase_deduped = merge_points_by_confidence(merged_beta_galactosidase, 6, 2, tomogram)
            ribosome_deduped = merge_points_by_confidence(merged_ribosome, 12, 3, tomogram)
            thyroglobulin_deduped = merge_points_by_confidence(merged_thyroglobulin, 10, 4, tomogram)
            virus_like_particle_deduped = merge_points_by_confidence(merged_virus_like_particle, 9, 5, tomogram)
            
            
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
        destroy_process_group(new_group)
            
            
    cuda_context.pop()
    del context
    del buffers
    del engine
    
    if rank == 0:
        print(f'Beta Scores: {beta_scores}')
        print(f'Mean Beta Score: *****{np.mean(beta_scores)}*****')
        
def main(rank :int, world_size :int):
    ddp_setup(rank, world_size)
    inference_fn(rank, world_size)
    destroy_process_group()

if __name__ == "__main__":
    rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.cuda.device_count()
    
    main(rank, world_size)


