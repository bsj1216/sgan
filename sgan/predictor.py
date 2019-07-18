#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function call returns positions predicted by Social GAN
 
    IN: the multi-dimensional array (obs_len, num_agents, positions)
    OUT: the multi-dimensional array (pred_len, num_agents, positions)

Created on Fri Jun 21 16:52:59 2019

@author: sbae
"""
import torch
import sys
import numpy as np
from attrdict import AttrDict
import time

sys.path.insert(0,'/home/sbae/NNMPC.jl/python/nnmpc/sgan/')

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.utils import relative_to_abs


class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.generator = self.get_generator(self.model_path)
        
    def get_generator(self, path):
        paths = [path]
        for path in paths:
            checkpoint = torch.load(path)
            self.args = AttrDict(checkpoint['args'])
            return self.gen(checkpoint)        

    def gen(self, checkpoint):
        args = AttrDict(checkpoint['args'])
        generator = TrajectoryGenerator(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            decoder_h_dim=args.decoder_h_dim_g,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            noise_dim=args.noise_dim,
            noise_type=args.noise_type,
            noise_mix_type=args.noise_mix_type,
            pooling_type=args.pooling_type,
            pool_every_timestep=args.pool_every_timestep,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            neighborhood_size=args.neighborhood_size,
            grid_size=args.grid_size,
            batch_norm=args.batch_norm)
        generator.load_state_dict(checkpoint['g_state'])
        generator.cuda()
        generator.train()
        return generator


    def predict(self, data):
        # data example: 
    #    data = np.array([
    #       [ 1,  1.000e+00,  8.460e+00,  3.590e+00],
    #       [ 1,  2.000e+00,  1.364e+01,  5.800e+00],
    #       [ 2,  1.000e+00,  9.570e+00,  3.790e+00],
    #       [ 2,  2.000e+00,  1.364e+01,  5.800e+00],
    #       [ 3,  1.000e+00,  1.067e+01,  3.990e+00],
    #       [ 3,  2.000e+00,  1.364e+01,  5.800e+00],
    #       [ 4,  1.000e+00,  1.173e+01,  4.320e+00],
    #       [ 4,  2.000e+00,  1.209e+01,  5.750e+00],
    #       [ 5,  1.000e+00,  1.281e+01,  4.610e+00],
    #       [ 5,  2.000e+00,  1.137e+01,  5.800e+00],
    #       [ 6,  1.000e+00,  1.281e+01,  4.610e+00],
    #       [ 6,  2.000e+00,  1.031e+01,  5.970e+00],
    #       [ 7,  1.000e+00,  1.194e+01,  6.770e+00],
    #       [ 7,  2.000e+00,  9.570e+00,  6.240e+00],
    #       [ 8,  1.000e+00,  1.103e+01,  6.840e+00],
    #       [ 8,  2.000e+00,  8.730e+00,  6.340e+00]])
        data = np.array(data)
        dset, _ = data_loader(self.args, data)            
        pred_traj = self.run_generator(self.args, dset, self.generator)
        return pred_traj
    
        
    def run_generator(self, args, dset, generator):
#        start_time0 = time.time()
#        start_time = time.time()
#        batch = [batch for batch in loader][0]
#        print("time elapsed 0: {}".format(time.time() - start_time))
        obs_traj = dset.obs_traj.permute(2,0,1).cuda()
        obs_traj_rel = dset.obs_traj_rel.permute(2,0,1).cuda() 
        seq_start_end = torch.tensor(dset.seq_start_end).cuda() 
        
        with torch.no_grad():    
#            for batch in loader:
#            start_time = time.time()
#            batch = [tensor.cuda() for tensor in batch]
#            (obs_traj, _, obs_traj_rel, _,
#             _, _, seq_start_end) = batch
#            print("time elapsed 1: {}".format(time.time() - start_time))

#            start_time = time.time()
            pred_traj_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
#            print("generator time elapsed: {}".format(time.time() - start_time))
            
            pred_traj = relative_to_abs(
                pred_traj_rel, obs_traj[-1]
            )

#            print("time elapsed total: {}".format(time.time() - start_time0))
        
        return pred_traj[0,:,:].tolist()
    
    
    def predict_batch(self, obs_traj, obs_traj_rel, seq_start_end):
        obs_traj = np.array(obs_traj)
        obs_traj_rel = np.array(obs_traj_rel)
        seq_start_end = np.array(seq_start_end)

        obs_traj = torch.from_numpy(obs_traj).type(torch.float).cuda()
        obs_traj_rel = torch.from_numpy(obs_traj_rel).type(torch.float).cuda()
        seq_start_end = torch.from_numpy(seq_start_end).cuda()
        pred_traj = self.run_generator_batch(obs_traj, obs_traj_rel, seq_start_end, self.generator)
#        print("obs_traj size: {}".format(obs_traj.shape))
#        print("obs_traj_rel size: {}".format(obs_traj_rel.shape))
        return pred_traj
        

    def run_generator_batch(self, obs_traj, obs_traj_rel, seq_start_end, generator):
        with torch.no_grad():    
            pred_traj_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj = relative_to_abs(
                pred_traj_rel, obs_traj[-1]
            )
        
        return pred_traj[0,:,:].tolist()