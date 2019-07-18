#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function call returns positions predicted by Social GAN
 
    IN: the multi-dimensional array (obs_len, num_agents, positions)
    OUT: the multi-dimensional array (pred_len, num_agents, positions)

Created on Fri Jun 21 16:52:59 2019

@author: sbae
"""
import argparse
import os
import torch
import sys

from attrdict import AttrDict

sys.path.insert(0,'/home/sbae/sgan')

import numpy as np
import time

from attrdict import AttrDict

sys.path.insert(0,'/home/sbae/automotive-control-temporary/python/nnmpc/sgan/')


from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='../models/sgan-models/eth_8_model.pt', type=str)
parser.add_argument('--obs_traj', default=20, type=int)


def get_generator(checkpoint):
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


def main(args):
    # data example:

    data = np.array([
       [ 1,  1.000e+00,  8.460e+00,  3.590e+00],
       [ 1,  2.000e+00,  1.364e+01,  5.800e+00],
       [ 2,  1.000e+00,  9.570e+00,  3.790e+00],
       [ 2,  2.000e+00,  1.364e+01,  5.800e+00],
       [ 3,  1.000e+00,  1.067e+01,  3.990e+00],
       [ 3,  2.000e+00,  1.364e+01,  5.800e+00],
       [ 4,  1.000e+00,  1.173e+01,  4.320e+00],
       [ 4,  2.000e+00,  1.209e+01,  5.750e+00],
       [ 5,  1.000e+00,  1.281e+01,  4.610e+00],
       [ 5,  2.000e+00,  1.137e+01,  5.800e+00],
       [ 6,  1.000e+00,  1.281e+01,  4.610e+00],
       [ 6,  2.000e+00,  1.031e+01,  5.970e+00],
       [ 7,  1.000e+00,  1.194e+01,  6.770e+00],
       [ 7,  2.000e+00,  9.570e+00,  6.240e+00],
       [ 8,  1.000e+00,  1.103e+01,  6.840e+00],
       [ 8,  2.000e+00,  8.730e+00,  6.340e+00]])
    
    if os.path.isdir(args.model_path):
        raise Exception("model_path cannot be a directory")
    else:
        path = [args.model_path]
        
    
#    obs_traj = torch.tensor(args.obs_traj)
    checkpoint = torch.load(path)
    generator = get_generator(checkpoint)
    data = args.data
    _args = AttrDict(checkpoint['args'])
    _, loader = loader(_args, data)
    pred_traj = predict(_args, loader, generator)
        paths = [args.model_path]
    
    for path in paths:        
    #    obs_traj = torch.tensor(args.obs_traj)
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
#        data = args.data
        _args = AttrDict(checkpoint['args'])
        _, loader = data_loader(_args, data)
        start_time = time.time()
        pred_traj = predict(_args, loader, generator)
        print("time elapsed: {}".format(time.time() - start_time))
    
    print("pred_traj : {}".format(pred_traj))


def predict(args, loader, generator):
    # TODO: check fi
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, _, obs_traj_rel, _,
             _, _, seq_start_end) = batch

            pred_traj_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj = relative_to_abs(
                pred_traj_rel, obs_traj[-1]
            )
    
    return pred_traj[0,:,:].tolist()

#    with torch.no_grad():
    # for batch in loader:
    #     batch = [tensor.cuda() for tensor in batch]
    #     (obs_traj, _, obs_traj_rel, _,
    #      _, _, seq_start_end) = batch

    #     pred_traj_rel = generator(
    #         obs_traj, obs_traj_rel, seq_start_end
    #     )
    #     pred_traj = relative_to_abs(
    #         pred_traj_rel, obs_traj[-1]
    #     )
    
    # return pred_traj[0,:,:].tolist()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
