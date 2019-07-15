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
    data = [[840.  ,   2.  ,   9.57,   6.24],
       [840.  ,   3.  ,  11.94,   6.77],
       [850.  ,   2.  ,   8.73,   6.34],
       [850.  ,   3.  ,  11.03,   6.84],
       [850.  ,   4.  ,  -1.32,   5.11],
       [850.  ,   5.  ,  -1.48,   4.43],
       [850.  ,   6.  ,  11.84,   5.82],
       [860.  ,   2.  ,   7.94,   6.5 ]]
    
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

