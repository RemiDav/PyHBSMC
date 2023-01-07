# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import torch
import pandas as pd
import sys
import os
import time
import multiprocessing as mp

from importlib import reload
# %% Set working directory and import SOI module
os.chdir(r'D:\Dropbox\Dev\DiscreteChoiceSMC\PyHBSMC2.0\Example')
sys.path.append(r'D:\Dropbox\Dev\DiscreteChoiceSMC\PyHBSMC2.0')
import HBSMC
# %% Start Algorithm

# %% Start Algorithm
os.chdir(os.path.dirname(os.path.abspath( __file__ )))
if __name__ == "__main__":
    
    # %% Create algorithm options dict.
    opts = {}
    opts['G'] = 2 # Number of particles groups
    opts['P'] = 256 # Number of particles per group
    opts['tag'] = 'PyHBSMC-Model1' # Added to saved files' filename, e.g. Results-MyTag.csv
    opts['save_path'] = '.' + os.sep
    opts['adaptive'] = True # Use adaptive version of SMC
    opts['ress_threshold'] = 0.8 # If adaptive, min ress to skip resampling and mutation
    
    # Parallel options
    opts['parallel'] = True

    
    # Mutate options
    opts['smc_m_steps'] = 10 # number of moves per mutation
    opts['min_accept_ratio'] = 0.1 # Min accept ratio in a given mutation (redo mutation if below)
    opts['target_accept_ratio'] = 0.15 # Redo mutation until sum(accept ratio) > target
    opts['max_mutate_loop'] = 5 # Max number of redone mutation step
    opts['precision_base'] = 16 # change of step size if mutation is redone
    opts['hmc_eps'] = 0.01
    opts['hmc_leaps'] = 10
    
    # Model option
    opts['m_opts'] = {}
    opts['m_opts']['name'] = 'ProbitExample' # Model name
    # Prior definition
    opts['m_opts']['sub_gam'] = torch.tensor(np.atleast_2d( [[1,1],[1,1]]).transpose(), dtype=torch.double) # One prior per column: [[shapes],[scales]]
    opts['m_opts']['sub_bet'] = torch.tensor(np.atleast_2d( [[1,1],[1,1]] ).transpose(), dtype=torch.double)  # One prior per column
    opts['m_opts']['sub_uvn'] = torch.tensor(np.atleast_2d( [[0.,1.],[0.,1.]] ).transpose(), dtype=torch.double)  # One prior per column
    opts['m_opts']['sub_mvn'] = [None] * 0 # == /!\ Not implemented yet ==
    
    # Prior boundaries
    # Make sure that the boundaries are at most the domain of the distribution
    opts['m_opts']['gam_bound'] = np.atleast_2d( [[0,np.inf],[0,np.inf]]) # one row per parameter: [low bound,high bound]
    opts['m_opts']['bet_bound'] = np.atleast_2d( [[0,1],[0,1]]) # one row per parameter: [low bound,high bound]
    opts['m_opts']['uvn_bound'] = np.atleast_2d( [[-np.inf,np.inf],[-np.inf,np.inf]]) # one row per parameter: [low bound,high bound]
    
    
    # %% Load / Generate Data for one individual
    # model: y = argmax Sum (B+g*e) * x^a + s * r
    # B ~ UVN
    # g,s ~ Gam
    # a ~ Bet
    # noise: eps,r ~ uvn(0.,1.)
    num_obs = 200
    num_alt = 4
    num_draws = 100 # number of simulated normal draws for probit computation
    true_gam = np.array([0.1,0.1])
    true_bet = np.array([0.8,0.8])
    true_uvn = np.array([1.,-1.])
    X = np.random.gamma(2,2,(num_obs,2,num_alt))
    eps = np.random.normal(0.,1.,(num_obs,2))
    r = np.random.normal(0.,1.,(num_obs,num_alt))
    v = np.sum((true_uvn+ true_gam[0:2] * eps)[:,:,np.newaxis] * X**true_bet[:,None] ,axis=1) + r
    y = np.argmax(v,axis=1)
    
    IndivData = {"y":torch.tensor(y),"x":torch.tensor(X)}
    
    
    opts['num_obs'] = num_obs
    opts['m_opts']['normdraws'] = torch.normal(0.,1.,size=(num_draws,num_alt)).double()
    # %% Draw Paricles
    SubParticles = HBSMC.sf.InitParticles(opts)
    # %% Run HSMC Loop  (auto)
    SubParticles = HBSMC.IndivEstim(SubParticles,IndivData,opts)
    # %% Run HSMC Loop  (manual)
#    for obs in range(opts['num_obs']):
#        time_start = time.time()
#        SubParticles = HBSMC.UpdateSubParticles(SubParticles,IndivData,obs,opts,pool=pool)
#        print("SMC Update completed (%.1f sec)" % (time.time() - time_start))
#        if obs % 10 == 0:
#            # Get posterior stats
#            stats = HBSMC.sf.ParticleStats(SubParticles,opts['m_opts'])
#            print("Model stats (per particle group):")
#            for g in range(opts['G']):
#                print("\n",stats[g])
    # %% Print posterior stats
    stats = HBSMC.sf.ParticleStats(SubParticles,opts['m_opts'])
    for g in range(opts['G']):
        print("\n",stats[g])







































