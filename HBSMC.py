# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 02:42:25 2019

@author: R Daviet
"""

import numpy as np
import torch
import SupportFunc as sf
import model_functions as modfunc
import drawval
import Mutate
import MutateH
import torch.multiprocessing as mp
import time
import os
import copy
import sys
import gc
import datetime
import psutil

def IndivEstim(SubParticles,IndivData,opts):
    np.set_printoptions(precision=5, floatmode='fixed')
    print(datetime.datetime.now())
    print("Begin SMC estimation (%dx%d particles)" % (SubParticles['particle'].shape[0],SubParticles['particle'].shape[1]))
    if opts['parallel']:
        if 'num_proc' not in opts.keys():
            opts['num_proc'] = 1
        pool = mp.Pool(max(1,opts['num_proc']))
    else:
        pool = None
    for obs in range(opts['num_obs']):
        time_start = time.time()
        SubParticles = UpdateSubParticles(SubParticles,IndivData,obs,opts,pool=pool)
        print("SMC Update completed (%.1f sec)" % (time.time() - time_start))
        if obs % 10 == 0:
            # Get posterior stats
            stats = sf.ParticleStats(SubParticles,opts['m_opts'])
            print("Model stats (per particle group):")
            for g in range(opts['G']):
                print("--Particle Group %d--" % g)
                for key, val in stats[g].items(): 
                    print(key,': ',val)

    if opts['parallel']:
        pool.close()
        pool.join()
    return SubParticles

def UpdateSubParticles(SubParticles,IndivData,obs,opts,pool=None):
    np.seterr(divide='ignore')
    ress = np.nan
    m_opts = opts['m_opts']
#    if opts['parallel']:
#        pool = mp.Pool()
    while np.isnan(ress):
        # %% C Phase
        print('\n===C Phase (obs %d)===' % obs)
        old_weights = SubParticles['logweights'].copy()
        old_loglike = np.zeros(SubParticles['particle'].shape)
        old_logmarglike = SubParticles['log_marg_like'].copy()
        for g in range(SubParticles['particle'].shape[0]):
            for p in range(SubParticles['particle'].shape[1]):
                log_proba_obs = modfunc.logProbaObs(obs,IndivData,SubParticles['particle'][g,p],m_opts).cpu().numpy()
                old_loglike[g,p] = SubParticles['particle'][g,p]['log_like']
                SubParticles['particle'][g,p]['log_like'] = SubParticles['particle'][g,p]['log_like'] + log_proba_obs
                SubParticles['logweights'][g,p] = SubParticles['logweights'][g,p] + log_proba_obs
        ress = np.sum(np.exp(SubParticles['logweights']))**2 / ( SubParticles['particle'].size * np.sum( np.exp(2 * SubParticles['logweights']) )  )
        # Save marginal likelihood for current observation
        log_w_bar = np.log( np.sum( np.exp(SubParticles['logweights']), axis=1) / np.sum(np.exp(old_weights),axis=1) )
        log_w_bar = np.atleast_2d(log_w_bar).transpose()
        if SubParticles['log_marg_like_obs'].shape[1] > obs:
            SubParticles['log_marg_like_obs'][:,obs] = log_w_bar.flatten()
        else:
            SubParticles['log_marg_like_obs'] = np.append(SubParticles['log_marg_like_obs'],log_w_bar,axis=1)   
        SubParticles['logweights'] = SubParticles['logweights'] - np.max(SubParticles['logweights'])
        SubParticles['log_marg_like'] = SubParticles['log_marg_like'] + log_w_bar.flatten()
        print("RESS: %.5f, p(obs %d) = %.3f" % (ress,obs,np.exp(np.mean(log_w_bar))) )  
        ML_SE = SubParticles['log_marg_like'].std() / SubParticles['log_marg_like'].size**0.5
        print("P(y_0:%d), logML: %.2f (Avg. Like: %.4f)" % (obs,np.mean(SubParticles['log_marg_like']),np.exp(np.mean(SubParticles['log_marg_like'])/(obs+1)) ), " - SE: %.2f" % ML_SE)
        print("By group: " , SubParticles['log_marg_like'] )
        sys.stdout.flush()
        
        if (not opts['adaptive']) or ( ress < opts['ress_threshold']) or np.isnan(ress) or (opts['num_obs'] == (obs +1) ):
            # %% S Phase
            last_obs = obs
            # if the ress is a NAN, refresh the particles with a move based on
            # the previous observation. This will be repeated until the ress is
            # a real number
            if np.isnan(ress):
                last_obs = obs - 1
                SubParticles['logweights'] = old_weights.copy()
                SubParticles['log_marg_like'] = old_logmarglike.copy()
                for g in range(SubParticles['particle'].shape[0]):
                    for p in range(SubParticles['particle'].shape[1]):
                        SubParticles['particle'][g,p]['log_like'] = old_loglike[g,p]
            # Make a copy of the particles
            NewSubP = copy.deepcopy(SubParticles)
            NewSubP['particle'] =  NewSubP['particle'][:,:opts['P']]
            for g in range(SubParticles['particle'].shape[0]):
                # Resample if weights are not all the same
                if np.unique(SubParticles['logweights'][g,:]).size > 1:
                    resample = drawval.draw_idx(np.arange(SubParticles['logweights'].shape[1]),np.exp(SubParticles['logweights'][g,:]),opts['P'])
                    for p in range(opts['P']):
                        NewSubP['particle'][g,p] = copy.deepcopy(SubParticles['particle'][g,resample[p]])
            NewSubP['logweights'] = np.zeros((SubParticles['particle'].shape[0],opts['P']))  
            SubParticles = NewSubP
            
            # %% M Phase
            if 'smc_m_min' in opts.keys():
                m_steps = opts['smc_m_start'] - (opts['smc_m_start'] - opts['smc_m_min']) * last_obs / (opts['smc_m_cooling']+last_obs)
                opts['smc_m_steps'] = int(m_steps)
            if ress < 0.25:
                opts['smc_m_steps'] = opts['smc_m_steps'] * 2
            print('===M Phase (obs %d): %d steps, %dx%d particles===' % (obs,
                                                                           opts['smc_m_steps'],
                                                                           SubParticles['particle'].shape[0],
                                                                           SubParticles['particle'].shape[1]))
            sum_accept = 0.
            accept_ratio = 0.
            opts['precision_coef'] = 0
            # Compute posterior statistics
            stats = sf.ParticleStats(SubParticles,m_opts)
            # Mutate
            count_mutate = 0
            while accept_ratio < opts['min_accept_ratio'] and sum_accept < opts['target_accept_ratio'] and count_mutate < opts['max_mutate_loop']:
                count_mutate = count_mutate + 1
                accept_count = np.zeros((opts['G'],opts['P']))
                for g in range(SubParticles['particle'].shape[0]):
                    m_opts['stats'] = stats[g]
                    if opts['parallel']:
                        g_time = time.time()
                        batch_size = 512
                        pool_results = [None] * batch_size
                        # Run in batches of 256
                        p_batch,last_batch = divmod(opts['P'], batch_size)
                        print('Group %d: ' % g, end='')
                        for b in range(p_batch):
                            for pp in range(batch_size):
                                p = b * batch_size + pp
                                try:
                                    pool_results[pp] = pool.apply_async(Mutate.Mutate,args=( \
                                            IndivData,last_obs,SubParticles['particle'][g,p],opts,m_opts \
                                            ))
                                except:
                                    print('Parent: Error during mutation phase')
                                    sys.exit(1)
                            for pp in range(batch_size):
                                p = b * batch_size + pp
                                try:
                                    (SubParticles['particle'][g,p],accept_count[g,p]) = pool_results[pp].get()
                                except:
                                    print('Parent:Error during mutation phase while gathing results from Pool.')
                                    sys.exit(1)
                                        
                            print("batch %d," % b, end='')
                        if last_batch:
                            for pp in range(last_batch):
                                p = p_batch * batch_size + pp
                                try:
                                    pool_results[pp] = pool.apply_async(Mutate.Mutate,args=( \
                                            IndivData,last_obs,SubParticles['particle'][g,p],opts,m_opts \
                                            ))   
                                except:
                                    print('Error during mutation phase')
                                    sys.exit(1) 
                            for pp in range(last_batch):
                                p = p_batch * batch_size + pp
                                try:
                                    (SubParticles['particle'][g,p],accept_count[g,p]) = pool_results[pp].get()
                                except:
                                    print('Error during mutation phase while gathing results from Pool.')
                                    sys.exit(1)
                            print("batch %d" % p_batch, end="")
                        print("(%.1f sec)" % (time.time() - g_time)) 
#                        for p in range(opts['P']):
#                            pool_results[p] = pool.apply_async(Mutate.Mutate,args=( \
#                                        IndivData,last_obs,SubParticles['particle'][g,p],opts,m_opts \
#                                        ))
#                        for p in range(opts['P']):
#                            (SubParticles['particle'][g,p],accept_count[g,p]) = pool_results[p].get()
                    else:
                        for p in range(opts['P']):
                            (SubParticles['particle'][g,p],accept_count[g,p]) = Mutate.Mutate(IndivData,last_obs,SubParticles['particle'][g,p],opts,m_opts)
                accept_ratio = np.mean(accept_count)
                sum_accept = sum_accept + accept_ratio
                print("M Phase accept. ratio (%d steps): %.5f" % (opts['smc_m_steps'],accept_ratio))
                if accept_ratio < opts['min_accept_ratio']  and sum_accept < opts['target_accept_ratio']:
                    opts['hmc_eps'] = opts['hmc_eps'] / 2
                    opts['precision_coef'] = opts['precision_coef']+ 1
                    print('M phase accept. ratio too low, redo M phase with precision coef: %d' % opts['precision_coef'] )
                sys.stdout.flush()
#    if opts['parallel']:
#        pool.close()
#        pool.join()
    return SubParticles


#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################

# %% ================= Hierarchical level=============
def HierarchUpdate(subj,SupParticles,VectSubParticles,opts,pool=None):
    np.seterr(divide='ignore')
    print("\n\n=======================================================")
    print("==================== Subject %d =======================" % subj)
    print("=======================================================")
    print(datetime.datetime.now())
    time_start = time.time()
    # %% Add new subject data if not already in list
    if VectSubParticles['sourcefile'][subj] in SupParticles['subj_list']:
        print("Subject found: %s\nSkipping update." % VectSubParticles['sourcefile'][subj])
        print("logP(y_%d|H) = %.4f \t logP(y_%d|M) = %.4f " % (subj,SupParticles['log_marg_like_subj'][:,subj].mean(),subj,VectSubParticles['m_logML'][subj]) )  
        print("logMarginal Likelihood: P(y_0:%d|H)= %.4f \t logP(y_0:%d|M) = %.4f  " % (subj,SupParticles['log_marg_like_subj'][:,0:subj+1].mean(0).sum(),subj, VectSubParticles['m_logML'][0:subj+1].sum()) ) 
        ML_SE = SupParticles['log_marg_like_subj'][:,0:subj+1].sum(1).std() / SupParticles['log_marg_like_subj'][:,0:subj+1].sum(1).size**0.5
        print("P(y_0:%d|H), By group: " %  subj , SupParticles['log_marg_like_subj'][:,0:subj+1].sum(1), " - SE: %.2f" % ML_SE)
        
    else:
        # Add subject
        print("Updating with data from:\n%s\n" % VectSubParticles['sourcefile'][subj])
        SupParticles['subj_list'].append(VectSubParticles['sourcefile'][subj])
        for g in range(SupParticles['particle'].shape[0]):
            for p in range(SupParticles['particle'].shape[1]):
                if len(SupParticles['subj_list']) > len(SupParticles['particle'][g,p]['log_hprob_y']):
                    SupParticles['particle'][g,p]['log_hprob_y'] = torch.cat((SupParticles['particle'][g,p]['log_hprob_y'],torch.tensor([0.],dtype=opts['torch_type'])))
                elif len(SupParticles['subj_list']) < len(SupParticles['particle'][g,p]['log_hprob_y']):
                    print("Particle %d,%d Length P(Y|H): %d \t Length Sub. List: %d" % (g,p,len(SupParticles['particle'][g,p]['log_hprob_y']),len(SupParticles['subj_list'])) )
                    raise Exception("Error: Length of particle's P(Y|H) does not match the subject list")
        
        # Multiply particles if clustering
        if SupParticles['h_opts']['dirichlet_param'] > 0.:
            if (SupParticles['particle'].shape[1] < (opts['P'] * 5) ):
                coef = 5
            else:
                coef = 1
            new_particles = np.empty((opts['G'],SupParticles['particle'].shape[1] * coef),dtype='object')
            new_logweights = np.zeros((opts['G'],SupParticles['particle'].shape[1] * coef))
            for g in range(SupParticles['particle'].shape[0]):
                for p in range(SupParticles['particle'].shape[1]):
                    for pp in range(coef):
                        p_new = p * coef + pp
                        new_particles[g,p_new] = copy.deepcopy(SupParticles['particle'][g,p])
                        new_logweights[g,p_new] = SupParticles['logweights'][g,p].copy() 
            SupParticles['particle'] = new_particles
            SupParticles['logweights'] = new_logweights.copy()
                    
                    
        # Assign new cluster           
        if len(SupParticles['subj_list']) > 1:
            for g in range(SupParticles['particle'].shape[0]):
                for p in range(SupParticles['particle'].shape[1]):
                    if SupParticles['particle'][g,p]['sub_clust'].size < len(SupParticles['subj_list']):
                        SupParticles['particle'][g,p]['sub_clust'] = np.append(SupParticles['particle'][g,p]['sub_clust'],0)
                        if SupParticles['h_opts']['dirichlet_param']:
                            c_max = np.max(SupParticles['particle'][g,p]['sub_clust'][0:subj+1]) + 1
                            p_clust = np.zeros(c_max + 1, dtype='float')
                            p_clust[-1] =  SupParticles['h_opts']['dirichlet_param']
                            for c in range(c_max):
                                p_clust[c] = np.count_nonzero(SupParticles['particle'][g,p]['sub_clust'][0:subj+1] == c) 
                            cum_p_clust = p_clust.cumsum() / p_clust.sum()
                            unif_draws = np.random.rand(1)[0]
                            c_draw = (unif_draws < cum_p_clust).argmax()
#                            c_draw = drawval.draw_idx(np.arange(c_max+1),p_clust,1)
                            SupParticles['particle'][g,p]['sub_clust'][subj] = c_draw   
                            SupParticles['particle'][g,p] = sf.AddCluster(SupParticles['particle'][g,p],opts)
            
        ress = np.nan       
        while np.isnan(ress):
            # %% C Phase
            G = SupParticles['particle'].shape[0]
            P = SupParticles['particle'].shape[1]
            print('===C Phase (Subj %d): %dx%d particles===' % (subj,G,P))
            old_weights = SupParticles['logweights'].copy()
            old_loglike = np.zeros(SupParticles['particle'].shape)
            old_logmarglike = SupParticles['log_marg_like'].copy()
            
            
            # Correction
            for g in range(SupParticles['particle'].shape[0]):
                for p in range(SupParticles['particle'].shape[1]):
                    if 'dtype' not in SupParticles['particle'][g,p].keys():
                        SupParticles['particle'][g,p]['dtype'] =  opts['torch_type']
                    # get weight
                    log_proba_obs = sf.logHProbaObs(subj,VectSubParticles,SupParticles['particle'][g,p],opts['h_opts']).cpu().numpy()
                    SupParticles['particle'][g,p]['log_hprob_y'][subj] = torch.tensor(log_proba_obs,dtype=opts['torch_type'],device=opts['torch_device'])
                    old_loglike[g,p] = SupParticles['particle'][g,p]['log_hlike']
                    SupParticles['particle'][g,p]['log_hlike'] = SupParticles['particle'][g,p]['log_hlike'] + log_proba_obs
                    SupParticles['logweights'][g,p] = SupParticles['logweights'][g,p] + log_proba_obs
            max_weight = SupParticles['logweights'].max(1)
            norm_weight = SupParticles['logweights'] - max_weight[:,None]
            ress = np.sum(np.exp(norm_weight))**2 / ( SupParticles['particle'].size * np.sum( np.exp(2 * norm_weight) )  )

            # Save marginal likelihood for current observation
            log_w_bar = np.log( np.sum( np.exp(SupParticles['logweights']), axis=1) / np.sum(np.exp(old_weights),axis=1) )
            log_w_bar = np.atleast_2d(log_w_bar).transpose()
            if SupParticles['log_marg_like_subj'].shape[1] > subj:
                SupParticles['log_marg_like_subj'][:,subj] = log_w_bar.flatten()
            else:
                SupParticles['log_marg_like_subj'] = np.append(SupParticles['log_marg_like_subj'],log_w_bar,axis=1)   
            SupParticles['logweights'] = SupParticles['logweights'] - np.max(SupParticles['logweights'])
            SupParticles['log_marg_like'] = SupParticles['log_marg_like'] + log_w_bar.flatten()
            print("RESS: %.5f, logP(y_%d|H) = %.4f \t logP(y_%d|M) = %.4f " % (ress,subj,np.mean(log_w_bar),subj,VectSubParticles['m_logML'][subj]) )  
            print("logMarginal Likelihood: P(y_0:%d|H)= %.4f \t logP(y_0:%d|M) = %.4f  " % (subj,np.mean(SupParticles['log_marg_like']),subj, VectSubParticles['m_logML'][0:subj+1].sum()) ) 
            ML_SE = SupParticles['log_marg_like'].std() / SupParticles['log_marg_like'].size**0.5
            print("P(y_0:%d|H), By group: " %  subj , SupParticles['log_marg_like'] , " - SE: %.2f" % ML_SE)
            sys.stdout.flush()
            # %% S Phase
            if (not opts['adaptive']) or ( ress < opts['ress_threshold']) or np.isnan(ress) or (VectSubParticles['m_logML'].size()[0] == (subj +1) ):
               
                last_subj = subj
                # if the ress is a NAN, refresh the particles with a move based on
                # the previous observation. This will be repeated until the ress is
                # a real number
                if np.isnan(ress):
                    last_subj = subj - 1
                    SupParticles['logweights'] = old_weights.copy()
                    SupParticles['log_marg_like'] = old_logmarglike.copy()
                    for g in range(SupParticles['particle'].shape[0]):
                        for p in range(SupParticles['particle'].shape[1]):
                            SupParticles['particle'][g,p]['log_like'] = old_loglike[g,p]
                # Make a copy of the particles
                NewSubP = copy.deepcopy(SupParticles)
                NewSubP['particle'] =  NewSubP['particle'][:,:opts['P']]
                for g in range(SupParticles['particle'].shape[0]):
                    # Resample if weights are not all the same
                    if np.unique(SupParticles['logweights'][g,:]).size > 1:
                        resample = drawval.draw_idx(np.arange(SupParticles['logweights'].shape[1]),np.exp(SupParticles['logweights'][g,:]),opts['P'])
                        for p in range(opts['P']):
                            NewSubP['particle'][g,p] = copy.deepcopy(SupParticles['particle'][g,resample[p]])
                NewSubP['logweights'] = np.zeros((SupParticles['particle'].shape[0],opts['P']))  
                SupParticles = NewSubP
                
                # %% M Phase
                # Number of M steps
                if 'smc_m_min' in opts.keys():
                    m_steps = opts['smc_m_start'] - (opts['smc_m_start'] - opts['smc_m_min']) * last_subj / (opts['smc_m_cooling']+last_subj)
                    opts['smc_m_steps'] = int(m_steps)
                print('===M Phase (obs %d): %d steps, %dx%d particles===' % (subj,opts['smc_m_steps'],G,opts['P']))
                sum_accept = 0.
                accept_ratio = 0.
                precision_coef = 0 
                # Compute posterior statistics
                stats = sf.SParticleStats(SupParticles,opts['h_opts'])
                # Mutate
                count_mutate = 0
                opts['hmc_eps'] = opts['hmc_eps_base'] / (subj + 1.)
                opts['precision_coef'] = (subj + 1.)
                while accept_ratio < opts['min_accept_ratio'] and sum_accept < opts['target_accept_ratio'] and count_mutate < opts['max_mutate_loop']:
                    count_mutate = count_mutate + 1
                    accept_count = np.zeros(SupParticles['particle'].shape)
                    m_time_start = time.time()
                    for g in range(SupParticles['particle'].shape[0]):
                        opts['h_opts']['stats'] = stats[g]
                        count_mutate_g = 0
                        while np.mean(accept_count[g]) < opts['min_accept_ratio'] and count_mutate_g < opts['max_mutate_loop']:
                            opts['precision_coef'] = precision_coef + count_mutate_g
                            accept_count[g] = np.zeros(SupParticles['particle'].shape[1])
                            count_mutate_g = count_mutate_g + 1
                            mg_time_start = time.time()
                            if opts['parallel']:
                                pool_results = [None] * SupParticles['particle'].shape[1]
                                try:
                                    num_cores = psutil.cpu_count(logical=False)
                                    with mp.Pool(max(2,num_cores-4)) as loc_pool:
                                        for p in range(opts['P']):
                                            pool_results[p] = loc_pool.apply_async(MutateH.MutateH,
                                                        args=(VectSubParticles,last_subj,SupParticles['particle'][g,p],opts ))
                                        for p in range(opts['P']):
                                            (SupParticles['particle'][g,p],accept_count[g,p]) = pool_results[p].get()
                                except:
                                    print('Parent: Error during mutation phase. Terminating script.')
                                    sys.exit(1)  
                            else:
                                for p in range(opts['P']):
                                    (SupParticles['particle'][g,p],accept_count[g,p]) = MutateH.MutateH(VectSubParticles,last_subj,SupParticles['particle'][g,p],opts)
                            print("Group %d accept. ratio: %0.4f (%.1f sec)" % (g,np.mean(accept_count[g]),time.time() - mg_time_start))
                            sys.stdout.flush()
                    accept_ratio = np.mean(accept_count)
                    sum_accept = sum_accept + accept_ratio
                    print("M Phase accept. ratio: %.5f (%.1f sec)" % (accept_ratio,time.time() - m_time_start))
                    if accept_ratio < opts['min_accept_ratio']  and sum_accept < opts['target_accept_ratio']:
                        opts['hmc_eps'] = opts['hmc_eps'] / 2
                        precision_coef = precision_coef + 1
                        print('M phase accept. ratio too low, redo M phase with precision coef: %d' % opts['precision_coef'] )
        sup_path = os.path.join(opts['savepath'],'Stage2-' + opts['suptag'] + '.dict')
        torch.save(SupParticles,sup_path)
        subj_path = os.path.join(opts['savepath'],'Stage2-' + opts['suptag'] + '-Subj%02d.dict' % subj)
        torch.save(SupParticles,subj_path)
        
    print("SMC Update completed (%.1f sec)" % (time.time() - time_start))
    sys.stdout.flush()
    # %% Return updated particles
    return SupParticles

