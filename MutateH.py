# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:07:54 2019

@author: R
"""
import numpy as np
import SupportFunc as sf
import torch
import copy
    
def MutateH(VectSubParticles,last_subj,sparticle,opts):
    try:
        # %% Init
        accept = 0.
        tentatives = 0.
        h_opts = opts['h_opts']
        has_gam = h_opts['base_gam'].numel() > 0
        has_bet = h_opts['base_bet'].numel() > 0
        has_uvn = h_opts['base_uvn'].numel() > 0
        if has_gam:
            d_gam = sparticle['h_gam'].shape[1:3]
        if has_bet:
            d_bet = sparticle['h_bet'].shape[1:3]
        if has_uvn:
            d_uvn = sparticle['h_uvn'].shape[1:3]
        # %% Redraw cluster   
        if last_subj > 0:
            for subj in range(last_subj+1):
                c_max = np.max(sparticle['sub_clust'][0:last_subj+1]) + 1
                c_orig = sparticle['sub_clust'][subj]
                # Compute cluster probas
                p_clust = np.zeros(c_max+1, dtype='float')
                log_p_y_clust = np.zeros(c_max+1, dtype='float')
                prop_part = copy.deepcopy(sparticle)
                prop_part['sub_clust'][subj] = c_max 
                prop_part = sf.AddCluster(prop_part,opts)
                for c in range(c_max):
                    c_count = np.count_nonzero(sparticle['sub_clust'][0:last_subj+1] == c) - (c == c_orig)
                    p_clust[c] = c_count
                    if c == c_orig:
                        log_p_y_clust[c] = sparticle['log_hprob_y'][subj].item()
                        if not p_clust[c]:
                            p_clust[c] = h_opts['dirichlet_param'] * np.exp(log_p_y_clust[c])
                        else : 
                            p_clust[c] = p_clust[c] * np.exp(log_p_y_clust[c])
                            prop_part['sub_clust'][subj] =  c_max                      
                            log_p_y_clust[-1] = sf.logHProbaObs(subj,VectSubParticles,prop_part,h_opts)
                            p_clust[-1] = h_opts['dirichlet_param'] * np.exp(log_p_y_clust[-1])
                    else:
                        if p_clust[c]:
                            prop_part['sub_clust'][subj] =  c
                            log_p_y_clust[c] = sf.logHProbaObs(subj,VectSubParticles,prop_part,h_opts)
                            p_clust[c] = p_clust[c] * np.exp(log_p_y_clust[c])
                cum_p_clust = p_clust.cumsum() / p_clust.sum()
                unif_draws = np.random.rand(1)[0]
                c_draw = (unif_draws < cum_p_clust).argmax()
    #            c_draw = drawval.draw_idx(np.arange(c_max+1),p_clust,1)[0]
                if c_draw == c_max:
                    sparticle = prop_part
                sparticle['sub_clust'][subj] =  c_draw
                sparticle['log_hprob_y'][subj] = torch.tensor(log_p_y_clust[c_draw],dtype=torch.double,device=opts['torch_device'])
                if has_gam:
                    if torch.isnan(sparticle['h_gam']).any():
                        print("NaN after redrawing cluster, subj:",subj,"\n",sparticle)
            # clear unused clusters
            c_list = np.unique(sparticle['sub_clust'])
            K = sparticle['log_hprior'].numel()
            for c in range(K-1,-1,-1):
                if c not in c_list:
                     c_up = sparticle['sub_clust'] > c
                     sparticle['sub_clust'][c_up] = sparticle['sub_clust'][c_up] - 1
                     c_mask = ~(c == np.arange(sparticle['log_hprior'].numel()))
                     sparticle['log_hprior'] = sparticle['log_hprior'][c_mask]
                     if has_gam:
                         sparticle['h_gam'] = sparticle['h_gam'][c_mask]
                     if has_bet:
                         sparticle['h_bet'] = sparticle['h_bet'][c_mask]
                     if has_uvn:
                         sparticle['h_uvn'] = sparticle['h_uvn'][c_mask]  
                if has_gam:
                    if torch.isnan(sparticle['h_gam']).any():
                        print("NaN after clearing cluster:",c,"\n",sparticle)              
    
        # %% Redraw H for each cluster
        clust_list = np.unique(sparticle['sub_clust'])
        for clust in clust_list:
            list_subj = np.where(sparticle['sub_clust'] == clust)[0]
            list_subj = list_subj[list_subj<=last_subj]
            if opts['smc_type'].lower() == 'hamiltonian':
                U_init = - sparticle['log_hprior'][clust] -  sparticle['log_hprob_y'][list_subj].sum() 
                momentum_init_gam = torch.tensor(0.,device=opts['torch_device'])
                momentum_init_bet = torch.tensor(0.,device=opts['torch_device'])
                momentum_init_uvn = torch.tensor(0.,device=opts['torch_device'])
                momentum_gam = torch.tensor(0.,device=opts['torch_device'])
                momentum_bet = torch.tensor(0.,device=opts['torch_device'])
                momentum_uvn = torch.tensor(0.,device=opts['torch_device'])
                for m in range(opts['smc_m_steps']):
                    tentatives = tentatives + 1
                    prop_part = copy.deepcopy(sparticle)
                    has_nan = False
                    p_acc = 0.
                    #initial momentum
                    coef = opts['precision_base'] ** (-opts['precision_coef'])
                    if has_gam :              
                        momentum_init_gam = torch.randn(d_gam,device=opts['torch_device']).double() * coef
                        prop_part['h_gam'].requires_grad = True
                        if prop_part['h_gam'].grad is not None:
                            prop_part['h_gam'].grad.zero_()
                        
                    if has_bet :
                        momentum_init_bet = torch.randn(d_bet,device=opts['torch_device']).double() * coef
                        prop_part['h_bet'].requires_grad = True
                        if prop_part['h_bet'].grad is not None:
                            prop_part['h_bet'].grad.zero_()
                        
                    if has_uvn :
                        momentum_init_uvn = torch.randn(d_uvn,device=opts['torch_device']).double() * coef
                        prop_part['h_uvn'].requires_grad = True
                        if prop_part['h_uvn'].grad is not None:
                            prop_part['h_uvn'].grad.zero_()
                    
                    K_init = 0.5 * ( torch.sum(momentum_init_gam**2) + torch.sum(momentum_init_bet**2) + torch.sum(momentum_init_uvn**2) )
                    # Get Grad_U : - Grad[ logP(y|H) + logP(H)]
                    U = - sf.LogHPrior(prop_part,clust,h_opts) - sf.getHProbs(list_subj,VectSubParticles,prop_part,h_opts).sum()
                    if torch.isinf(U):
                        continue
                    if torch.isnan(U):
                        print("\nNaN U (m=%d,first half-step): " % (m),U)
                        continue
                    U.backward()
                    
                    # momentum half step
                    if has_gam :
                        grad_U_gam = prop_part['h_gam'].grad.data.clone()
                        momentum_gam = momentum_init_gam - 0.5 * opts['hmc_eps']* grad_U_gam[clust]
                        prop_part['h_gam'].requires_grad = False
                        if torch.isnan(grad_U_gam).any().item():
                            print("NaN Gamma Gradient (m:%d,first half-step): " % m, grad_U_gam, "\n\nat:\n", prop_part['h_gam'][clust] )
                            print("U: ",U)
                            continue
                        
                    if has_bet :
                        grad_U_bet = prop_part['h_bet'].grad.data.clone()
                        momentum_bet = momentum_init_bet - 0.5 * opts['hmc_eps']* grad_U_bet[clust]
                        prop_part['h_bet'].requires_grad = False
                        if torch.isnan(grad_U_bet).any() | torch.isinf(grad_U_bet).any():
                            print("NaN Beta Gradient (m:%d,first half-step): " % m, grad_U_bet, "\n\nat:\n", prop_part['h_bet'][clust] )
                            print("U: ",U)
                            continue
                        
                    if has_uvn :
                        grad_U_uvn = prop_part['h_uvn'].grad.data.clone()
                        momentum_uvn = momentum_init_uvn - 0.5 * opts['hmc_eps']* grad_U_uvn[clust]
                        prop_part['h_uvn'].requires_grad = False
                        if torch.isnan(grad_U_uvn).any().item():
                            print("NaN UVN Gradient (m:%d,first half-step): " % m, grad_U_uvn, "\n\nat:\n", prop_part['h_uvn'][clust] )
                            print("U: ",U)
                            continue
                    
                    
                    for s in range(opts['hmc_leaps']):
                        #Update position
                        if has_gam :
                            prop_part['h_gam'][clust] = prop_part['h_gam'][clust] + opts['hmc_eps'] * momentum_gam    
                            if torch.isnan(prop_part['h_gam']).sum():
                                print("NaN in Gamma proposal before bound check (m=%d,s=%d): " % (m,s))
                                print("Proposal:", prop_part['h_gam'])
                                print("Momentum:" ,momentum_gam)
                                has_nan = True
                                break   
                            # check constraints
                            for d0 in range(prop_part['h_gam'][clust].shape[0]):
                                low_bound_check = prop_part['h_gam'][clust,d0] < opts['h_opts']['gam_bound'][d0,0]
                                high_bound_check = prop_part['h_gam'][clust,d0] > opts['h_opts']['gam_bound'][d0,1]
                                while low_bound_check.any() or high_bound_check.any():
                                    prop_part['h_gam'][clust,d0,low_bound_check] = 2. * opts['h_opts']['gam_bound'][d0,0,low_bound_check] \
                                        - prop_part['h_gam'][clust,d0,low_bound_check]
                                    momentum_gam[d0,low_bound_check] = - momentum_gam[d0,low_bound_check]
                                    prop_part['h_gam'][clust,d0,high_bound_check] = 2. * opts['h_opts']['gam_bound'][d0,1,high_bound_check] \
                                        - prop_part['h_gam'][clust,d0,high_bound_check]
                                    momentum_gam[d0,high_bound_check] = - momentum_gam[d0,high_bound_check]
                                    low_bound_check = prop_part['h_gam'][clust,d0] < opts['h_opts']['gam_bound'][d0,0]
                                    high_bound_check = prop_part['h_gam'][clust,d0] > opts['h_opts']['gam_bound'][d0,1]    
                            if torch.isnan(prop_part['h_gam']).sum():
                                print("NaN in Gamma proposal (c=%d,m=%d,s=%d): " % (clust,m,s), torch.isnan(prop_part['h_gam']).any())
                                print("Proposal ():", prop_part['h_gam'])
                                print("Momentum:" ,momentum_gam)
                                has_nan = True
                                break                 
                            prop_part['h_gam'].requires_grad = True
                            if prop_part['h_gam'].grad is not None:
                                prop_part['h_gam'].grad.zero_()
                                        
                        if has_bet :
                            prop_part['h_bet'][clust] = prop_part['h_bet'][clust] + opts['hmc_eps'] * momentum_bet
                            if torch.isnan(prop_part['h_bet']).sum():
                                print("NaN in Beta proposal before bound check (m=%d,s=%d): " % (m,s))
                                print("Proposal:", prop_part['h_bet'])
                                print("Momentum:" ,momentum_bet)
                                has_nan = True
                                break     
                            # check constraints
                            for d0 in range(prop_part['h_bet'][clust].shape[0]):
                                low_bound_check = prop_part['h_bet'][clust,d0] < opts['h_opts']['bet_bound'][d0,0]
                                high_bound_check = prop_part['h_bet'][clust,d0] > opts['h_opts']['bet_bound'][d0,1]
                                while low_bound_check.any() or high_bound_check.any():
                                    prop_part['h_bet'][clust,d0,low_bound_check] = 2 * opts['h_opts']['bet_bound'][d0,0,low_bound_check] \
                                        - prop_part['h_bet'][clust,d0,low_bound_check]
                                    momentum_bet[d0,low_bound_check] = - momentum_bet[d0,low_bound_check]
                                    prop_part['h_bet'][clust,d0,high_bound_check] = 2 * opts['h_opts']['bet_bound'][d0,1,high_bound_check] \
                                        - prop_part['h_bet'][clust,d0,high_bound_check]
                                    momentum_bet[d0,high_bound_check] = - momentum_bet[d0,high_bound_check]
                                    low_bound_check = prop_part['h_bet'][clust,d0] < opts['h_opts']['bet_bound'][d0,0]
                                    high_bound_check = prop_part['h_bet'][clust,d0] > opts['h_opts']['bet_bound'][d0,1]
                            if torch.isnan(prop_part['h_bet']).sum():
                                print("NaN in Beta proposal (c=%d,m=%d,s=%d): " % (clust,m,s),torch.isnan(prop_part['h_bet']).any())
                                print("Proposal:", prop_part['h_bet'])
                                print("Momentum:" ,momentum_bet)
                                has_nan = True
                                break                     
                            prop_part['h_bet'].requires_grad = True
                            if prop_part['h_bet'].grad is not None:
                                prop_part['h_bet'].grad.zero_()
                                        
                                        
                        if has_uvn :
                            prop_part['h_uvn'][clust] = prop_part['h_uvn'][clust] + opts['hmc_eps'] * momentum_uvn
                            if torch.isnan(prop_part['h_uvn']).sum():
                                print("NaN in UVN proposal before bound check (m=%d,s=%d): " % (m,s))
                                print("Proposal:", prop_part['h_uvn'])
                                print("Momentum:" ,momentum_uvn)
                                has_nan = True
                                break      
                            # check constraints
                            for d0 in range(prop_part['h_uvn'][clust].shape[0]):
                                low_bound_check = prop_part['h_uvn'][clust,d0] < opts['h_opts']['uvn_bound'][d0,0]
                                high_bound_check = prop_part['h_uvn'][clust,d0] > opts['h_opts']['uvn_bound'][d0,1]
                                while low_bound_check.any() or high_bound_check.any():
                                    prop_part['h_uvn'][clust,d0,low_bound_check] = 2 * opts['h_opts']['uvn_bound'][d0,0,low_bound_check] \
                                        - prop_part['h_uvn'][clust,d0,low_bound_check]
                                    momentum_uvn[d0,low_bound_check] = - momentum_uvn[d0,low_bound_check]
                                    prop_part['h_uvn'][clust,d0,high_bound_check] = 2 * opts['h_opts']['uvn_bound'][d0,1,high_bound_check] \
                                        - prop_part['h_uvn'][clust,d0,high_bound_check]
                                    momentum_uvn[d0,high_bound_check] = - momentum_uvn[d0,high_bound_check]
                                    low_bound_check = prop_part['h_uvn'][clust,d0] < opts['h_opts']['uvn_bound'][d0,0]
                                    high_bound_check = prop_part['h_uvn'][clust,d0] > opts['h_opts']['uvn_bound'][d0,1]
                            if torch.isnan(prop_part['h_uvn']).sum():
                                print("NaN in UVN proposal (c=%d,m=%d,s=%d): " % (clust,m,s), torch.isnan(prop_part['h_uvn']).any())
                                print("Proposal:", prop_part['h_uvn'])
                                print("Momentum:" ,momentum_uvn)
                                has_nan = True
                                break                     
                            prop_part['h_uvn'].requires_grad = True
                            if prop_part['h_uvn'].grad is not None:
                                prop_part['h_uvn'].grad.zero_()
                             
                        # Get Gradient of U
                        log_prior = sf.LogHPrior(prop_part,clust,h_opts)
                        log_like = sf.getHProbs(list_subj,VectSubParticles,prop_part,h_opts)
                        U = - log_prior - log_like.sum()
                        if torch.isinf(U):
                            break
                        if torch.isnan(U):
                            print("\nNaN U (m=%d,s=%d): " % (m,s),U)
                            print("\nLogPrior:", log_prior,"\nLokgLike:",log_like )
                            has_nan = True
                            break
                        U.backward()
                        
                        if has_gam :
                            grad_U_gam = prop_part['h_gam'].grad.data.clone()
                            prop_part['h_gam'].requires_grad = False
                            if torch.isnan(grad_U_gam).any().item():
                                print("\n\nNaN Gamma Gradient (m:%d,s=%d): " % (m,s), grad_U_gam, "\n\nat:\n", prop_part['h_gam'][clust] )
                                print("U: ",U)
                                has_nan = True
                                break
                            momentum_last_gam = momentum_gam - 0.5 * opts['hmc_eps']* grad_U_gam[clust]
                            momentum_gam = momentum_gam - opts['hmc_eps'] * grad_U_gam[clust]
                             
                            
                        if has_bet:
                            grad_U_bet = prop_part['h_bet'].grad.data.clone()
                            prop_part['h_bet'].requires_grad = False
                            if torch.isnan(grad_U_bet).any().item():
                                print("\n\nNaN Beta Gradient (m:%d,s=%d): " % (m,s), grad_U_bet, "\n\nat:\n", prop_part['h_bet'][clust] )
                                print("U: ",U)
                                has_nan = True
                                break
                            momentum_last_bet =momentum_bet - 0.5 * opts['hmc_eps']* grad_U_bet[clust]
                            momentum_bet = momentum_bet - opts['hmc_eps']* grad_U_bet[clust]
                        if has_uvn:
                            grad_U_uvn = prop_part['h_uvn'].grad.data.clone()
                            prop_part['h_uvn'].requires_grad = False
                            if torch.isnan(grad_U_uvn).any().item():
                                print("\n\nNaN UVN Gradient (m:%d,s=%d): " % (m,s), grad_U_uvn, "\n\nat:\n", prop_part['h_uvn'][clust] )
                                print("U: ",U)
                                has_nan = True
                                break
                            momentum_last_uvn = momentum_uvn - 0.5 * opts['hmc_eps']* grad_U_uvn[clust]
                            momentum_uvn = momentum_uvn - opts['hmc_eps']* grad_U_uvn[clust]
                        
                        K = 0.5 * ( torch.sum(momentum_last_gam**2) + torch.sum(momentum_last_bet**2) + torch.sum(momentum_last_uvn**2) )
                        dH = U_init.detach() - U.detach()  + K_init - K
                        
                        if dH.abs() > 4:
                            break
    
                    # Acceptance probability                
                    if not has_nan :
                        if dH.abs() < 4:
                            p_acc = np.minimum(1.,torch.exp(dH).cpu().numpy())
                            if  np.random.random() < p_acc:
                                sparticle = prop_part
                                sparticle['log_hprior'][clust] = log_prior.cpu().detach().item()
                                sparticle['log_hprob_y'][list_subj] = log_like.cpu().detach()
                                sparticle['log_like'] = sparticle['log_hprob_y'][0:last_subj+1].sum().item()
                                U_init = U.detach().clone()
                                accept = accept + 1
                    else:
                        print("NaN during HMC (m:%d), subj:" % m,last_subj)
                        print("\nSubjList:",list_subj)
                        print("\nLogPrior:", sf.LogHPrior(prop_part,clust,h_opts),"\nLokgLike:",sf.getHProbs(list_subj,VectSubParticles,sparticle,h_opts) )
                        print("\nMomentum Gamma: ",momentum_gam,"\nMomentum Beta: ",momentum_bet,"\nMomentum UVN: ",momentum_uvn)
                        print("\n\nParticle:------------\n",sparticle)
                    
            
            else:
                for m in range(opts['smc_m_steps']):
                    # %% Update params
                    prop_part = copy.deepcopy(sparticle)
                    prop_part['log_hprior'][clust] = 0.
                    logQRatio = 0.
                    # %% Gamma proposal
                    if has_gam :
                        step_coef = opts['precision_base']**(opts['precision_coef']+1) * 8  * np.ones(prop_part['h_gam'][clust].shape) / np.maximum(h_opts['stats']['gam_sd'],0.000000001)**2 # * np.sqrt(last_subj+1.)
                        step_coef = np.maximum(step_coef,10)
                        step = np.random.gamma(step_coef,1/step_coef)
                        prop_part['h_gam'][clust] = prop_part['h_gam'][clust] * torch.tensor(step,device=prop_part['h_gam'].device)
                        # check constraints
                        for d0 in range(prop_part['h_gam'][clust].shape[0]):
                            low_bound_check = prop_part['h_gam'][clust,d0] < opts['h_opts']['gam_bound'][d0,0]
                            high_bound_check = prop_part['h_gam'][clust,d0] > opts['h_opts']['gam_bound'][d0,1]
                            while low_bound_check.any() or high_bound_check.any():
                                prop_part['h_gam'][clust,d0,low_bound_check] = 2. * opts['h_opts']['gam_bound'][d0,0,low_bound_check] \
                                    - prop_part['h_gam'][clust,d0,low_bound_check]
                                prop_part['h_gam'][clust,d0,high_bound_check] = 2. * opts['h_opts']['gam_bound'][d0,1,high_bound_check] \
                                    - prop_part['h_gam'][clust,d0,high_bound_check]
                                low_bound_check = prop_part['h_gam'][clust,d0] < opts['h_opts']['gam_bound'][d0,0]
                                high_bound_check = prop_part['h_gam'][clust,d0] > opts['h_opts']['gam_bound'][d0,1]    
                        
                        logQRatio = logQRatio + np.sum(- (2 * step_coef - 1) * np.log(step) + step_coef * (step - 1./step))
    
                    # Beta proposal
                    if has_bet:
                        step_coef = np.maximum(h_opts['stats']['bet_sd'],0.000000001) * opts['precision_base']**-(opts['precision_coef']+1) * np.ones(prop_part['h_bet'][clust].shape) * 0.5 # / np.sqrt(last_subj+1.)
                        step_coef = np.maximum(step_coef,0.000000001)
                        step = np.random.normal(0, step_coef )
                        prop_part['h_bet'][clust] = prop_part['h_bet'][clust] + torch.tensor(step,device=prop_part['h_bet'].device)
                        # check constraints
                        for d0 in range(prop_part['h_bet'][clust].shape[0]):
                            low_bound_check = prop_part['h_bet'][clust,d0] < opts['h_opts']['bet_bound'][d0,0]
                            high_bound_check = prop_part['h_bet'][clust,d0] > opts['h_opts']['bet_bound'][d0,1]
                            while low_bound_check.any() or high_bound_check.any():
                                prop_part['h_bet'][clust,d0,low_bound_check] = 2 * opts['h_opts']['bet_bound'][d0,0,low_bound_check] \
                                    - prop_part['h_bet'][clust,d0,low_bound_check]
                                prop_part['h_bet'][clust,d0,high_bound_check] = 2 * opts['h_opts']['bet_bound'][d0,1,high_bound_check] \
                                    - prop_part['h_bet'][clust,d0,high_bound_check]
                                low_bound_check = prop_part['h_bet'][clust,d0] < opts['h_opts']['bet_bound'][d0,0]
                                high_bound_check = prop_part['h_bet'][clust,d0] > opts['h_opts']['bet_bound'][d0,1]
                    # univariate normal param
                    if has_uvn :
                        step_coef = np.maximum(h_opts['stats']['uvn_sd'],0.000000001) * opts['precision_base']**-(opts['precision_coef']+1) * np.ones(prop_part['h_uvn'][clust].shape) * 0.5 # / np.sqrt(last_subj+1.)
                        step_coef = np.maximum(step_coef ,0.000000001)
                        step = np.random.normal(0,step_coef)
                        prop_part['h_uvn'][clust] = prop_part['h_uvn'][clust] + torch.tensor(step,device=prop_part['h_uvn'].device)
                        # check constraints
                        for d0 in range(prop_part['h_uvn'][clust].shape[0]):
                            low_bound_check = prop_part['h_uvn'][clust,d0] < opts['h_opts']['uvn_bound'][d0,0]
                            high_bound_check = prop_part['h_uvn'][clust,d0] > opts['h_opts']['uvn_bound'][d0,1]
                            while low_bound_check.any() or high_bound_check.any():
                                prop_part['h_uvn'][clust,d0,low_bound_check] = 2 * opts['h_opts']['uvn_bound'][d0,0,low_bound_check] \
                                    - prop_part['h_uvn'][clust,d0,low_bound_check]
                                prop_part['h_uvn'][clust,d0,high_bound_check] = 2 * opts['h_opts']['uvn_bound'][d0,1,high_bound_check] \
                                    - prop_part['h_uvn'][clust,d0,high_bound_check]
                                low_bound_check = prop_part['h_uvn'][clust,d0] < opts['h_opts']['uvn_bound'][d0,0]
                                high_bound_check = prop_part['h_uvn'][clust,d0] > opts['h_opts']['uvn_bound'][d0,1]
                    # Accept / reject
                    prop_part['log_hprior'][clust] = sf.LogHPrior(prop_part,clust,h_opts)
                    prop_part['log_hprob_y'][list_subj] = sf.getHProbs(list_subj,VectSubParticles,prop_part,h_opts).cpu().detach()
                    loglike_ratio = prop_part['log_hprob_y'][list_subj].sum() - sparticle['log_hprob_y'][list_subj].sum()
                    if np.log(np.random.rand()) <= loglike_ratio + prop_part['log_hprior'][clust] - sparticle['log_hprior'][clust] + logQRatio:
                        sparticle = prop_part
                        accept = accept + 1
                    tentatives = tentatives + 1
        # Check for NaNs
        if has_gam :
            if torch.isnan(sparticle['h_gam']).any():
                print("h_gam : NaN after MCMC step, subj:",last_subj,"\n",sparticle)
                sparticle['h_gam'].requires_grad = False
        if has_bet :
            if torch.isnan(sparticle['h_bet']).any():
                print("h_bet : NaN after MCMC step, subj:",last_subj,"\n",sparticle)
                sparticle['h_bet'].requires_grad = False
        if has_uvn :
            if torch.isnan(sparticle['h_uvn']).any():
                print("h_uvn : NaN after MCMC step, subj:",last_subj,"\n",sparticle)
                sparticle['h_gam'].requires_grad = False
            
        return (sparticle,accept/tentatives)
    except:
        raise Exception('Error in MutateH during execution')