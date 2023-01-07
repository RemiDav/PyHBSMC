# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:07:54 2019

@author: R
"""
import numpy as np
import SupportFunc as sf
import torch
import copy
    

def Mutate(IndivData,last_obs,particle,opts,m_opts):
    try:
        accept = 0.
        tentatives = 0.
        # Initialize
        if opts['smc_type'].lower() == 'hamiltonian':
            U_init = - particle['log_prior'] - particle['log_like']
            momentum_init_gam = torch.tensor(0.,device=opts['torch_device'],dtype=opts['torch_type'])
            momentum_init_bet = torch.tensor(0.,device=opts['torch_device'],dtype=opts['torch_type'])
            momentum_init_uvn = torch.tensor(0.,device=opts['torch_device'],dtype=opts['torch_type'])
            momentum_gam = torch.tensor(0.,device=opts['torch_device'],dtype=opts['torch_type'])
            momentum_bet = torch.tensor(0.,device=opts['torch_device'],dtype=opts['torch_type'])
            momentum_uvn = torch.tensor(0.,device=opts['torch_device'],dtype=opts['torch_type'])
            for m in range(opts['smc_m_steps']):
                prop_part = copy.deepcopy(particle)
                #initial momentum
                coef = 2** -opts['precision_coef']
                if m_opts['sub_gam'].numel() > 0 :
                    d_gam = particle['sub_gam'].shape
                    momentum_init_gam = torch.randn(d_gam,device=opts['torch_device'],dtype=opts['torch_type']) * coef
                    prop_part['sub_gam'].requires_grad = True
                    if prop_part['sub_gam'].grad is not None:
                        prop_part['sub_gam'].grad.zero_()
                    
                if m_opts['sub_bet'].numel() > 0 :
                    d_bet = particle['sub_bet'].shape
                    momentum_init_bet = torch.randn(d_bet,device=opts['torch_device'],dtype=opts['torch_type']) * coef
                    prop_part['sub_bet'].requires_grad = True
                    if prop_part['sub_bet'].grad is not None:
                        prop_part['sub_bet'].grad.zero_()
                    
                if m_opts['sub_uvn'].numel() > 0 :
                    d_uvn = particle['sub_uvn'].shape
                    momentum_init_uvn = torch.randn(d_uvn,device=opts['torch_device'],dtype=opts['torch_type']) * coef
                    prop_part['sub_uvn'].requires_grad = True
                    if prop_part['sub_uvn'].grad is not None:
                        prop_part['sub_uvn'].grad.zero_()
                    
                # Get Grad_U        
                U = - sf.LogPrior(prop_part,m_opts) - sf.loglikelihood(IndivData,last_obs,prop_part,opts)
                U.backward()
                
                # momentum half step
                if m_opts['sub_gam'].numel() > 0 :
                    grad_U_gam = prop_part['sub_gam'].grad.data.clone()
                    momentum_gam = momentum_init_gam - 0.5 * opts['hmc_eps']* grad_U_gam
                    prop_part['sub_gam'].requires_grad = False
                    
                if m_opts['sub_bet'].numel() > 0 :
                    grad_U_bet = prop_part['sub_bet'].grad.data.clone()
                    momentum_bet = momentum_init_bet - 0.5 * opts['hmc_eps']* grad_U_bet
                    prop_part['sub_bet'].requires_grad = False
                    
                if m_opts['sub_uvn'].numel() > 0 :
                    grad_U_uvn = prop_part['sub_uvn'].grad.data.clone()
                    momentum_uvn = momentum_init_uvn - 0.5 * opts['hmc_eps']* grad_U_uvn
                    prop_part['sub_uvn'].requires_grad = False
                
                
                for s in range(opts['hmc_leaps']):
                    #Update position
                    if m_opts['sub_gam'].numel() > 0 :
                        prop_part['sub_gam'] = prop_part['sub_gam'] + opts['hmc_eps'] * momentum_gam
                        # check constraints
                        for d in range(prop_part['sub_gam'].numel()):
                            low_bound_check = prop_part['sub_gam'][d] < opts['m_opts']['gam_bound'][d,0]
                            high_bound_check = prop_part['sub_gam'][d] > opts['m_opts']['gam_bound'][d,1]
                            while low_bound_check or high_bound_check:
                                if low_bound_check:
                                    prop_part['sub_gam'][d] = 2 * opts['m_opts']['gam_bound'][d,0] - prop_part['sub_gam'][d]
                                    momentum_gam[d] = - momentum_gam[d]
                                elif high_bound_check:
                                    prop_part['sub_gam'][d] = 2 * opts['m_opts']['gam_bound'][d,1] - prop_part['sub_gam'][d]
                                    momentum_gam[d] = - momentum_gam[d]
                                low_bound_check = prop_part['sub_gam'][d] < opts['m_opts']['gam_bound'][d,0]
                                high_bound_check = prop_part['sub_gam'][d] > opts['m_opts']['gam_bound'][d,1]                      
                        prop_part['sub_gam'].requires_grad = True
                        if prop_part['sub_gam'].grad is not None:
                            prop_part['sub_gam'].grad.zero_()
                                    
                    if m_opts['sub_bet'].numel() > 0 :
                        prop_part['sub_bet'] = prop_part['sub_bet'] + opts['hmc_eps'] * momentum_bet
                        # check constraints
                        for d in range(prop_part['sub_bet'].numel()):
                            low_bound_check = prop_part['sub_bet'][d] < opts['m_opts']['bet_bound'][d,0]
                            high_bound_check = prop_part['sub_bet'][d] > opts['m_opts']['bet_bound'][d,1]
                            while low_bound_check or high_bound_check:
                                if low_bound_check:
                                    prop_part['sub_bet'][d] = 2 * opts['m_opts']['bet_bound'][d,0] - prop_part['sub_bet'][d]
                                    momentum_bet[d] = - momentum_bet[d]
                                elif high_bound_check:
                                    prop_part['sub_bet'][d] = 2 * opts['m_opts']['bet_bound'][d,1] - prop_part['sub_bet'][d]
                                    momentum_bet[d] = - momentum_bet[d]
                                low_bound_check = prop_part['sub_bet'][d] < opts['m_opts']['bet_bound'][d,0]
                                high_bound_check = prop_part['sub_bet'][d] > opts['m_opts']['bet_bound'][d,1]
                        prop_part['sub_bet'].requires_grad = True
                        if prop_part['sub_bet'].grad is not None:
                            prop_part['sub_bet'].grad.zero_()
                                    
                    if m_opts['sub_uvn'].numel() > 0 :
                        prop_part['sub_uvn'] = prop_part['sub_uvn'] + opts['hmc_eps'] * momentum_uvn
                        # check constraints
                        for d in range(prop_part['sub_uvn'].numel()):
                            low_bound_check = prop_part['sub_uvn'][d] < opts['m_opts']['uvn_bound'][d,0]
                            high_bound_check = prop_part['sub_uvn'][d] > opts['m_opts']['uvn_bound'][d,1]
                            while low_bound_check or high_bound_check:
                                if low_bound_check:
                                    prop_part['sub_uvn'][d] = 2 * opts['m_opts']['uvn_bound'][d,0] - prop_part['sub_uvn'][d]
                                    momentum_uvn[d] = - momentum_uvn[d]
                                elif high_bound_check:
                                    prop_part['sub_uvn'][d] = 2 * opts['m_opts']['uvn_bound'][d,1] - prop_part['sub_uvn'][d]
                                    momentum_uvn[d] = - momentum_uvn[d]
                                low_bound_check = prop_part['sub_uvn'][d] < opts['m_opts']['uvn_bound'][d,0]
                                high_bound_check = prop_part['sub_uvn'][d] > opts['m_opts']['uvn_bound'][d,1]
                        prop_part['sub_uvn'].requires_grad = True
                        if prop_part['sub_uvn'].grad is not None:
                            prop_part['sub_uvn'].grad.zero_()
                         
                    # Get Gradient of U
                    log_prior = sf.LogPrior(prop_part,m_opts)
                    log_like = sf.loglikelihood(IndivData,last_obs,prop_part,opts)
                    U = - log_prior - log_like
                    U.backward()
                    
                    # update momentum
                    if s < (opts['hmc_leaps']-1):
                        leap_coef = 1.
                        mom_coef = 1.
                    else: # for last loop
                        leap_coef = 0.5
                        mom_coef = -1.
                    
                    if m_opts['sub_gam'].numel() > 0 :
                        grad_U_gam = prop_part['sub_gam'].grad.data.clone()
                        momentum_gam = mom_coef * (momentum_gam - leap_coef * opts['hmc_eps']* grad_U_gam)
                        prop_part['sub_gam'].requires_grad = False
                        
                    if m_opts['sub_bet'].numel() > 0 :
                        grad_U_bet = prop_part['sub_bet'].grad.data.clone()
                        momentum_bet = mom_coef * (momentum_bet - leap_coef * opts['hmc_eps']* grad_U_bet)
                        prop_part['sub_bet'].requires_grad = False
                        
                    if m_opts['sub_uvn'].numel() > 0 :
                        grad_U_uvn = prop_part['sub_uvn'].grad.data.clone()
                        momentum_uvn = mom_coef * (momentum_uvn - leap_coef * opts['hmc_eps']* grad_U_uvn)
                        prop_part['sub_uvn'].requires_grad = False
                    
                
                # Acceptance probability
                p_acc = np.minimum(1, torch.exp(U_init - U.detach()  + 0.5 * (\
                                       (momentum_init_gam**2 - momentum_gam**2).sum()  \
                                       + (momentum_init_bet**2 - momentum_bet**2).sum() \
                                       + (momentum_init_uvn**2- momentum_uvn**2 ).sum() ) ).cpu().numpy() )
                if  np.random.random() < p_acc:
                    particle = prop_part
                    particle['log_prior'] = log_prior.cpu().detach().item()
                    particle['log_like'] = log_like.cpu().detach().item()
                    U_init = U.detach().clone()
                    accept = accept + 1
                tentatives = tentatives + 1
        else:
            for m in range(opts['smc_m_steps']):
                # %% Update params
                prop_part = copy.deepcopy(particle)
                prop_part['log_prior'] = 0.
                logQRatio = 0.
                # %% Gamma proposal
                if m_opts['sub_gam'].numel() > 0 :
                    step_coef = 2* m_opts['sub_gam'].shape[1] * opts['precision_base']**-1 * 2**(opts['precision_coef']) / np.maximum(m_opts['stats']['gam_sd'],0.000000001)**2
                    step_coef = np.maximum(step_coef,10)
                    step = np.random.gamma(step_coef,1/step_coef)
                    prop_part['sub_gam'] = prop_part['sub_gam'] * torch.tensor(step,device=prop_part['sub_gam'].device,dtype=opts['torch_type'])
                    for d in range(prop_part['sub_gam'].numel()):
                        low_bound_check = prop_part['sub_gam'][d] < opts['m_opts']['gam_bound'][d,0]
                        high_bound_check = prop_part['sub_gam'][d] > opts['m_opts']['gam_bound'][d,1]
                        # Reflect step if out of bounds
                        while low_bound_check or high_bound_check:
                            if low_bound_check:
                                prop_part['sub_gam'][d] = 2 * opts['m_opts']['gam_bound'][d,0] - prop_part['sub_gam'][d]
                            elif high_bound_check:
                                prop_part['sub_gam'][d] = 2 * opts['m_opts']['gam_bound'][d,1] - prop_part['sub_gam'][d]
                            low_bound_check = prop_part['sub_gam'][d] < opts['m_opts']['gam_bound'][d,0]
                            high_bound_check = prop_part['sub_gam'][d] > opts['m_opts']['gam_bound'][d,1]
                    
                    logQRatio = logQRatio + np.sum(- (2 * step_coef - 1) * np.log(step) + step_coef * (step - 1./step))
                    prop_part['log_prior'] = prop_part['log_prior'] + ( \
                        (m_opts['sub_gam'][0,:]-1) * prop_part['sub_gam'].log() \
                        - prop_part['sub_gam'] / m_opts['sub_gam'][1,:]).sum()
                # Beta proposal
                if m_opts['sub_bet'].numel() > 0 :
                    step_coef = 0.5 * np.maximum(opts['precision_base']**-1 * 2**(-opts['precision_coef']) * m_opts['stats']['bet_sd'] ,0.000000001) / m_opts['sub_bet'].shape[1]
                    step = np.random.normal(0, step_coef )
                    prop_part['sub_bet'] = prop_part['sub_bet'] + torch.tensor(step,device=prop_part['sub_bet'].device,dtype=opts['torch_type'])
                    # Reflect step if out of bounds
                    prop_part['sub_bet'] = (prop_part['sub_bet'].trunc() % 2 - (prop_part['sub_bet'] % 1) ).abs()
                    prop_part['log_prior'] = prop_part['log_prior'] + torch.sum( \
                        (m_opts['sub_bet'][0,:]-1) * torch.log(prop_part['sub_bet']) \
                        + (m_opts['sub_bet'][1,:]-1) * torch.log(1 - prop_part['sub_bet']) )
                # univariate normal param
                if m_opts['sub_uvn'].numel() > 0 :
                    step_coef = 0.5 * np.maximum(opts['precision_base']**-1 * 2**(-opts['precision_coef']) * m_opts['stats']['uvn_sd'],0.000000001) / m_opts['sub_uvn'].shape[1]
                    step = np.random.normal(0,step_coef)
                    prop_part['sub_uvn'] = prop_part['sub_uvn'] + torch.tensor(step,device=prop_part['sub_uvn'].device,dtype=opts['torch_type'])
                    prop_part['log_prior'] = prop_part['log_prior'] + torch.sum( \
                         -0.5 * ((prop_part['sub_uvn'] - m_opts['sub_uvn'][0,:]) / m_opts['sub_uvn'][1,:] )**2)
                # Accept / reject
                prop_part['log_like'] = sf.loglikelihood(IndivData,last_obs,prop_part,opts).cpu().detach().item()
                if np.log(np.random.rand()) <= prop_part['log_like'] - particle['log_like'] + prop_part['log_prior'] - particle['log_prior'] + logQRatio:
                    particle = prop_part
                    accept = accept + 1
                tentatives = tentatives + 1
            
            
        return (particle,accept/tentatives)
    except:
        raise Exception("Error while executing mutate function.")