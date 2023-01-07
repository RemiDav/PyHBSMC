# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:02:39 2019

@author: R
"""

import torch
  
# %% Probability of observation : p(observation|parameter)
def logProbaObs(obs,IndivData,particle,m_opts,return_all = False):
    # model: y = argmax Sum (B+g*e) * x^a + s * r
    # B ~ UVN
    # g ~ Gam
    # a ~ Bet
    if particle['model'] == 'ProbitExample':
        K = IndivData['x'][obs].shape[0]
        J = IndivData['x'][obs].shape[1]
        gam = particle['sub_gam']
        B = particle['sub_uvn']
        xa = IndivData['x'][obs] ** particle['sub_bet'][:,None]
        vx = torch.sum(B[:,None] * xa,0)
        xa_outer = torch.bmm(xa.unsqueeze(2),xa.unsqueeze(1) )
        cov = torch.sum(gam[0:K,None,None] * xa_outer,0)  + torch.eye(J,dtype=torch.double)
        chol_cov = torch.cholesky(cov,upper=True)
        sim_draws = vx + torch.matmul(m_opts['normdraws'][:,0:J],chol_cov)
        sim_draws = sim_draws - sim_draws.max(1)[0][:,None]
        exp_draws = torch.exp(sim_draws)
        lproba = torch.mean( exp_draws[:,IndivData['y'][obs]] / exp_draws.sum(1) ).log()
        return lproba
    else:
         raise Exception('ProbaObs: Model undefined')
         
def logLikelihood(IndivData,last_obs,particle,opts):
    if particle['model'] == 'ProbitExample':
        m_opts = opts['m_opts']
        K = IndivData['x'].shape[1]
        J = IndivData['x'].shape[2]
        gam = particle['sub_gam']
        B = particle['sub_uvn']
        
        xa = IndivData['x'][0:last_obs+1] ** particle['sub_bet'][:,None]
        vx = torch.sum(B[:,None] * xa,1)
        xa_outer = xa[:,:,:,None]*xa[:,:,None,:]
        cov = torch.sum(gam[0:K,None,None] * xa_outer,1) + torch.eye(J,dtype=torch.double)
        chol_cov = torch.cholesky(cov,upper=True)
        sim_draws = vx[:,None,:] + torch.matmul(m_opts['normdraws'][None,:,:],chol_cov[:,:,:])
        sim_draws = sim_draws - sim_draws.max(-1)[0][:,:,None]
        exp_draws = torch.exp(sim_draws)
        sum_exp = exp_draws.sum(-1)
        loglike = torch.tensor(0.)
        for obs in range(last_obs+1):
            loglike = loglike + torch.mean( exp_draws[obs,:,IndivData['y'][obs]] / sum_exp[obs] ,0).log()
    else:
         raise Exception('ProbaObs: Model undefined')
    return loglike