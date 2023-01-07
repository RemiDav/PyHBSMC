# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:29:15 2019

@author: R
"""

import numpy as np
import torch
import os
import model_functions as modfunc



# %% Initialize the particle system from prior
def InitParticles(opts,num_P=None):
    # %% Initialize particles
    if (not num_P):
        P = int(opts['P'])
    else:
        P = int(num_P)
    P = P * 8
    # Check boundaries: one row per parameter: [low bound,high bound]
    if opts['m_opts']['sub_gam'].numel() > 0 :
        if 'gam_bound' not in opts['m_opts'].keys():
            opts['m_opts']['gam_bound'] = np.tile([0.,np.inf],[opts['m_opts']['sub_gam'].numel(),1])
    if opts['m_opts']['sub_bet'].numel() > 0 :
        if 'bet_bound' not in opts['m_opts'].keys():
            opts['m_opts']['bet_bound'] = np.tile([0.,1.],[opts['m_opts']['sub_bet'].numel(),1])
    if opts['m_opts']['sub_uvn'].numel() > 0 :
        if 'uvn_bound' not in opts['m_opts'].keys():
            opts['m_opts']['uvn_bound'] = np.tile([-np.inf,np.inf],[opts['m_opts']['sub_uvn'].numel(),1])
    

    Particles = {}
    Particles['active'] = True
    Particles['model'] = opts['m_opts']['name']
    Particles['m_opts'] = opts['m_opts']
    Particles['particle'] = np.empty((opts['G'],P),dtype='object')
    Particles['logweights'] = np.zeros((opts['G'],P))
    Particles['log_marg_like_obs'] = np.zeros((opts['G'],0))
    Particles['log_marg_like'] = np.zeros(opts['G'])
    for g in range(opts['G']):
        for p in range(P):
            Particles['particle'][g,p] = DrawParticle(opts, opts['m_opts'])
    return Particles

# %% Draw a particle from prior          
def DrawParticle(opts, m_opts ):
    device = opts['torch_device']
    # %% initialize structure
    particle = {}
    particle['device'] = device
    particle['dtype'] = opts['torch_type']
    particle['model'] = m_opts['name'];
    particle['log_like'] = 0.
    particle['log_prior'] = 0.
    # %% Draw Parameters
    # Gamma param
    if m_opts['sub_gam'].numel() > 0 :
        gam_param = np.random.gamma(m_opts['sub_gam'][0].cpu().numpy(),m_opts['sub_gam'][1].cpu().numpy())
        particle['sub_gam'] = torch.tensor(gam_param, dtype=opts['torch_type']).to(device)
    # Beta param
    if m_opts['sub_bet'].numel() > 0 :
        bet_param = np.random.beta(m_opts['sub_bet'][0].cpu().numpy(),m_opts['sub_bet'][1].cpu().numpy())
        particle['sub_bet'] = torch.tensor(bet_param, dtype=opts['torch_type']).to(device)
    # univariate normal param
    if m_opts['sub_uvn'].numel() > 0 :
        uvn_param = np.random.normal(m_opts['sub_uvn'][0].cpu().numpy(),m_opts['sub_uvn'][1].cpu().numpy())
        particle['sub_uvn'] = torch.tensor(uvn_param, dtype=opts['torch_type']).to(device)
    # %% Return result
    particle['log_prior'] = LogPrior(particle,m_opts).cpu().detach().item()
    return particle

# %% Compute log prior
def LogPrior(particle,m_opts):
    # %% Get the prior for each parameter types
    gam_prior = torch.tensor(0.,dtype = particle['dtype'], device=particle['device'])
    bet_prior = torch.tensor(0.,dtype = particle['dtype'], device=particle['device'])
    uvn_prior = torch.tensor(0.,dtype = particle['dtype'], device=particle['device'])
    # Gamma param
    if m_opts['sub_gam'].numel() > 0 :
        gam_prior = (m_opts['sub_gam'][0] - 1) * torch.log(particle['sub_gam'])  \
            - particle['sub_gam'] / m_opts['sub_gam'][1]  #\
            # - torch.lgamma(m_opts['sub_gam'][0]) \
            # - m_opts['sub_gam'][0] * torch.log(m_opts['sub_gam'][1])
    # Beta param
    if m_opts['sub_bet'].numel() > 0 :
        bet_prior = (m_opts['sub_bet'][0] - 1) * torch.log(particle['sub_bet']) \
            + (m_opts['sub_bet'][1] - 1) * torch.log(1.-particle['sub_bet']) #\
            # + torch.lgamma(m_opts['sub_gam']).sum(0) \
            # - + torch.lgamma(m_opts['sub_gam'].sum(0))
    # univariate normal param
    if m_opts['sub_uvn'].numel() > 0 :
        uvn_prior = - 0.5 * ( (particle['sub_uvn'] - m_opts['sub_uvn'][0]) / m_opts['sub_uvn'][1])**2 # \
            # - torch.log(m_opts['sub_uvn'][1] * ) - 0.9189385332046727
    log_prior = gam_prior.sum() + bet_prior.sum() + uvn_prior.sum()
    # %% Return result
    return log_prior

# %% Get Particle Statistics
def ParticleStats(SubParticles,m_opts):
    # %% The function vectorises particles then average them by group
    stats = [None] * SubParticles['particle'].shape[0]
    for g in range(SubParticles['particle'].shape[0]):
        stats[g] = {}
        weights = np.exp(SubParticles['logweights'][g])
        if m_opts['sub_gam'].numel() > 0 :
            gam_vect = np.zeros((SubParticles['particle'].shape[1],m_opts['sub_gam'].shape[1]))
            for p in range(SubParticles['particle'].shape[1]):
                gam_vect[p,:] = SubParticles['particle'][g,p]['sub_gam'].cpu().numpy()
            stats[g]['gam_mean'] = np.average(gam_vect,axis=0,weights=weights)
            variance = np.average((gam_vect-stats[g]['gam_mean'])**2,axis=0,weights=weights)
            stats[g]['gam_sd'] = np.sqrt(variance)
        if m_opts['sub_bet'].numel() > 0 :
            bet_vect = np.zeros((SubParticles['particle'].shape[1],m_opts['sub_bet'].shape[1]))
            for p in range(SubParticles['particle'].shape[1]):
                bet_vect[p,:] = SubParticles['particle'][g,p]['sub_bet'].cpu().numpy()
            stats[g]['bet_mean'] = np.average(bet_vect,axis=0,weights=weights)
            variance = np.average((bet_vect-stats[g]['bet_mean'])**2,axis=0,weights=weights)
            stats[g]['bet_sd'] = np.sqrt(variance)
        if m_opts['sub_uvn'].numel() > 0 :
            uvn_vect = np.zeros((SubParticles['particle'].shape[1],m_opts['sub_uvn'].shape[1]))
            for p in range(SubParticles['particle'].shape[1]):
                uvn_vect[p,:] = SubParticles['particle'][g,p]['sub_uvn'].cpu().numpy()
            stats[g]['uvn_mean'] = np.average(uvn_vect,axis=0,weights=weights)
            variance = np.average((uvn_vect-stats[g]['uvn_mean'])**2,axis=0,weights=weights)
            stats[g]['uvn_sd'] = np.sqrt(variance)
    # %% Return results
    return stats

# %% Compute log likelihood
def loglikelihood(IndivData,last_obs,particle,opts):
    if "logLikelihood" in dir(modfunc):
        loglike = modfunc.logLikelihood(IndivData,last_obs,particle,opts)
    else:
        np.seterr(divide='ignore')
        loglike = 0.
        for obs in range(last_obs+1):
            loglike = loglike + modfunc.logProbaObs(obs,IndivData,particle,opts['m_opts'])
    return loglike

######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
# %% Super Particles Initialization
    
def InitSupParticles(opts,num_P=None):
    # %% Initialize particles
    if (not num_P):
        P = int(opts['P'])
    else:
        P = int(num_P)
    P = P * 8
    SupParticles = {}
    SupParticles['tag'] = opts['tag']
    SupParticles['h_opts'] = opts['h_opts']
    SupParticles['particle'] = np.empty((opts['G'],P),dtype='object')
    SupParticles['logweights'] = np.zeros((opts['G'],P))
    SupParticles['log_marg_like_subj'] = np.zeros((opts['G'],0))
    SupParticles['log_marg_like'] = np.zeros(opts['G'])
    SupParticles['subj_list'] = []
    for g in range(opts['G']):
        for p in range(P):
            SupParticles['particle'][g,p] = DrawSupParticle(opts)
    return SupParticles


# %% Draw a super particle from prior          
def DrawSupParticle(opts):
    h_opts = opts['h_opts']
    device = opts['torch_device']
    # %% initialize structure
    sparticle = {}
    sparticle['device'] = device
    sparticle['dtype'] = opts['torch_type']
    sparticle['model'] = opts['tag'];
    sparticle['log_hlike'] = 0.
    sparticle['log_hprob_y'] = torch.tensor([0.],dtype=opts['torch_type'])
    sparticle['log_hprior'] = torch.tensor([], dtype=opts['torch_type'])
    sparticle['sub_clust'] = np.zeros(1,dtype='int')
    # %% init. parameters vectors
    # Gamma param
    if h_opts['base_gam'].numel() > 0 :
        h_gam = np.empty((0,2,h_opts['base_gam'].shape[1]))
        sparticle['h_gam'] = torch.tensor(h_gam, dtype=opts['torch_type']).to(device)
        sparticle['const_prior_gam'] = - torch.lgamma(h_opts['base_gam'][[0,2],:]) \
            - h_opts['base_gam'][[0,2],:] * torch.log(h_opts['base_gam'][[1,3],:])
        sparticle['const_prior_gam'] = sparticle['const_prior_gam'].sum()
    # Beta param
    if h_opts['base_bet'].numel() > 0 :
        h_bet = np.empty((0,2,h_opts['base_bet'].shape[1]))
        sparticle['h_bet'] = torch.tensor(h_bet, dtype=opts['torch_type']).to(device)
        sparticle['const_prior_bet'] = - torch.lgamma(h_opts['base_bet'][[0,2],:]) \
            - h_opts['base_bet'][[0,2],:] * torch.log(h_opts['base_bet'][[1,3],:])
        sparticle['const_prior_bet'] = sparticle['const_prior_bet'].sum()
    # univariate normal param
    if h_opts['base_uvn'].numel() > 0 :
        h_uvn = np.empty((0,2,h_opts['base_uvn'].shape[1]))
        sparticle['h_uvn'] = torch.tensor(h_uvn, dtype=opts['torch_type']).to(device)
        sparticle['const_prior_uvn'] = - torch.log(h_opts['base_uvn'][1] ) - 0.9189385332046727 \
            - torch.lgamma(h_opts['base_uvn'][2,:]) \
            - h_opts['base_uvn'][2,:] * torch.log(h_opts['base_uvn'][3,:])
        sparticle['const_prior_uvn'] = sparticle['const_prior_uvn'].sum()
    # %% Draw parameters for cluster 0
    sparticle = AddCluster(sparticle,opts)
    # %% Return result
    return sparticle

def AddCluster(sparticle,opts):
    # %% Check if max list clust == size H and add 1 if not
    new_clust = False
    num_clust = sparticle['sub_clust'].max() + 1
    if opts['h_opts']['base_gam'].numel() > 0 :
        while num_clust > sparticle['h_gam'].shape[0]:
            b_param = opts['h_opts']['base_gam'].cpu().numpy()
            h_param = torch.tensor( np.random.gamma(b_param[[0,2],:],b_param[[1,3],:]) , device = opts['torch_device'], dtype=opts['torch_type'])
            sparticle['h_gam'] = torch.cat((sparticle['h_gam'],h_param[None,:,:]))
            new_clust = True
    if opts['h_opts']['base_bet'].numel() > 0 :
        while num_clust > sparticle['h_bet'].shape[0]:
            b_param = opts['h_opts']['base_bet'].cpu().numpy()
            h_param = torch.tensor( np.random.gamma(b_param[[0,2],:],b_param[[1,3],:]) , device = opts['torch_device'], dtype=opts['torch_type'])
            sparticle['h_bet'] = torch.cat((sparticle['h_bet'],h_param[None,:,:]))
            new_clust = True
    if opts['h_opts']['base_uvn'].numel() > 0 :
        while num_clust > sparticle['h_uvn'].shape[0]:
            b_param = opts['h_opts']['base_uvn'].cpu().numpy()
            h_param1 = torch.tensor( np.random.normal(b_param[0,:],b_param[1,:]) , device = opts['torch_device'], dtype=opts['torch_type'])
            h_param2 = torch.tensor( np.random.gamma(b_param[2,:],b_param[3,:]) , device = opts['torch_device'], dtype=opts['torch_type'])
            h_param = torch.stack((h_param1,h_param2))
            sparticle['h_uvn'] = torch.cat((sparticle['h_uvn'],h_param[None,:,:]))
            new_clust = True
    if new_clust:
        while num_clust > sparticle['log_hprior'].size():
            lprior = LogHPrior(sparticle,sparticle['log_hprior'].size(),opts['h_opts'])
            sparticle['log_hprior'] = torch.cat((sparticle['log_hprior'],lprior[None]))
    return sparticle


# %% Compute log prior for hyper params P(H_c)
def LogHPrior(sparticle,clust,h_opts):
    # %% Get the prior for each parameter types
    gam_prior = 0.
    bet_prior = 0.
    uvn_prior = 0.
    # Gamma param
    if h_opts['base_gam'].numel() > 0 :
        gam_prior = (h_opts['base_gam'][[0,2],:] - 1) * torch.log(sparticle['h_gam'][clust])  \
            - sparticle['h_gam'][clust] / h_opts['base_gam'][[1,3],:]
        gam_prior = gam_prior.sum() + sparticle['const_prior_gam']
    # Beta param
    if h_opts['base_bet'].numel() > 0 :
        bet_prior = (h_opts['base_bet'][[0,2],:] - 1) * torch.log(sparticle['h_bet'][clust])  \
            - sparticle['h_bet'][clust] / h_opts['base_bet'][[1,3],:]
        bet_prior = bet_prior.sum() + sparticle['const_prior_bet']
    # univariate normal param
    if h_opts['base_uvn'].numel() > 0 :
        uvn_prior1 = - 0.5 * ( (sparticle['h_uvn'][clust][0] - h_opts['base_uvn'][0]) / h_opts['base_uvn'][1])**2 # \            
        uvn_prior2 = (h_opts['base_uvn'][2] - 1) * torch.log(sparticle['h_uvn'][clust][1])  \
            - sparticle['h_uvn'][clust][1] / h_opts['base_uvn'][3]
        uvn_prior = uvn_prior1.sum() + uvn_prior2.sum() + sparticle['const_prior_uvn']
    log_prior = gam_prior + bet_prior + uvn_prior
    # %% Return result
    return log_prior

# %% Get the log p(y_i|H,c_i)
def logHProbaObs(subj,VectSubParticles,sparticle,h_opts):
    # %% compute P(theta|H)
    clust = sparticle['sub_clust'][subj]
    h_Ptheta_gam = torch.tensor(1.,dtype=sparticle['dtype'],device=sparticle['log_hprob_y'].device)
    h_Ptheta_bet = torch.tensor(1.,dtype=sparticle['dtype'],device=sparticle['log_hprob_y'].device)
    h_Ptheta_uvn = torch.tensor(1.,dtype=sparticle['dtype'],device=sparticle['log_hprob_y'].device)
#    h_logPrior_gam = torch.tensor(1.,dtype=torch.double,device=sparticle['log_hprob_y'].device)
#    h_logPrior_bet = torch.tensor(0.,dtype=torch.double,device=sparticle['log_hprob_y'].device)
#    h_logPrior_uvn = torch.tensor(0.,dtype=torch.double,device=sparticle['log_hprob_y'].device)
    if h_opts['base_gam'].numel() > 0 :
        vec_gam = VectSubParticles['vparticles'][subj]['vec_gam']
#        prior_const = - torch.lgamma(sparticle['h_gam'][clust,0,:]) \
#            - sparticle['h_gam'][clust,0,:] * torch.log(sparticle['h_gam'][clust,1,:])
#        prior_const = prior_const.sum()
#        log_paramPrior = (sparticle['h_gam'][clust,0,:] - 1) * torch.log(vec_gam)  \
#            - vec_gam / sparticle['h_gam'][clust,1,:]
#        h_logPrior_gam = log_paramPrior.sum(1) + prior_const
        h_Ptheta_gam = GamPDF(vec_gam, sparticle['h_gam'][clust,0,:], sparticle['h_gam'][clust,1,:]).prod(1)
    if h_opts['base_bet'].numel() > 0 :
        vec_bet = VectSubParticles['vparticles'][subj]['vec_bet']
#        prior_const = - torch.lgamma(sparticle['h_bet'][clust,0,:]) \
#            - torch.lgamma(sparticle['h_bet'][clust,1,:]) \
#            + torch.lgamma(sparticle['h_bet'][clust].sum(0))
#        prior_const = prior_const.sum()
#        log_paramPrior = (sparticle['h_bet'][clust,0,:] - 1) * torch.log(vec_bet)  \
#            +  (sparticle['h_bet'][clust,1,:] - 1) * torch.log(1. - vec_bet)
#        h_logPrior_bet = log_paramPrior.sum(1) + prior_const
        h_Ptheta_bet = BetPDF(vec_bet, sparticle['h_bet'][clust,0,:], sparticle['h_bet'][clust,1,:]).prod(1)
    if h_opts['base_uvn'].numel() > 0 :
        vec_uvn = VectSubParticles['vparticles'][subj]['vec_uvn']
#        prior_const = - torch.log(sparticle['h_uvn'][clust,1,:] ) - 0.9189385332046727 
#        prior_const = prior_const.sum()
#        log_paramPrior = - 0.5 * ((vec_uvn - sparticle['h_uvn'][clust,0,:])  \
#            /  sparticle['h_uvn'][clust,1,:]) ** 2.
#        h_logPrior_uvn = log_paramPrior.sum(1) + prior_const
        h_Ptheta_uvn = UVNPDF(vec_uvn, sparticle['h_uvn'][clust,0,:], sparticle['h_uvn'][clust,1,:]).prod(1)
    # %% Prior switching -> P(y|H)   
    log_w =  - VectSubParticles['vparticles'][subj]['m_logPrior']
    w = h_Ptheta_gam * h_Ptheta_bet * h_Ptheta_uvn * log_w.exp()
    has_nan = torch.isnan(w)
    if has_nan.any():
        print("\n%d NaNs in weights : P(th|H)/P(th|M)" % (has_nan.sum()))
    log_hproby =  VectSubParticles['m_logML'][subj] + w.mean().log()
    if torch.isnan(log_hproby).any():
        print("\nNaN in log_hproby : p(y_i|H,c_i) =  ",log_hproby.item())
#        print("log_w.mean():", log_w.mean())
#        print("log_w.min():", log_w.min())
        print("w.mean().log():", w.mean().log())
        print("w.mean():", w.mean())
        print("w.min():", w.min())
        print("h_gam:",sparticle['h_gam'][clust])
        print("h_bet:",sparticle['h_bet'][clust])
        print("h_uvn:",sparticle['h_uvn'][clust])
    return log_hproby

def getHProbs(list_subj,VectSubParticles,sparticle,h_opts):
    log_likes_list = torch.zeros(list_subj.shape,dtype=sparticle['dtype'])
    for i in range(len(list_subj)):
        log_likes_list[i] = logHProbaObs(list_subj[i],VectSubParticles,sparticle,h_opts)
    return log_likes_list
    

# %% Prepare SubParticles for Hierarchical Stage
def VectorizeSubParticles(AllSubParticles,opts):
    # %%
    num_subj = len(AllSubParticles)
    VectSubParticles = {}
    VectSubParticles['vparticles'] = np.empty(num_subj,dtype='object')
    VectSubParticles['m_logML'] = torch.zeros(num_subj,dtype=opts['torch_type'])
    VectSubParticles['sourcefile'] = []
    for subj in range(num_subj):
        SubParticles = AllSubParticles[subj]
        G,P = SubParticles['particle'].shape
        VectSubParticles['m_logML'][subj] = SubParticles['log_marg_like'].mean()
        VectSubParticles['vparticles'][subj] = {}
        VectSubParticles['sourcefile'].append(AllSubParticles[subj]['sourcefile'] )
        has_gam = SubParticles['m_opts']['sub_gam'].numel() > 0 
        has_bet = SubParticles['m_opts']['sub_bet'].numel() > 0 
        has_uvn = SubParticles['m_opts']['sub_uvn'].numel() > 0 
        if has_gam:
            VectSubParticles['vparticles'][subj]['vec_gam'] = torch.zeros((G*P,SubParticles['m_opts']['sub_gam'].size(1)),
                            dtype=opts['torch_type'],device=opts['torch_device'])
        if has_bet:
            VectSubParticles['vparticles'][subj]['vec_bet'] = torch.zeros((G*P,SubParticles['m_opts']['sub_bet'].size(1)),
                            dtype=opts['torch_type'],device=opts['torch_device'])
        if has_uvn:
            VectSubParticles['vparticles'][subj]['vec_uvn'] = torch.zeros((G*P,SubParticles['m_opts']['sub_uvn'].size(1)),
                            dtype=opts['torch_type'],device=opts['torch_device'])
        for g in range(G):
            for p in range(P):
                if has_gam :
                    VectSubParticles['vparticles'][subj]['vec_gam'][g*P+p,:] = SubParticles['particle'][g,p]['sub_gam'].type(opts['torch_type'])
                if has_bet :
                    VectSubParticles['vparticles'][subj]['vec_bet'][g*P+p,:] = SubParticles['particle'][g,p]['sub_bet'].type(opts['torch_type'])
                if has_uvn :
                    VectSubParticles['vparticles'][subj]['vec_uvn'][g*P+p,:] = SubParticles['particle'][g,p]['sub_uvn'].type(opts['torch_type'])
        # %% Compute exact log prior (model M):
        VectSubParticles['vparticles'][subj]['m_logPrior'] = torch.zeros(G*P,dtype=opts['torch_type'],device=opts['torch_device'])
        if has_gam :
            prior_const = - torch.lgamma(SubParticles['m_opts']['sub_gam'][0,:]) \
                - SubParticles['m_opts']['sub_gam'][0,:] * torch.log(SubParticles['m_opts']['sub_gam'][1,:])
            prior_const = prior_const.sum()
            vec_gam = VectSubParticles['vparticles'][subj]['vec_gam']
            log_paramPrior = (SubParticles['m_opts']['sub_gam'][0,:] - 1) * torch.log(vec_gam)  \
                - vec_gam / SubParticles['m_opts']['sub_gam'][1,:].double()
            VectSubParticles['vparticles'][subj]['m_logPrior'] = VectSubParticles['vparticles'][subj]['m_logPrior'] + \
                log_paramPrior.sum(1) + prior_const
        if has_bet :
            prior_const = - torch.lgamma(SubParticles['m_opts']['sub_bet'][0,:]) \
                - torch.lgamma(SubParticles['m_opts']['sub_bet'][1,:]) \
                + torch.lgamma(SubParticles['m_opts']['sub_bet'].sum(0))
            prior_const = prior_const.sum()
            vec_bet = VectSubParticles['vparticles'][subj]['vec_bet']
            log_paramPrior = (SubParticles['m_opts']['sub_bet'][0,:] - 1) * torch.log(vec_bet)  \
                +  (SubParticles['m_opts']['sub_bet'][1,:] - 1) * torch.log(1. - vec_bet)
            VectSubParticles['vparticles'][subj]['m_logPrior'] = VectSubParticles['vparticles'][subj]['m_logPrior'] + \
                log_paramPrior.sum(1) + prior_const
        if has_uvn :
            prior_const = - torch.log(SubParticles['m_opts']['sub_uvn'][1,:] ) - 0.9189385332046727 
            prior_const = prior_const.sum()
            vec_uvn = VectSubParticles['vparticles'][subj]['vec_uvn']
            log_paramPrior = - 0.5 * ((vec_uvn - SubParticles['m_opts']['sub_uvn'][0,:])  \
                /  SubParticles['m_opts']['sub_uvn'][1,:]) ** 2.
            VectSubParticles['vparticles'][subj]['m_logPrior'] = VectSubParticles['vparticles'][subj]['m_logPrior'] + \
                log_paramPrior.sum(1) + prior_const
    return VectSubParticles
            

###############################################
    

# %% Differentiable Gamma PDF module
class GamPDF_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, th, a, b):
        const = b**(-a) * torch.exp( -torch.lgamma(a) )
        pdf = th**(a-1) * torch.exp(-th/b)  * const
        ctx.save_for_backward(pdf,th,a,b)
        return pdf
    @staticmethod
    def backward(ctx, grad_output):
        pdf,th,a,b = ctx.saved_tensors
        grad_a = pdf * ( torch.log(th) - torch.log(b) - torch.digamma(a) )
        grad_b = pdf * (th / b - a ) / b
        return (None, grad_a * grad_output, grad_b * grad_output)

def GamPDF(th, a, b):
    return GamPDF_.apply(th, a, b)

# %% Differentiable Beta PDF module
class BetPDF_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, th, a, b):
        const = torch.exp( - torch.lgamma(a) - torch.lgamma(b) + torch.lgamma(a+b) )
        pdf = th**(a-1) * (1-th)**(b-1)  * const
        ctx.save_for_backward(pdf,th,a,b)
        return pdf
    @staticmethod
    def backward(ctx, grad_output):
        pdf,th,a,b = ctx.saved_tensors
        dg_ab = torch.digamma(a+b) 
        grad_a = pdf * ( torch.log(th) - torch.digamma(a) +  dg_ab)
        grad_b = pdf * ( torch.log(1-th) - torch.digamma(b) +  dg_ab)
        return (None, grad_a * grad_output, grad_b * grad_output)
    
def BetPDF(th, a, b):
    return BetPDF_.apply(th, a, b)

# %% Differentiable UVN PDF module
class UVNPDF_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, th, m, s):
        th_norm = (th - m) / s
        const = 1 / (s * 2.5066282746310002)
        pdf = const * torch.exp(-0.5 * th_norm**2)
        ctx.save_for_backward(pdf,th_norm,m,s)
        return pdf
    @staticmethod
    def backward(ctx, grad_output):
        pdf,th_norm,m,s = ctx.saved_tensors
        grad_m = - pdf * th_norm / s
        grad_s = pdf * (th_norm**2-1) / s
        return (None, grad_m * grad_output, grad_s * grad_output)

def UVNPDF(th, m, s):
    return UVNPDF_.apply(th, m, s)


# %% Get Particle Statistics
def SParticleStats(SupParticles,h_opts):
    # %% The function vectorises particles then average them by group
    stats = [None] * SupParticles['particle'].shape[0]
    for g in range(SupParticles['particle'].shape[0]):
        stats[g] = {}
        weights = np.exp(SupParticles['logweights'][g])
        has_gam = False
        has_bet = False
        has_uvn = False
        if h_opts['base_gam'].numel() > 0 :
            has_gam = True
            h_gam_vect = np.zeros((SupParticles['particle'].shape[1],2,h_opts['base_gam'].shape[1]))
        if h_opts['base_bet'].numel() > 0 :
            has_bet = True
            h_bet_vect = np.zeros((SupParticles['particle'].shape[1],2,h_opts['base_bet'].shape[1]))
        if h_opts['base_uvn'].numel() > 0 :
            has_uvn = True
            h_uvn_vect = np.zeros((SupParticles['particle'].shape[1],2,h_opts['base_uvn'].shape[1]))
            
            
        for p in range(SupParticles['particle'].shape[1]):
            clust_list,clust_counts = np.unique(SupParticles['particle'][g,p]['sub_clust'],return_counts=True)
            clust_weights = clust_counts / clust_counts.sum()
            for c,w in enumerate(clust_weights):
                if has_gam:
                    h_gam_vect[p] = h_gam_vect[p] + w * SupParticles['particle'][g,p]['h_gam'][c].cpu().numpy()
                if has_bet:
                    h_bet_vect[p] = h_bet_vect[p] + w * SupParticles['particle'][g,p]['h_bet'][c].cpu().numpy()
                if has_uvn:
                    h_uvn_vect[p] = h_uvn_vect[p] + w * SupParticles['particle'][g,p]['h_uvn'][c].cpu().numpy()
            
        if has_gam:
            stats[g]['gam_mean'] = np.average(h_gam_vect,axis=0,weights=weights)
            variance = np.average((h_gam_vect-stats[g]['gam_mean'])**2,axis=0,weights=weights)
            stats[g]['gam_sd'] = np.sqrt(variance)
        if has_bet:
            stats[g]['bet_mean'] = np.average(h_bet_vect,axis=0,weights=weights)
            variance = np.average((h_bet_vect-stats[g]['bet_mean'])**2,axis=0,weights=weights)
            stats[g]['bet_sd'] = np.sqrt(variance)
        if has_uvn:
            stats[g]['uvn_mean'] = np.average(h_uvn_vect,axis=0,weights=weights)
            variance = np.average((h_uvn_vect-stats[g]['uvn_mean'])**2,axis=0,weights=weights)
            stats[g]['uvn_sd'] = np.sqrt(variance)
            
    # %% Return results
    return stats