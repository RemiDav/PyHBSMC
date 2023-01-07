# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:02:39 2019

@author: R
"""

import numpy as np
import pandas as pd
import torch
  
# %% Probability of observation : p(observation|parameter)
def logProbaObs(obs,IndivData,particle,m_opts,return_all = False):
    ObsData = IndivData[obs]
    if particle['model'] == 'Model1':
        param = particle['sub_gam']
        lproba = - param[0] * torch.log(param[1]) + (param[0] - 1) * torch.log(ObsData) - ObsData / param[1] - torch.lgamma(param[0])
        return lproba
    else:
         raise Exception('ProbaObs: Model undefined')
         
def logLikelihood(IndivData,last_obs,particle,opts):
    np.seterr(divide='ignore')
    param = particle['sub_gam']
    loglike = - param[0] * torch.log(param[1]) + (param[0] - 1) * torch.log(IndivData[0:last_obs+1]) - IndivData[0:last_obs+1] / param[1] - torch.lgamma(param[0])
    loglike = loglike.sum()
    return loglike