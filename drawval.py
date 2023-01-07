# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 01:11:53 2018

@author: R Daviet
"""

import numpy as np

def choice_vect(source,weights):
    # Draw N choices, each picked among K options
    # source: K x N ndarray
    # weights: K x N ndarray or K vector of probabilities
    weights = np.atleast_2d(weights)
    source = np.atleast_2d(source)
    N = source.shape[1]
    if weights.shape[0] == 1 and N > 1:
        weights = np.tile(weights.transpose(),(1,N))
    cum_weights = weights.cumsum(axis=0)
    cum_weights = cum_weights / cum_weights[-1,:]
    unif_draws = np.random.rand(1,N)
    choices = (unif_draws < cum_weights)
    bool_indices = choices > np.vstack( (np.zeros((1,N),dtype='bool'),choices ))[0:-1,:]
    return source.T[bool_indices.T]

def draw_idx(source,weight,n):
    # Draw n elements from source with replacement with weights
    weight = np.array(weight).flatten()
    cum_weights = np.atleast_2d(weight.cumsum() / np.sum(weight))
    cum_weights = np.tile(cum_weights.T,(1,n))
    unif_draws = np.random.rand(1,n)
    choices = (unif_draws < cum_weights)
    source = np.tile(np.atleast_2d(source).T,(1,n))
    bool_indices = choices > np.vstack( (np.zeros((1,n),dtype='bool'),choices ))[0:-1,:]
    return source[bool_indices].flatten()

def choice_vect_unif(K_vect):
    # Draw N choices, each picked among K options
    # source:  N ndarray (K1,K2,...,KN)
    N = np.size(K_vect)
    draws = np.random.rand(N) *  K_vect
    return np.floor(draws).astype(int)