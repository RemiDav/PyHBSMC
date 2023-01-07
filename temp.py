# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:25:27 2020

@author: R Daviet
"""

import torch
# %% loop
A = torch.randn((3,3),dtype=torch.double)
for i in range(1000):
    A.requires_grad = True
    b = A[0,0] + torch.tensor(float('nan'))
    b.backward()
    A.requires_grad = False
    A[0,0] = 1.
    A = A / 2 + torch.randn((3,3),dtype=torch.double)
    if torch.isnan(A).any():
        print("NaN detected (i=%d): " % i, torch.isnan(A).any())
        print("A:", A)
    