import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1)
from core.misc import trans2spnumpy, construct_measurement_matrix
import fenics as fe
import scipy
###############################################################################
from core.model import Domain1D
from core.probability import GaussianElliptic2
for equ_nx in (15,20,30,50,75,100,200,300):
#[dim,20]
    dim=equ_nx+1
    eigs=np.zeros((dim,20))
    domain_equ = Domain1D(n=equ_nx, mesh_type='CG', mesh_order=1)
    u = fe.TrialFunction(domain_equ.function_space)
    v = fe.TestFunction(domain_equ.function_space)
    bb = fe.inner(u, v) * fe.dx
    M_ = fe.assemble(bb)
    M = trans2spnumpy(M_)
    for w in range(20):
        y=np.cos(w*np.pi*np.linspace(0,1,dim))
        y=y/np.sqrt(y@(M@y))
        eigs[:,w]=y
    np.save('RESULT/eig_vec_L2_'+str(dim)+'.npy',eigs)




