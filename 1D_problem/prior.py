import sys, os
import numpy as np
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
from core.probability import GaussianElliptic2
import scipy.sparse.linalg as spsl
import torch
equ_nx = 100
domain_equ = Domain1D(n=equ_nx, mesh_type='CG', mesh_order=1)
k=1
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=(0.1/k), a_fun=(1.0/k), theta=1.0, boundary='Neumann'
    )
M=prior_measure.M
K=prior_measure.K
def csr_torch(matrix):
    Mcoo = matrix.tocoo()
    Mpt = torch.sparse.FloatTensor(torch.LongTensor([Mcoo.row.tolist(), Mcoo.col.tolist()]),
                                   torch.FloatTensor(Mcoo.data.astype(float)))
    Mpt = Mpt.to(torch.float64)
    return Mpt
############################################################################
M_torch=csr_torch(M)
K_torch=csr_torch(K)
from core.misc import trans2spnumpy, spnumpy2sptorch,sptorch2spnumpy
class PriorFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, prior):
        M = spnumpy2sptorch(prior.M)
        K = spnumpy2sptorch(prior.K)
        ctx.save_for_backward(input, M, K)
        m_vec = np.array(input.detach(), dtype=np.float32)
        val = prior.K.T @ spsl.spsolve(prior.M, prior.K @ m_vec.T)
        val=np.atleast_2d(val)
        if (val.shape[0]==1):
            val=val.T
        batchsize=val.shape[1]
        output=torch.zeros((batchsize))
        for i in range(batchsize):
            val_i=val[:,i]
            m_vec_i=m_vec[i,:]
            output[i]=val_i@m_vec_i
        output=output/2
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, M, K = ctx.saved_tensors
        M = sptorch2spnumpy(M)
        K = sptorch2spnumpy(K)
        m_vec = np.array(input, dtype=np.float32)
        val1 = spsl.spsolve(M, K.T @ spsl.spsolve(M, K @ m_vec.T))
        val=val1
        val = torch.tensor(val, dtype=torch.float32)
        a=grad_output * val
        a=a.T
        return a, None,None
prior=PriorFun.apply
def negtive_log_prior(x):
    return prior(x,prior_measure)
def prior_sample_torch(num,prior):
    sample = []
    for i in range(num):
        x = prior.generate_sample()
        sample.append(x)
        Sample = np.array(sample)
    Sample_torch = torch.from_numpy(Sample)
    return Sample_torch
z=prior_sample_torch(num=10,prior=prior_measure)
ans=negtive_log_prior(z)
######################################prior_GPU
M_dense=M.todense()
K_dense=K.todense()
M_torch_dense=torch.from_numpy(M_dense)
K_torch_dense=torch.from_numpy(K_dense)
M_torch_dense_inv=torch.inverse(M_torch_dense)
K_torch_dense_inv=torch.inverse(K_torch_dense)
def prior_dense(x):
    m_vec = x
    val=K_torch_dense.T@M_torch_dense_inv@K_torch_dense@m_vec.T
    val = torch.atleast_2d(val)
    if (val.shape[0] == 1):
        val = val.T
    batchsize = val.shape[1]
    output = torch.zeros((batchsize))
    for i in range(batchsize):
        val_i = val[:, i]
        m_vec_i = m_vec[i, :]
        output[i] = val_i @ m_vec_i
    output = output / 2
    return output









