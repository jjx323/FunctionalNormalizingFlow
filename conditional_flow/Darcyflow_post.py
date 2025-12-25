import fenics as fe
import sys, os
import numpy as np
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.probability import GaussianElliptic2
import scipy.sparse.linalg as spsl
import torch
from core.misc import load_expre, smoothing
from common_PDE import EquSolver, ModelDarcyFlow
from core.model import Domain2D
noise_level = 0.05
from core.noise import NoiseGaussianIID
DATA_DIR = './DATA/'
RESULT_DIR = './RESULT/MAP/'
equ_nx = 20
domain_equ = Domain2D(nx=equ_nx, ny=equ_nx,mesh_type='CG', mesh_order=1)
## setting the prior measure
k=1
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=(0.1/k), a_fun=(1.0/k), theta=1.0, boundary='Neumann'
    )
## loading coordinates of the measurement points
measurement_points = np.load(DATA_DIR + "measurement_points_2D.npy")
## setting the forward problem
f_expre = load_expre(DATA_DIR + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f_expre, degree=2), domain_equ.function_space)
class PostFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, model):
        m_vec = np.array(input.detach(), dtype=np.float32)
        val = model.prior.K.T @ spsl.spsolve(model.prior.M, model.prior.K @ m_vec.T)
        val=np.atleast_2d(val)
        if (val.shape[0]==1):
            val=val.T
        batchsize = val.shape[1]
        datadim=val.shape[0]
        output1=torch.zeros((batchsize))
        for i in range(batchsize):
            val_i=val[:,i]
            m_vec_i=m_vec[i,:]
            output1[i]=val_i@m_vec_i
        output1=output1/2
        output2 = torch.zeros((batchsize))
        for i in range(batchsize):
            m_vec_i=m_vec[i,:]
            model.update_m(m_vec_i.flatten(), update_sol=True)
            output2[i] = model.loss_residual()
        output=output1+output2
        M = model.prior.M
        K = model.prior.K
        m_vec = np.array(input, dtype=np.float32)
        val1 = spsl.spsolve(M, K.T @ spsl.spsolve(M, K @ m_vec.T))
        batchsize = m_vec.shape[0]
        datadim = m_vec.shape[1]
        val2 = np.zeros((datadim, batchsize))
        for i in range(batchsize):
            m_vec_i = m_vec[i, :]
            val2[:, i] = model.gradient(m_vec=m_vec_i)[1]
        val = val1 + val2
        val=M@val
        val = torch.tensor(val, dtype=torch.float32)
        ctx.save_for_backward(input, val)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, val = ctx.saved_tensors
        a=grad_output * val
        a=a.T
        return a, None
post=PostFun.apply
sample=[]
def prior_sample_torch(num,prior):
    for i in range(num):
        x = prior.generate_sample()
        sample.append(x)
        Sample = np.array(sample)
    Sample_torch = torch.from_numpy(Sample)
    return Sample_torch






