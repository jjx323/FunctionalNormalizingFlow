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
from core.noise import NoiseGaussianIID
noise_level = 0.05
DATA_DIR = './DATA/'
equ_nx = 20
domain_equ = Domain2D(nx=equ_nx, ny=equ_nx,mesh_type='CG', mesh_order=1)
mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'CG', 1)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')
truth_fun = fe.interpolate(truth_fun, domain_equ.function_space)
k=1
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=(0.1/k), a_fun=(1.0/k), theta=1.0, boundary='Neumann'
    )
measurement_points = np.load(DATA_DIR + "measurement_points_2D.npy")
f_expre = load_expre(DATA_DIR + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f_expre, degree=2), domain_equ.function_space)
equ_solver = EquSolver(domain_equ=domain_equ, m=truth_fun, f=f, points=measurement_points)
d = np.load(DATA_DIR + "measurement_noise_2D" + "_" + str(noise_level) + ".npy")
d_clean = np.load(DATA_DIR + "measurement_clean_2D.npy")
noise_level_ = noise_level*max(np.absolute(d_clean))
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)
## setting the Model
model = ModelDarcyFlow(
    d=d_clean, domain_equ=domain_equ, prior=prior_measure,
    noise=noise, equ_solver=equ_solver
    )
###################################################################################
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
        val = M @ val
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
def Darcyflow_negtive_log_post(x):
    return post(x,model)
sample=[]




