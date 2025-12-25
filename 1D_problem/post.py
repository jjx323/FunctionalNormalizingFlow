import fenics as fe
import sys, os
import numpy as np
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
from core.probability import GaussianElliptic2
import torch
from common_PDE import EquSolver, ModelSS
from core.noise import NoiseGaussianIID
equ_nx = np.load('equ_nx.npy')
domain_equ = Domain1D(n=equ_nx, mesh_type='CG', mesh_order=1)
noise_level = 0.05
DATA_DIR = './DATA/'
k=1
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=(0.1/k), a_fun=(1.0/k), theta=1.0, boundary='Neumann'
    )
M=prior_measure.M
K=prior_measure.K
measurement_points = np.load(DATA_DIR + "measurement_points_1D.npy")
mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'CG', 1)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')
truth_fun = fe.interpolate(truth_fun, domain_equ.function_space)
alpha = 0.01
equ_solver = EquSolver(domain_equ=domain_equ, alpha=alpha, \
                           points=np.array([measurement_points]).T, m=truth_fun)
d = np.load(DATA_DIR + "measurement_noise_1D" + "_" + str(noise_level) + ".npy")
d_clean = np.load(DATA_DIR + "measurement_clean_1D.npy")
noise_level_ = noise_level*max(np.absolute(d_clean))
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)
## setting the Model
model = ModelSS(
        d=d_clean, domain_equ=domain_equ, prior=prior_measure,
        noise=noise, equ_solver=equ_solver
    )
##############################################################################################
def prior_sample_torch(num,prior):
    sample = []
    for i in range(num):
        x = prior.generate_sample()
        sample.append(x)
        Sample = np.array(sample)
    Sample_torch = torch.from_numpy(Sample)
    return Sample_torch
S=model.S
F_inv=equ_solver.Finv
#######################################post_GPU
M_dense=M.todense()
K_dense=K.todense()
S_dense=S.todense()
F_inv_dense=F_inv.todense()
M_torch_dense=torch.from_numpy(M_dense)
K_torch_dense=torch.from_numpy(K_dense)
F_inv_torch_dense=torch.from_numpy(F_inv_dense)
M_torch_dense_inv=torch.inverse(M_torch_dense)
K_torch_dense_inv=torch.inverse(K_torch_dense)
F_torch_dense=torch.inverse(F_inv_torch_dense)
S_torch_dense=torch.from_numpy(S_dense)
d_torch=torch.from_numpy(d_clean)
noise_level_ = noise_level*max(np.absolute(d_clean))
def post(x):
    m_vec = x
    val=K_torch_dense.T@M_torch_dense_inv@K_torch_dense@m_vec.T
    val = torch.atleast_2d(val)
    if (val.shape[0] == 1):
        val = val.T
    batchsize = val.shape[1]
    output1 = torch.zeros((batchsize))
    for i in range(batchsize):
        val_i = val[:, i]
        m_vec_i = m_vec[i, :]
        output1[i] = val_i @ m_vec_i
    output1 = output1 / 2


    output2 = torch.zeros((batchsize))
    for i in range(batchsize):
        m_vec_i = m_vec[i, :]
        sol=F_torch_dense@(M_torch_dense@m_vec_i)
        sol_ob=S_torch_dense@sol
        delta=sol_ob-d_torch
        output2[i]=(delta@delta)/(2*noise_level_**2)
    output = output1 + output2
    return output

def post_multi(x):
    m_vec = x
    val=K_torch_dense.T@M_torch_dense_inv@K_torch_dense@m_vec.T
    val = torch.atleast_2d(val)
    if (val.shape[0] == 1):
        val = val.T
    batchsize = val.shape[1]
    output1 = torch.zeros((batchsize))
    for i in range(batchsize):
        val_i = val[:, i]
        m_vec_i = m_vec[i, :]
        output1[i] = val_i @ m_vec_i
    output1 = output1 / 2


    output2 = torch.zeros((batchsize))
    for i in range(batchsize):
        m_vec_i = m_vec[i, :]
        noise_level_=0.02
        ss=0.5
        f1=ss*torch.cos(2*torch.pi*torch.linspace(0,1,equ_nx+1))
        f2=-ss*torch.cos(2*torch.pi*torch.linspace(0,1,equ_nx+1))
        #print(m_vec_i.shape)
        output2_1=-(f1-m_vec_i)@(M_torch_dense@(f1-m_vec_i))/(2*noise_level_**2)
        output2_2 = -(f2 - m_vec_i) @ (M_torch_dense @ (f2 - m_vec_i))/(2*noise_level_**2)

        output2[i]=-torch.logsumexp(torch.stack([output2_1, output2_2]),dim=0)
    output = output1 + output2
    return output

