import numpy as np
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt
import fenics as fe
from core.noise import NoiseGaussianIID
import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianElliptic2
from common_PDE import EquSolver, ModelDarcyFlow
from core.misc import save_expre, generate_points
DATA_DIR = './DATA/'
equ_nx = 20
domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)
k=1
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=(0.1/k), a_fun=(1.0/k), theta=1.0, boundary='Neumann'
    )
num=10000
dim=(equ_nx+1)**2
m_data=np.zeros((num,dim))
for i in range(num):
    print(i)
    m_data[i,:]=prior_measure.generate_sample()
os.makedirs(DATA_DIR, exist_ok=True)
np.save(DATA_DIR + 'm_data', m_data)
print('m_data',m_data.shape)
####################################################################################
## specify the measurement points
num_x, num_y = 20, 20
x = np.linspace(1, 20, num_x)/(num_x+1)
y = np.linspace(1, 20, num_y)/(num_y+1)
coordinates = generate_points(x, y)
f_expre = "sin(1*pi*x[0])*sin(1*pi*x[1])"
save_expre(DATA_DIR + 'f_2D.txt', f_expre)
f = fe.interpolate(fe.Expression(f_expre, degree=2), domain_equ.function_space)
####################################################################################
#solve the PDE
truth_fun_vec_all=np.load(DATA_DIR + 'm_data.npy')
solution_data=np.zeros((num,dim))
for i in range(num):
    if (i%100==1):
        print(i)
    truth_fun = fe.Function(domain_equ.function_space)
    truth_fun_vec=truth_fun_vec_all[i,:]
    truth_fun.vector()[:] = truth_fun_vec
    equ_solver = EquSolver(domain_equ=domain_equ, m=truth_fun, f=f, points=coordinates)
    solution_data[i,:]=equ_solver.forward_solver()
np.save(DATA_DIR + 'solution_data', solution_data)
print('solution_data',solution_data.shape)
###################################################################################
solution_data=np.load(DATA_DIR + 'solution_data.npy')
points_number=num_x*num_y
clean_data_before=np.zeros((num,points_number))
clean_data=np.zeros((num,num_x,num_y))
for i in range(num):
    sol = fe.Function(domain_equ.function_space)
    sol.vector()[:]=solution_data[i,:]
    d=[sol(point) for point in coordinates]
    clean_data_before[i,:]=d
    d1 = np.reshape(d, (num_x, num_y))
    d1 = np.flip(d1, axis=1)
    clean_data[i,:,:] = d1.T
np.save(DATA_DIR + 'clean_data', clean_data)
np.save(DATA_DIR + 'clean_data_before_reshape', clean_data_before)
print('clean_data',clean_data.shape)
#######################################################################################
np.save(DATA_DIR + 'measurement_points_2D', coordinates)
clean_data=np.load(DATA_DIR + 'clean_data.npy')
clean_data_before=np.load(DATA_DIR+'clean_data_before_reshape.npy')
noise_level = 0.05
noise_data_1=np.zeros((num,num_x,num_y))
noise_data_before=np.zeros((num,points_number))
for i in range(num):
    clean_data_i=clean_data[i,:,:]
    clean_data_before_i=clean_data_before[i,:]
    data_max = np.max(np.absolute(clean_data_i))
    data = clean_data_i + noise_level * data_max * np.random.normal(0, 1, (len(clean_data_i),))
    data_before=clean_data_before_i+ noise_level * data_max * np.random.normal(0, 1, (len(clean_data_before_i),))
    noise_data_1[i,:,:]=data
    noise_data_before[i,:]=data_before
np.save(DATA_DIR + 'noise_data_0.05', noise_data_1)
np.save(DATA_DIR + 'noise_data_before_reshape_0.05', noise_data_before)
print('noise_data',noise_data_1.shape)
print('before',noise_data_before.shape)
#########################################################################
m=np.load(DATA_DIR + 'm_data.npy')
sol_data=np.load(DATA_DIR + 'solution_data.npy')
d_clean_data=np.load(DATA_DIR + 'clean_data_before_reshape.npy')
d_noise_data=np.load(DATA_DIR + 'noise_data_before_reshape_0.05.npy')
f = fe.interpolate(fe.Expression(f_expre, degree=2), domain_equ.function_space)
truth_fun=fe.Function(domain_equ.function_space)
truth_fun.vector()[:]=m[0,:]
k=1
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=(0.1/k), a_fun=(1.0/k), theta=1.0, boundary='Neumann'
    )
measurement_points = coordinates
equ_solver = EquSolver(domain_equ=domain_equ, m=truth_fun, f=f, points=measurement_points)
d_clean=d_clean_data[0,:]
noise_level=0.01
noise_level_ = noise_level*max(d_clean)
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)
model = ModelDarcyFlow(
    d=d_clean, domain_equ=domain_equ, prior=prior_measure,
    noise=noise, equ_solver=equ_solver
    )
S=model.S
M=prior_measure.M
Psi_r=np.load('./RESULT/'+'eig_vec_'+str(dim)+'.npy')
project_adj=np.zeros((num,20))
for i in range(num):
    print(i)
    d_noise=d_noise_data[i,:]
    adj=spsl.spsolve(M,S.T@d_noise)
    proadj=Psi_r.T@(M@adj)
    project_adj[i,:]=proadj
np.save(DATA_DIR + 'project_adj',project_adj)
project=np.load(DATA_DIR + 'project_adj.npy')/400
print(project.shape)
mean=np.mean(project,axis=0)
plt.plot(mean)
plt.show()
print(mean)
var=np.var(project,axis=0)
print(var)
project_standard=(project-mean)/np.sqrt(var)
plt.plot(project_standard.T)
plt.show()
np.save(DATA_DIR + 'pro_mean',mean)
np.save(DATA_DIR + 'pro_var',var)

    











