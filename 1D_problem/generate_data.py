import numpy as np
import matplotlib.pyplot as plt
import fenics as fe
import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
from core.probability import GaussianElliptic2
from common_PDE import EquSolver
DATA_DIR = './DATA/'
## domain for solving PDE
equ_nx_inverse=100
np.save('equ_nx',equ_nx_inverse)

equ_nx = 10000
dim=equ_nx+1
k=1
domain_equ = Domain1D(n=equ_nx, mesh_type='CG', mesh_order=1)
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=(0.1/k), a_fun=(1.0/k), theta=1.0, boundary='Neumann'
    )
######################################################################
truth_fun = fe.Expression ( 'exp(-50*(x[0]-0.3)*(x[0]-0.3))-exp(-50*(x[0]-0.7)*(x[0]-0.7))', degree=1)
truth_fun = fe.interpolate(truth_fun, domain_equ.function_space)
# truth_fun.vector()[:]=prior_measure.generate_sample()
## save the truth
os.makedirs(DATA_DIR, exist_ok=True)
np.save(DATA_DIR + 'truth_vec', truth_fun.vector()[:])
file1 = fe.File(DATA_DIR + "truth_fun.xml")
file1 << truth_fun
file2 = fe.File(DATA_DIR + 'saved_mesh_truth.xml')
file2 << domain_equ.function_space.mesh()
## load the ground truth
truth_fun = fe.Function(domain_equ.function_space)
truth_fun.vector()[:] = np.load(DATA_DIR + 'truth_vec.npy')
## specify the measurement points
coordinates = np.linspace(0, 1, 11)
alpha = 0.01
equ_solver = EquSolver(domain_equ=domain_equ, alpha=alpha, \
                        points=np.array([coordinates]).T, m=truth_fun)
sol = fe.Function(domain_equ.function_space)
sol.vector().set_local(equ_solver.forward_solver())
clean_data = [sol(point) for point in coordinates]
np.save(DATA_DIR + 'measurement_points_1D', coordinates)
np.save(DATA_DIR + 'measurement_clean_1D', clean_data)
data_max = max(np.absolute(clean_data))
## add noise to the clean data
noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05]
for noise_level in noise_levels:
    data = clean_data + noise_level*data_max*np.random.normal(0, 1, (len(clean_data),))
    path = DATA_DIR + 'measurement_noise_1D' + '_' + str(noise_level)
    np.save(path, data)












