import numpy as np
import fenics as fe
import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianElliptic2
from core.misc import save_expre, generate_points
from common_PDE import EquSolver
DATA_DIR = './DATA/'
equ_nx = 500
domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)
k=1
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=(0.1/k), a_fun=(1.0/k), theta=1.0, boundary='Neumann'
    )
truth_fun_shape = fe.Expression('1*(exp(-20*(x[0]-0.3)*(x[0]-0.3)-20*(x[1]-0.3)*(x[1]-0.3))+exp(-20*(x[0]-0.7)*(x[0]-0.7)-20*(x[1]-0.7)*(x[1]-0.7)))', degree=1)
truth_fun = fe.interpolate(truth_fun_shape, domain_equ.function_space)
os.makedirs(DATA_DIR, exist_ok=True)
np.save(DATA_DIR + 'truth_vec', truth_fun.vector()[:])
file1 = fe.File(DATA_DIR + "truth_fun.xml")
file1 << truth_fun
file2 = fe.File(DATA_DIR + 'saved_mesh_truth.xml')
file2 << domain_equ.function_space.mesh()
truth_fun = fe.Function(domain_equ.function_space)
truth_fun.vector()[:] = np.load(DATA_DIR + 'truth_vec.npy')
num_x, num_y = 20, 20
x = np.linspace(1, 20, num_x)/(num_x+1)
y = np.linspace(1, 20, num_y)/(num_y+1)
coordinates = generate_points(x, y)
np.save(DATA_DIR + "coordinates_2D", coordinates)
f_expre = "1*sin(1*pi*x[0])*sin(1*pi*x[1])"
f = fe.Expression(f_expre, degree=1)
save_expre(DATA_DIR + 'f_2D.txt', f_expre)
equ_solver = EquSolver(domain_equ=domain_equ, m=truth_fun, f=f, points=coordinates)
sol = fe.Function(domain_equ.function_space)
sol.vector()[:] = equ_solver.forward_solver()

clean_data = [sol(point) for point in coordinates]
np.save(DATA_DIR + 'measurement_points_2D', coordinates)
np.save(DATA_DIR + 'measurement_clean_2D', clean_data)
np.save('clean_100', clean_data)
data_max = max(np.absolute(clean_data))
## add noise to the clean data
noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05]
for noise_level in noise_levels:
    data = clean_data + noise_level*data_max*np.random.normal(0, 1, (len(clean_data),))
    path = DATA_DIR + 'measurement_noise_2D' + '_' + str(noise_level)
    np.save(path, data)

    











