import numpy as np
import fenics as fe
import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianElliptic2
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from core.misc import save_expre, generate_points
from common_PDE import EquSolver
DATA_DIR = './DATA/'
equ_nx = 20
domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)
k=1
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=(0.1/k), a_fun=(1.0/k), theta=1.0, boundary='Neumann'
    )

for i in range(3):
    sample=prior_measure.generate_sample()
    fun=fe.Function(domain_equ.function_space)
    fun.vector()[:]=sample
    fig = fe.plot(fun)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar = plt.colorbar(fig)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.3)
    cbar.locator = plt.MaxNLocator(6)
    cbar.update_ticks()
    plt.show()