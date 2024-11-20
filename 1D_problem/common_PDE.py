import numpy as np
import scipy.sparse.linalg as spsl
import fenics as fe
import sys, os
sys.path.append(os.pardir)
from core.model import ModelBase
from core.misc import construct_measurement_matrix, trans2spnumpy
class EquSolver(object):
    def __init__(self, domain_equ, alpha, points, m=None):
        self.domain = domain_equ
        self.alpha = alpha
        self.points = points.copy()
        if m == None:
            m = fe.Constant('0.0')
        self.m_param = fe.interpolate(m, self.domain.function_space)
        self.m_vec_ = fe.interpolate(m, self.domain.function_space).vector()
        u_, v_ = fe.TrialFunction(self.domain.function_space), fe.TestFunction(self.domain.function_space)
        self.M_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        self.Finv_ = fe.assemble(fe.inner(u_, v_)*fe.dx + fe.Constant(self.alpha)*fe.inner(fe.grad(u_), fe.grad(v_))*fe.dx)
        self.FinvAdj_ = fe.assemble(fe.inner(v_, u_)*fe.dx + fe.Constant(self.alpha)*fe.inner(fe.grad(v_), fe.grad(u_))*fe.dx)
        self.S = construct_measurement_matrix(self.points, self.domain.function_space)
        def boundary(x, on_boundary):
            return on_boundary
        bc = fe.DirichletBC(self.domain.function_space, fe.Constant('0.0'), boundary)
        self.M = trans2spnumpy(self.M_)
        self.Finv = trans2spnumpy(self.Finv_) 
        self.FinvAdj = trans2spnumpy(self.FinvAdj_)
        #auxiliary vectors
        self.forward_sol_ = fe.Vector()
        self.Finv_.init_vector(self.forward_sol_, 1)
        self.adjoint_sol_ = fe.Vector()
        self.FinvAdj_.init_vector(self.adjoint_sol_, 1)
        self.rhs_adj_ = fe.Vector()
        self.FinvAdj_.init_vector(self.rhs_adj_, 0)
        self.temp_vec_ = fe.Vector()
        self.Finv_.init_vector(self.temp_vec_, 1)
    def update_m(self, m_vec):
        self.m_vec_[:] = m_vec[:]
        self.m_param.vector()[:] = self.m_vec_[:]
    
    def update_points(self, points):
        self.points = points
        self.S = construct_measurement_matrix(self.points, self.domain.function_space)

    def forward_solver(self, m_vec=None):
        if m_vec is not None:
            self.update_m(m_vec)
        return np.array(spsl.spsolve(self.Finv, self.M@self.m_vec_[:]))

    def incremental_forward_solver(self, m_hat=None):
        if m_hat is None:
            m_hat = np.array(self.m_vec_[:])
        val = spsl.spsolve(self.Finv, self.M@m_hat)
        return np.array(val)

    def adjoint_solver(self, res_vec):
        val = spsl.spsolve(self.FinvAdj.T, (self.S.T)@res_vec)
        val = spsl.spsolve(self.M, val)
        return np.array(val)

    def incremental_adjoint_solver(self, vec, m_hat=None):
        rhs_adj = self.S.T@vec
        val = spsl.spsolve(self.FinvAdj, rhs_adj)
        val = spsl.spsolve(self.M, val)
        return np.array(val)

    def construct_fun(self, f_vec):
        f = fe.Function(self.domain.function_space)
        f.vector().set_local(f_vec)
        return f
class ModelSS(ModelBase):
    def __init__(self, d, domain_equ, prior, noise, equ_solver):
        super().__init__(d, domain_equ, prior, noise, equ_solver)
    def update_m(self, m_vec, update_sol=True):
        self.m.vector()[:] = np.array(m_vec)
        self.equ_solver.update_m(self.m.vector())
        if update_sol is True:
            self.p.vector()[:] = self.equ_solver.forward_solver()
    def loss_residual(self):
        temp = (self.S@self.p.vector()[:] - self.noise.mean - self.d)
        if self.noise.precision is None:
            temp = temp@spsl.spsolve(self.noise.covariance, temp)
        else:
            temp = temp@self.noise.precision@temp
        return 0.5*temp
    def loss_residual_L2(self):
        temp = (self.S@self.p.vector()[:] - self.d)
        temp = temp@temp
        return 0.5*temp
    def eval_grad_residual(self, m_vec):
        self.equ_solver.update_m(m_vec)
        self.p.vector()[:] = self.equ_solver.forward_solver()
        res_vec = spsl.spsolve(self.noise.covariance, self.S@(self.p.vector()[:]) - self.d)
        self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        g = g_.vector()[:]
        return np.array(g)
    def eval_hessian_res_vec(self, dm):
        self.equ_solver.update_m(dm)
        self.p.vector()[:] = self.equ_solver.forward_solver()
        res_vec = spsl.spsolve(self.noise.covariance, self.S@(self.p.vector()[:]))
        self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        HM = g_.vector()[:]
        return np.array(HM)
