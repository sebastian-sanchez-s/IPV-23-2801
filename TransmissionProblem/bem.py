import numpy as np
import bempp.api
from scipy.sparse.linalg import gmres

it_count = 0

def total_exterior_cauchy_trace(
    h                   : np.float32,
    k_ext               : np.float32,
    k_int               : np.float32,
    rho_ext             : np.float32,
    rho_int             : np.float32,
    r                   : np.float32=1,
    epsilon             : np.float32=1E-5
) -> (np.ndarray, np.ndarray, int):
    grid = bempp.api.shapes.sphere(r=r, h=h)

    #
    # LHS
    #
    Ai = bempp.api.operators.boundary.helmholtz.multitrace_operator(grid, k_int)
    Ae = bempp.api.operators.boundary.helmholtz.multitrace_operator(grid, k_ext)

    Ai[0, 1] *= rho_int / rho_ext
    Ai[1, 0] *= rho_ext / rho_int
    
    lhs = (Ai + Ae).strong_form()

    #
    # RHS
    #
    @bempp.api.complex_callable
    def trace_dirichlet_pinc(x, n, domain_index, result):
        result[0] = np.exp(1j * k_ext * x[2])
        
    @bempp.api.complex_callable
    def trace_neumann_pinc(x, n, domain_index, result):
        # d_n e^{ikz} = (0, 0, ik e^{ikz}.(0,0,n3) = ik n3 e^{ikz}
        result[0] = 1j * k_ext * n[2] * np.exp(1j * k_ext * x[2])

    dspace = Ai[0, 0].domain
    nspace = Ai[0, 1].domain

    rhs = np.concatenate([
        bempp.api.GridFunction(dspace, fun=trace_dirichlet_pinc).coefficients,
        bempp.api.GridFunction(nspace, fun=trace_neumann_pinc).coefficients
    ])

    #
    # Solve
    #
    global it_count
    it_count = 0
    def iteration_counter(x):
        global it_count
        it_count += 1

    cauchy_trace, info = gmres(lhs, rhs, tol=epsilon, callback=iteration_counter)

    trace_dirichlet = bempp.api.GridFunction(
            dspace,
            coefficients=cauchy_trace[:dspace.global_dof_count]
    )
    trace_neumann = bempp.api.GridFunction(
            nspace,
            coefficients=cauchy_trace[dspace.global_dof_count:]
    )

    return trace_dirichlet, trace_neumann, it_count
