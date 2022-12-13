import numpy as np
import bempp.api
from bempp.api.linalg import gmres
from bempp.api.operators.potential import helmholtz as helmholtz_potential

# Solves neumann problem
def function_for_neumann(
    k   : np.float32,
    h   : np.float32,
    r   : np.float32=1,
    tol : np.float32=1E-5
) -> (bempp.api.space.space, np.ndarray, int):
    grid = bempp.api.shapes.sphere(r=r, h=h)
    space = bempp.api.function_space(grid, "P", 1)

    # lhs
    identity = bempp.api.operators.boundary.sparse.identity(space, space, space)
    dlp = bempp.api.operators.boundary.helmholtz.double_layer(space, space, space, k)
    hlp = bempp.api.operators.boundary.helmholtz.hypersingular(space, space, space, k)

    lhs = (0.5 * identity) - dlp + (1j/k) * hlp

    # rhs
    @bempp.api.complex_callable
    def combined_data(v, normal, domain_index, result):
        # u_i + i \partial_n u_i = e^{ikz} + i*i * k * e^{ikz} * n_z = e^{ikz} (1 - k * n_z)
        result[0] = np.exp(1j * k * v[2]) * (1 - normal[2])

    grid_fun = bempp.api.GridFunction(space, fun=combined_data)

    dirichlet_fun, info, it_count = gmres(lhs, grid_fun, tol=tol, return_iteration_count=True)

    return space, dirichlet_fun, it_count

# Solve dirichlet problem
def function_for_dirichlet(
    k   : np.float32,
    h   : np.float32,
    r   : np.float32=1,
    tol : np.float32=1E-5
) -> (bempp.api.space.space, np.ndarray, int):
    grid = bempp.api.shapes.sphere(r=r, h=h)
    space = bempp.api.function_space(grid, "P", 1)

    ''' Solve for psi := neumann_fun '''
    # lhs
    identity = bempp.api.operators.boundary.sparse.identity(space, space, space)
    adlp = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(space, space, space, k)
    slp = bempp.api.operators.boundary.helmholtz.single_layer(space, space, space, k)

    lhs = 0.5 * identity + adlp - 1j * k * slp

    # rhs
    @bempp.api.complex_callable
    def combined_data(v, normal, domain_index, result):
        # e^{ikz} \cdot n - i k e^{ikz} = ike^{ikz} (n_z - 1)
        result[0] = 1j * k * np.exp(1j * k * v[2]) * (normal[2] - 1)

    grid_fun = bempp.api.GridFunction(space, fun=combined_data)

    neumann_fun, info, it_count = gmres(lhs, grid_fun, tol=tol, return_iteration_count=True)

    return space, neumann_fun, it_count
