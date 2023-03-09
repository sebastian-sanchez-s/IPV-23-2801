import numpy as np
from scipy.special import spherical_jn, spherical_yn, eval_legendre

spherical_hankel1 = lambda n, z, derivative=False: spherical_jn(n, z, derivative=derivative) + 1j * spherical_yn(n, z, derivative=derivative)

BD_COEF = { # return a tuple with (numerator, denominator)
    'dirichlet': 
           lambda n,r,k: 
           (
                -(2*n + 1) * 1j**n * spherical_jn(n, k*r),
                spherical_hankel1(n, k*r)
            ),
    'neumann': 
           lambda n,r,k:
           (
                -(2*n + 1) * 1j**n * spherical_jn(n, k*r, derivative=True),
                spherical_hankel1(n, k*r, derivative=True)
            ),
}

def coefficients_for(
    boundary_data   : str,
    k               : np.float32,
    r               : np.float32=1,
    n_iter          : np.uint32=50,
    epsilon         : np.float32=1E-10,
) -> (np.ndarray):
    if boundary_data not in BD_COEF.keys():
        print(f'Boundary data \"{boundary_data}\" is invalid.')
        print('Valid options are:')
        for i, bd_name in enumerate(BD_COEF.keys()):
            print(f'{i:2}. {bd_name}')
        raise error

    coeff = np.zeros(n_iter, dtype=np.complex128) 
    for n in range(n_iter):
        an_num, an_den = BD_COEF[boundary_data](n, r, k)

        an = an_num / an_den

        coeff[n] = an

    return coeff
