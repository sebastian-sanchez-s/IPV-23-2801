import numpy as np
from scipy.special import spherical_jn as jn, spherical_yn as yn, eval_legendre

h2n = lambda n,a,derivative=False: jn(n, a, derivative=derivative) - 1j * yn(n, a, derivative=derivative)

def coefficients_for_transmission(
    k_ext           : np.float32,
    k_int           : np.float32,
    rho_ext         : np.float32,
    rho_int         : np.float32,
    r               : np.float32=1,
    epsilon         : np.float32=1E-5,
    max_ite         : int=50,
) -> (np.ndarray, np.ndarray):
    psca_coef = np.zeros(max_ite, dtype=np.complex128)
    pint_coef = np.zeros(max_ite, dtype=np.complex128)

    rho = rho_int/rho_ext
    k = k_ext/k_int

    n = 0
    while n < max_ite:
        jn_int = jn(n, k_int * r)
        jn_ext = jn(n, k_ext * r)
        h2n_ext = h2n(n, k_ext * r)

        d_jn_int = jn(n, k_int * r, derivative=True)
        d_jn_ext = jn(n, k_ext * r, derivative=True)
        d_h2n_ext = h2n(n, k_ext * r, derivative=True)

        tau_num = (2*n + 1) * (1j)**n * (d_jn_int * jn_ext - rho * k * jn_int * d_jn_ext)
        tau_den = rho * k * jn_int * d_h2n_ext - d_jn_int * h2n_ext

        tau_n = tau_num/tau_den
        ups_n = ((2*n + 1) * (1j)**n * jn_ext + tau_n * h2n_ext) / jn_int

        psca_coef[n] = tau_n
        pint_coef[n] = ups_n

        n += 1

    return psca_coef, pint_coef
