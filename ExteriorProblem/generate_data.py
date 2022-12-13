import os
import numpy as np
from scipy.special import spherical_jn as sjn, spherical_yn as syn, eval_legendre
from bempp.api.operators.potential import helmholtz as helmholtz_potential
from analytical import coefficients_for
from bem import function_for_neumann, function_for_dirichlet 

import parameters as params

hkl2 = lambda n, z, derivative=False: sjn(n, z, derivative=derivative) - 1j * syn(n, z, derivative=derivative)

if __name__ == '__main__':
    '''
        Create data directories
    '''
    for d in params.PATHS:
        os.makedirs(d, exist_ok=True)

    '''
        Generate data

        For each wave number computes solutions with different mesh size
    '''
    for k in params.WAVE_NUMBERS:
        print('Wavenumber:', k)
        '''
            Visual Elements
        '''
        X, Z = params.GRID(k)

        rs = np.sqrt(X**2 + Z**2)
        ps = Z/rs

        ext = rs > params.RAD 
        pinc = np.zeros_like(rs, dtype=np.complex128)
        pinc[ext] = np.exp(1j * k * Z[ext])

        ''' 
            Analytical Solutions 
        '''
        #
        # Dirichlet 
        #
        fpath = os.path.join(params.PATH_A_D, params.FNAME_A(k))

        print('\tAnalytical Dirichlet', end='')
        done_s = ''
        if not os.path.exists(fpath):
            done_s = '...'

            coef = coefficients_for('dirichlet', k, epsilon=params.EPSILON)
            psca = np.zeros_like(pinc, dtype=np.complex128)
            for (n, an) in enumerate(coef):
                psca[ext] += an * hkl2(n, k * rs[ext]) * eval_legendre(n, ps[ext])

            data = (np.real(psca + pinc), coef.size)
            np.save(fpath, data)

        print(done_s) 
        #
        # Neumann 
        #
        fpath = os.path.join(params.PATH_A_N, params.FNAME_A(k))

        print('\tAnalytical Neumann', end='')
        done_s = ''
        if not os.path.exists(fpath):
            done_s = '...'

            coef = coefficients_for('neumann', k, epsilon=params.EPSILON)
            psca = np.zeros_like(pinc, dtype=np.complex128)
            for (n, bn) in enumerate(coef):
                psca[ext] += bn * hkl2(n, k * rs[ext]) * eval_legendre(n, ps[ext])

            data = (np.real(psca + pinc), coef.size)
            np.save(fpath, data)
        print(done_s)

        '''
            BEM Solutions
        '''
        points = np.vstack((X.ravel(),
                        np.zeros(X.size),
                        Z.ravel()))

        idx = np.sqrt(points[0]**2 + points[2]**2) > params.RAD 

        for n in params.ELEM_WAVELENGTH:
            #
            # Dirichlet 
            #
            fpath = os.path.join(params.PATH_B_D, params.FNAME_B(k, n))

            print('\tBEM Dirichlet', n, end='')
            done_s = ''
            if not os.path.exists(fpath):
                done_s = '...'

                space_d, neumann_fun, it_d = function_for_dirichlet(k, params.ELEM_SIZE(k, n), tol=params.EPSILON)
                slp_pot = helmholtz_potential.single_layer(space_d, points[:, idx], k)

                utot_d = np.zeros(points.shape[1], dtype=np.complex128)

                res = np.exp(1j * k * points[2,idx]) - slp_pot.evaluate(neumann_fun)
                utot_d[idx] = res.flat

                utot_d = np.real(utot_d.reshape(X.shape))

                data = (np.real(utot_d), it_d)
                np.save(fpath, data)

            print(done_s)

            #
            # Neumann 
            #
            fpath = os.path.join(params.PATH_B_N, params.FNAME_B(k, n))

            print('\tBEM Neumann', n, end='')
            done_s = ''
            if not os.path.exists(fpath):
                done_s = '...'

                space_n, dirichlet_fun, it_n = function_for_neumann(k, params.ELEM_SIZE(k, n), tol=params.EPSILON)
                dl_pot = helmholtz_potential.double_layer(space_n, points[:,idx], k)

                utot_n = np.zeros(points.shape[1], dtype=np.complex128)

                res = np.exp(1j * k * points[2,idx]) + dl_pot.evaluate(dirichlet_fun)
                utot_n[idx] = res.flat
                utot_n = np.real(utot_n.reshape(X.shape))

                data = (np.real(utot_n), it_n)
                np.save(fpath, data)
            print(done_s)
