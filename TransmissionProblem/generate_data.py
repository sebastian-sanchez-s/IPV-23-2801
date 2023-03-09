import os
import numpy as np
import parameters as params
import bempp.api
from analytical import coefficients_for_transmission, h2n, jn, yn, eval_legendre
from bem import total_exterior_cauchy_trace 

def generate_data():
    '''
        Create data directories
    '''
    for d in params.PATHS:
        os.makedirs(d, exist_ok=True)

    '''
        Generate data
    '''
    for ke in params.WAVE_NUMBERS:
        for ki in params.WAVE_NUMBERS:
            '''
                Visual parameters
            '''
            k = max(ke, ki)
            x, z = params.GRID(k)

            rs = np.sqrt(x**2 + z**2)
            ps = z/rs

            ext = rs > params.RAD

            for re in params.DENSITIES:
                for ri in params.DENSITIES:
                    print(ke, ki, re, ri)

                    ''' 
                        Analytical Solutions 
                    '''
                    fpath = os.path.join(params.PATH_A, params.FNAME_A(ke,ki,re,ri))

                    print('\tAnalytical')
                    if not os.path.exists(fpath):
                        coef = coefficients_for_transmission(ke, ki, re, ri, 
                                                             r=params.RAD,
                                                             epsilon=params.EPSILON)

                        pinc = np.zeros_like(rs, dtype=np.complex128)
                        pinc[ext] = np.exp(-1j * ke * z[ext])

                        psca = np.zeros_like(x, dtype=np.complex128)
                        pint = np.zeros_like(x, dtype=np.complex128)

                        for n, (an, bn) in enumerate(zip(*coef)):
                            psca[ext] += an * h2n(n, ke * rs[ext]) * eval_legendre(n, ps[ext])
                            
                            pint[~ext] += bn* jn(n, ki * rs[~ext]) * eval_legendre(n, ps[~ext])

                        ptot = np.real(psca + pinc + pint)
                        np.save(fpath, (ptot, len(coef[0])))
                    
                    '''
                        BEM solution
                    '''
                    for n in params.ELEM_WAVELENGTH:
                        print('\tBEM', n)
                        fpath = os.path.join(params.PATH_B, params.FNAME_B(ke,ki,re,ri,n))

                        if os.path.exists(fpath):
                            continue
                        
                        tdf, tnf, icnt = total_exterior_cauchy_trace(params.ELEM_SIZE(k, n), 
                                                                   ke, ki, re, ri,
                                                                     epsilon=params.EPSILON)

                        pnts = np.vstack((x.ravel(), np.zeros(z.size), z.ravel()))

                        idx_ext = np.sqrt(pnts[0]**2 + pnts[2]**2) > params.RAD 
                        idx_int = np.sqrt(pnts[0]**2 + pnts[2]**2) <= params.RAD

                        pnts_ext = pnts[:, idx_ext]
                        pnts_interior = pnts[:, idx_int]

                        dspace = tdf.space
                        nspace = tnf.space

                        slp_pot_int = bempp.api.operators.potential.helmholtz.single_layer(dspace, pnts_interior, ki)
                        slp_pot_ext = bempp.api.operators.potential.helmholtz.single_layer(dspace, pnts_ext, ke)
                        dlp_pot_int = bempp.api.operators.potential.helmholtz.double_layer(dspace, pnts_interior, ki)
                        dlp_pot_ext = bempp.api.operators.potential.helmholtz.double_layer(dspace, pnts_ext, ke)

                        uint = (slp_pot_int * ((re/ri) * tnf) - dlp_pot_int * tdf).ravel()
                        uext = (dlp_pot_ext * tdf - slp_pot_ext * tnf).ravel() + np.exp(-1j * ke * pnts_ext[2])

                        utot = np.zeros(pnts.shape[1], dtype='complex128')
                        utot[idx_ext] = uext
                        utot[idx_int] = uint

                        utot = np.real(utot.reshape(x.shape))
                        np.save(fpath, (utot, icnt))

if __name__ == '__main__':
    generate_data()
