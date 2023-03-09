import os
import numpy as np

'''
    Simulation related
'''
RAD     = 1
EPSILON = 1E-5
DMIN    = -3
DMAX    = 3

DENSITIES       = range(1,2) 
WAVE_NUMBERS    = [i for i in range(1, 16 + 1) if i % 2 == 0]
ELEM_WAVELENGTH = range(1, 5 + 1)
ELEM_SIZE       = lambda k, n: 2*np.pi / (n * k)

''' 
    Visual 
'''
VN      = 30
VH      = lambda k: VN * (DMAX - DMIN) * k // 6
GRID    = lambda k: np.mgrid[DMIN:DMAX:VH(k) * 1j, DMIN:DMAX:VH(k) * 1j]

'''
    Data and Files
'''
PATH_A = os.path.join('data', 'analytical')
PATH_B = os.path.join('data', 'bem')
PATHS    = (PATH_A, PATH_B)

FNAME_A  = lambda ke,ki,re,ri: f'{ke}_{ki}_{re}_{ri}.npy'
FNAME_B  = lambda ke,ki,re,ri,n: f'{ke}_{ki}_{re}_{ri}_{n}.npy'
