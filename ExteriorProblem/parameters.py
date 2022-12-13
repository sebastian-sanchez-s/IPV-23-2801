import os
import numpy as np

'''
    Simulation related
'''
RAD     = 1
EPSILON = 1E-5
DMIN    = -3
DMAX    = 3

WAVE_NUMBERS    = range(1, 20 + 1)
ELEM_WAVELENGTH = np.arange(0.25, 5.0, 0.5)
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
PATH_A_D = os.path.join('data', 'analytical', 'dirichlet')
PATH_A_N = os.path.join('data', 'analytical', 'neumann')
PATH_B_D = os.path.join('data', 'bem', 'dirichlet')
PATH_B_N = os.path.join('data', 'bem', 'neumann')
PATHS    = (PATH_A_D, PATH_A_N, PATH_B_D, PATH_B_N)

FNAME_A  = lambda k: f'{k}.npy'
FNAME_B  = lambda k, n: f'{k}_{n}.npy'
