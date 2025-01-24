#*******************************************************************************#
#  StePS_IC.py - An initial condition generator for                             #
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2017-2025 Gabor Racz                                         #
#                                                                               #
#    This program is free software; you can redistribute it and/or modify       #
#    it under the terms of the GNU General Public License as published by       #
#    the Free Software Foundation; either version 2 of the License, or          #
#    (at your option) any later version.                                        #
#                                                                               #
#    This program is distributed in the hope that it will be useful,            #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#    GNU General Public License for more details.                               #
#*******************************************************************************#

import numpy as np

def interpolate():
    pass

# Functions for cosmological perturbation theory
def zeldovich(x, Lbox, overdensity_field, growth_rate, h):
    '''
    Perform the Zel'dovich approximation to compute particle positions
    and velocities.
    
    Parameters:
    -----------
    x : ndarray of shape (N, 3)
        Initial unperturbed particle positions.
    Lbox : float
        Box size in Mpc/h.
    overdensity_field : ndarray of shape (N, N, N)
        Target overdensity field (real field).
    growth_rate : float
        Time derivative of the growth factor D(t) in km/s/Mpc units.
    
    Returns:
    --------
    positions : ndarray of shape (N, 3)
        Updated particle positions.
    velocities : ndarray of shape (N, 3)
        Updated particle velocities.
    '''
    # TODO: Köbös rácson elmozdulásmező
    # 3D rácspontok (ezek) között kiinterpolálom ezt a mezőt
    # Interpoláció választása CIC
    nres = overdensity_field.shape[0]
    kk = np.fft.fftfreq(nres) * 2*np.pi/Lbox * nres
    ks = np.fft.rfftfreq(nres) * 2*np.pi/Lbox * nres
    kvec = np.array(np.meshgrid(kk, kk, ks))
    kmod = np.linalg.norm(kvec, axis=0)
    delta_k = np.fft.rfftn(overdensity_field)
    xpert = np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x, dtype=np.float32)
    for i, xi in enumerate(('x', 'y', 'z')):
        psi_i = np.zeros_like(kmod, dtype=complex)
        mask = kmod > 0.0
        psi_i[mask] = -1j * kvec[i, mask] / kmod[mask]**2 * delta_k[mask]
        disp_field = np.fft.irfftn(psi_i)
        max_disp = np.max(disp_field)
        print(f'Maximal \'{xi}\' displacement: {max_disp*1000} kpc/h; '\
              f'in units of mean particle separation: {max_disp * nres/Lbox}')
        xpert[i, ...] = x[i, ...] + disp_field
        v[i, ...] = disp_field*growth_rate
    # Periodic wrapping
    xpert = np.fmod(xpert+Lbox, Lbox)
    # Converting the velocities from km/s/h to km/s
    v /= h 
    return xpert, v

def second_order_lpt():
    pass