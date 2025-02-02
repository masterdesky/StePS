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


def interpolate(x, Lbox, disp_field):
    '''
    Perform trilinear interpolation of the displacement field onto
    particle positions.
    
    Parameters:
    -----------
    x : ndarray of shape (N, 3)
        Particle positions.
    Lbox : float
        Box size in Mpc/h.
    disp_field : ndarray of shape (N, N, N)
        Displacement field.
    
    Returns:
    --------
    disp_field_interp : ndarray of shape (N, 3)
        Interpolated displacement field at particle positions.
    '''
    nmesh = disp_field.shape[0]

    u, v, w = x.T / Lbox * nmesh

    i = np.floor(u).astype(int)
    j = np.floor(v).astype(int)
    k = np.floor(w).astype(int)

    u -= i
    v -= j
    w -= k

    i = i % nmesh
    j = j % nmesh
    k = k % nmesh

    ii = (i + 1) % nmesh
    jj = (j + 1) % nmesh
    kk = (k + 1) % nmesh

    # Calculate the trilinear interpolation coefficients for each
    # of the 8 vertices of the cube the particle is in
    f1 = (1 - u) * (1 - v) * (1 - w)
    f2 = (1 - u) * (1 - v) * w
    f3 = (1 - u) * v * (1 - w)
    f4 = (1 - u) * v * w
    f5 = u * (1 - v) * (1 - w)
    f6 = u * (1 - v) * w
    f7 = u * v * (1 - w)
    f8 = u * v * w

    disp_field_interp = (
        disp_field[i, j, k] * f1[:, None] +
        disp_field[i, j, kk] * f2[:, None] +
        disp_field[i, jj, k] * f3[:, None] +
        disp_field[i, jj, kk] * f4[:, None] +
        disp_field[ii, j, k] * f5[:, None] +
        disp_field[ii, j, kk] * f6[:, None] +
        disp_field[ii, jj, k] * f7[:, None] +
        disp_field[ii, jj, kk] * f8[:, None]
    )
    return disp_field_interp

def zeldovich(x, Lbox, overdensity_field, growth_rate, h):
    '''
    Performs the Zel'dovich approximation to compute particle positions
    and velocities.
    
    Parameters:
    -----------
    x : ndarray of shape (N, 3)
        Initial unperturbed particle positions.
    Lbox : float
        Box size in Mpc/h.
    overdensity_field : ndarray of shape (M, M, M)
        Target overdensity field (real field).
    growth_rate : float
        Time derivative of the growth factor D(t) in km/s/Mpc units.
    h : float
        Hubble constant in km/s/Mpc units.
    
    Returns:
    --------
    xpert : ndarray of shape (N, 3)
        Updated particle positions with periodic wrapping.
    v : ndarray of shape (N, 3)
        Updated particle velocities in km/s units.
    '''
    # Nagyskálás módusok mindig ugyanazok legyenek ugyanarra a seedre
    # Hermitikus kényszerek
    # Feltöltése a módusoknak nagyobbtól kisebbek irányába
    # Fourier térben kell komponenseket generálni
    nmesh = overdensity_field.shape[0]
    kk = np.fft.fftfreq(nmesh) * 2*np.pi/Lbox * nmesh
    ks = np.fft.rfftfreq(nmesh) * 2*np.pi/Lbox * nmesh
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
              f'in units of mean particle separation: {max_disp * nmesh/Lbox}')
        disp_field_interp = interpolate(x, Lbox, disp_field)
        xpert[:, i] = x[:, i] + disp_field_interp
        v[:, i] = disp_field_interp * growth_rate
    return xpert%Lbox, v/h

def second_order_lpt():
    '''
    Perform second-order Lagrangian perturbation theory.
    '''
    pass