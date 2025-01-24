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
from pynverse import inversefunc


def calculate_rlimits_i(i, d_s, N_r_bin, last_cell_size):
    r_i = d_s * np.tan(i*np.pi / (2.0*(N_r_bin+last_cell_size)))
    return r_i
def calculate_rlimits_i_cvol(i, d_s, N_r_bin, R_sim):
    '''
    Calculates the lower limit of the i-th bin for the constant volume binning in the
    non-compact space (constant volume in the compact space)

    Parameters:
    -----------
    i : int
        The ID of the boundary
    d_s : float
        The diameter of the 4D sphere
    N_r_bin : int
        Number of the radial bins
    R_sim : float
        The radius of the simulation volume in real space
    '''
    omega_max = 2.0 * np.arctan(R_sim/d_s)
    V_unit_bin = (2.0*omega_max - np.sin(2.0*omega_max)) / N_r_bin
    V_unit_to_i = i * V_unit_bin
    #inverting numerically the x-sin(x) function
    func = lambda x: x-np.sin(x)
    omega_i = inversefunc(func, y_values=V_unit_to_i) / 2.0
    r_i = d_s * np.tan(omega_i/2)
    return r_i
def calculate_r_i(r_func, i, d_s, N_r_bin, last_cell_size):
    ll = r_func(i, d_s, N_r_bin, last_cell_size)    # lower limit
    ul = r_func(i+1, d_s, N_r_bin, last_cell_size)  # upper limit
    #simple assumption with "conical frustum"
    r_i = 0.25 * (ul-ll) * (ll*ll + 2*ll*ul + 3*ul*ul) / (ll*ll + ll*ul + ul*ul) + ll
    return r_i