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

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import root_scalar


class SphericalBinner(ABC):
    '''TODO
    '''
    @abstractmethod
    def r_limit(self, i: int):
        '''TODO'''
        raise NotImplementedError
    @abstractmethod
    def r_centroid(self, i: int):
        '''TODO'''
        raise NotImplementedError
    @staticmethod
    def centroid(r0: float, r1: float):
        r'''
        Calculates the centroid of a conical frustum along the radius in
        3D space for the i-th bin between r0 and r1.
        The formula used is:
        .. math::
            r_i = \frac{1}{4} \frac{R_1^3 - R_0^3}{R_1^2 - R_0^2} + R_0
        where :math:`R_0` and :math:`R_1` are the lower and upper limits
        of the bin.
        
        Parameters:
        -----------
        r0 : float
            The lower limit of the bin.
        r1 : float
            The upper limit of the bin.
        
        Returns:
        --------
        r_i : float
            The centroid of the i-th bin.
        '''
        r_i = 0.25 * (r1-r0) * (r0*r0 + 2*r0*r1 + 3*r1*r1) / (r0*r0 + r0*r1 + r1*r1) + r0
        return r_i
    
    def invert_x_minus_sin_x(y: float, *, bracket=(0.0, np.pi), tol=1e-12):
        r'''
        Solve :math:`x - sin(x) = y for x` in :math:`[0, \pi]`, using
        Brent's method.

        Parameters
        ----------
        y : float
            Target value; must satisfy :math:`0 \leq y \leq \pi`.
        bracket : tuple of float, optional
            Bracketing interval for the root-finder. Defaults to
            :math:`(0, \pi)`.
        tol : float, optional
            Absolute tolerance for the solver.

        Returns
        -------
        x : float
            The unique solution in :math:`[0, \pi]` such that
            :math:`x - sin(x) = y`.
        '''
        if not (0.0 <= y <= np.pi):
            raise ValueError(f"y must be in [0, Pi], got y={y}")

        f = lambda x: x - np.sin(x) - y
        sol = root_scalar(f, method='brentq', bracket=bracket, xtol=tol)
        if not sol.converged:
            raise RuntimeError(f"Root finding did not converge for y={y}")

        return sol.root

class SphericalLinear(SphericalBinner):
    '''Equal step in angular space binning in the non-compact space'''
    def __init__(self, d_s, n_bins, last_cell_size):
        self.d_s = d_s
        self.n_bins = n_bins
        self.last_cell_size = last_cell_size
        self.d_omega = np.pi / (2 * (self.n_bins + self.last_cell_size))

    def r_limit(self, i):
        omega = i * self.d_omega
        return self.d_s * np.tan(omega)

    def r_centroid(self, i):
        r0 = self.r_limit(i)
        r1 = self.r_limit(i+1)
        return self.centroid(r0, r1)

class SphericalConstantVolume(SphericalBinner):
    '''
    Constant volume binning in the non-compact space (constant volume
    in the compact space)
    '''
    def __init__(self, d_s, n_bins, R_sim):
        self.d_s = d_s
        self.n_bins = n_bins
        self.omega_max = 2 * np.arctan(R_sim/d_s)
        self.unit_bin = (2*self.omega_max - np.sin(2*self.omega_max)) / n_bins

    def r_limit(self, i):
        omega = self.invert_x_minus_sin_x(i * self.unit_bin) / 2  # or 1/4??
        return self.d_s * np.tan(omega)

    def r_centroid(self, i):
        r0 = self.r_limit(i)
        r1 = self.r_limit(i+1)
        return self.centroid(r0, r1)


class CylindricalBinner(ABC):
    '''TODO
    '''
    @abstractmethod
    def r_limit(self, i: int):
        '''TODO'''
        raise NotImplementedError
    @abstractmethod
    def r_centroid(self, i: int):
        '''TODO'''
        raise NotImplementedError
    @staticmethod
    def centroid(r0: float, r1: float):
        '''
        Centroid of a cylindrical shell between r0 and r1 (assuming
        uniform height).
        '''
        return 2.0/3.0 * (r1*r1*r1 - r0*r0*r0) / (r1*r1 - r0*r0)

class CylindricalLinear(CylindricalBinner):
    '''TODO'''
    def __init__(self, R_max: float, n_bins: int):
        self.R_max = R_max
        self.n_bins = n_bins
        self.dr = R_max / n_bins

    def r_limit(self, i: int):
        return i * self.dr

    def r_centroid(self, i: int):
        r0 = self.r_limit(i)
        r1 = self.r_limit(i+1)
        # centroid of a cylindrical shell
        return self.centroid(r0, r1)

class CylindricalConstantVolume(CylindricalBinner):
    '''TODO'''
    def __init__(self, R_max: float, n_bins: int):
        self.R_max = R_max
        self.n_bins = n_bins
        self.dr = (R_max**2) / n_bins

    def r_limit(self, i: int):
        # invert A(r) = pi * r^2  ->  r_i = R_max * sqrt(i/n_bins)
        return np.sqrt(i * self.dr)

    def r_centroid(self, i: int):
        r0 = self.r_limit(i)
        r1 = self.r_limit(i+1)
        # centroid of a planar annulus
        return self.centroid(r0, r1)


def create_mass_nsample_lut(n_grid_samples, mass_list, M_box):
    '''
    Creates a lookup table for the number of samples per mass bin
    for a variable resolution grid in a regular StePS simulation.

    Parameters:
    -----------
    n_grid_samples : int
        Number of grids with different resolutions.
    mass_list : ndarray
        Array containing the unique particle masses.
    M_box : float
        Total mass in the simulation box (in Msol).

    Returns:
    --------
    mass_nsample_lut : ndarray of shape (n_grid_samples, 2)
        Lookup table for the number of samples per mass bin. The first column
        contains the unique mass values, and the second column contains the
        corresponding number of samples for each mass bin.
    '''
    
    mass_nsample = np.c_[
        np.double(mass_list),
        np.uint32(np.cbrt(M_box / mass_list))
    ]
    mass_nsample_lut = np.zeros((n_grid_samples, 2), dtype=np.uint32)
    d_nsample = np.uint32(np.ceil(len(mass_list) / n_grid_samples))
    
    print("The generated Nsample list:")
    print("ID\tNsample\tMass(in 10e11Msol)")
    # Populate lookup table by starting with the outermost mass bin,
    # while skipping the innermost layer
    mass_nsample_sorted = sorted(mass_nsample, key=lambda x: x[0])
    for si in reversed(range(n_grid_samples-1)):
        if si == n_grid_samples-2:
            idx = 0
        else:
            idx = len(mass_nsample_sorted) - 1 - si*d_nsample
        mass_nsample_lut[si] = mass_nsample_sorted[idx]
        print(f"{si}\t{mass_nsample_lut[si, 0]}\t{mass_nsample_lut[si, 1]/1e11:.6f}")
    return mass_nsample_lut