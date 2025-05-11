#*******************************************************************************#
#  StePS_IC.py - An initial condition generator for                             #
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2017-2025 Gabor Racz, Balazs Pal                             #
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

# StePS internal units
UNIT_T = 47.14829951063323      # Unit time in Gy
UNIT_V = 20.738652969925447     # Unit velocity in km/s
UNIT_D = 3.0856775814671917e24  # =1Mpc Unit distance in cm

class CosmoData:
    '''
    Implements a class for handling cosmological snapshot data stored in
    a conventional Gadget-like structure for StePS simulations.

    Parameters:
    -----------
    data : ndarray of shape (N, 7)
        Array containing the particle data, where N is the number of particles.
    idx : int, optional; default=0
        Index of the particle ID column in the data array.
    cidx : tuple, optional; default=(1, 2, 3)
        Indices of the x, y, z columns in the data array.
    vidx : tuple, optional; default=(4, 5, 6)
        Indices of the vx, vy, vz columns in the data array.
    midx : int, optional; default=6
        Index of the mass column in the data array.
    silent : bool, optional; default=False
        If True, suppresses output messages.
    '''
    def __init__(self, data, idx=0, cidx=(1, 2, 3), vidx=(4, 5, 6), midx=6,
                 silent=False):
        self.id = data[:, idx]     # Particle IDs
        self.pos = data[:, cidx]   # Particle positions
        self.vel = data[:, vidx]   # Particle velocities
        self.mass = data[:, midx]  # Particle masses
        self.silent = silent
    
        # Calculated values
        self.N_part = data.shape[0]  # Number of particles
        self.mass_list = None        # List of unique particle masses
        self.M_box = None            # Total mass in the box (in Msol)

    def deep_copy(self):
        '''Deep copy of the object'''
        new = self.__class__.__new__(self.__class__)
        new.id        = self.id.copy()
        new.pos       = self.pos.copy()
        new.vel       = self.vel.copy()
        new.mass      = self.mass.copy()
        new.silent    = self.silent

        new.N_part    = self.N_part
        new.mass_list = (
            self.mass_list.copy() if self.mass_list is not None else None
        )
        new.M_box     = self.M_box
        return new

    def __deepcopy__(self, memo):
        '''Standard hook for copy.deepcopy(object)'''
        if id(self) in memo:
            return memo[id(self)]
        dup = self.deep_copy()
        memo[id(self)] = dup
        return dup

    def rescale_snapshot_mass(self, params):
        '''
        Rescale the particle masses to fit the cosmological parameters.
        
        Parameters:
        -----------
        params : dict
            Dictionary containing the cosmological parameters.
        midx : int, optional; default=6
            Index of the mass column in the data array.
        '''
        print("Rescaling the particle masses to fit the cosmological parameters...")
        M_tot = np.sum(self.mass)
        V_sim = 4.0*np.pi/3.0 * params['RSIM']**3
        rho_crit = 3*params['H0']**2/(8*np.pi)/UNIT_V/UNIT_V
        rho_mean = params['OMEGAM']*rho_crit
        omegam_box = (M_tot / V_sim) / rho_crit
        if np.isclose(omegam_box, params['OMEGAM'], rtol=1e-9):
            if not self.silent:
                print(f"The cosmological Omega_m parameter, calculated from \
                      the particle masses: Omega_m={omegam_box:.6f}")
        else:
            self.mass *= params['OMEGAM'] / omegam_box
            if not self.silent:
                print(f"The particle masses were rescaled to fit with the \
                      cosmological parameter Omega_m={params['OMEGAM']}")
        # Calculate mass statistics after rescaling
        self.mass_list = np.unique(self.mass)
        if not self.silent:
            print(f"Number of different masses:\t{len(self.mass_list)}")
        self.M_box = rho_mean * params['LBOX']**3
        print("...done.\n")

    def periodic_shift(self, params):
        '''
        Periodically shift the input glass to the desired box size and
        center it in the box.
        
        Parameters:
        -----------
        params : dict
            Dictionary containing the cosmological parameters.
        '''
        print("Periodically shifting the input glass...")
        shift = np.array([params[k] for k in ('VOIX', 'VOIY', 'VOIZ')])
        self.pos = (self.pos + shift) % params['LBOX']
        print("...done.\n")