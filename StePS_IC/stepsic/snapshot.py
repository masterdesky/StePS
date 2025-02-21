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


class CosmoSnapshot:
    '''
    Implements a class for handling cosmological snapshot data stored in
    a conventional Gadget-like structure for StePS simulations.

    Parameters:
    -----------
    snapshot : ndarray
        Array containing the particle data.
    idx : int, optional; default=0
        Index of the particle ID column in the snapshot array.
    cidx : tuple, optional; default=(1, 2, 3)
        Indices of the x, y, z columns in the snapshot array.
    vidx : tuple, optional; default=(4, 5, 6)
        Indices of the vx, vy, vz columns in the snapshot array.
    midx : int, optional; default=6
        Index of the mass column in the snapshot array.
    '''
    def __init__(self, snapshot, idx=0, cidx=(1, 2, 3), vidx=(4, 5, 6), midx=6,
                 silent=False):
        self.snapshot = snapshot
        self.idx, self.cidx, self.vidx, self.midx = idx, cidx, vidx, midx
        self.silent = silent
    
        # Calculated values
        self.N_part = snapshot.shape[0]  # Number of particles
        self.mass_list = None            # List of unique particle masses
        self.M_box = None                # Total mass in the box (in Msol)

    def rescale_snapshot_mass(self, params):
        '''
        Rescale the particle masses to fit the cosmological parameters.
        
        Parameters:
        -----------
        params : dict
            Dictionary containing the cosmological parameters.
        midx : int, optional; default=6
            Index of the mass column in the snapshot array.
        '''
        M_tot = np.sum(self.snapshot[:, self.midx])
        V_sim = 4.0*np.pi/3.0 * params['RSIM']**3
        omegam_box = (M_tot / V_sim) / params['RHO_CRIT']
        if np.isclose(omegam_box, params['OMEGAM'], rtol=1e-9):
            print(f"The cosmological Omega_m parameter, calculated from \
                  the particle masses: Omega_m={omegam_box:.6f}")
        else:
            self.snapshot[:, self.midx] *= params['OMEGAM'] / omegam_box
            print(f"The particle masses were rescaled to fit with the \
                  cosmological parameter Omega_m={params['OMEGAM']}")
        # Calculate mass statistics after rescaling
        self.mass_list = np.unique(self.snapshot[:, self.midx])
        print(f"Number of different masses:\t{len(self.mass_list)}")
        self.M_box = params['RHO_MEAN'] * params['LBOX']**3

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
        for i, vi in enumerate(['VOIX', 'VOIY', 'VOIZ']):
            self.snapshot[:, self.cidx[i]] += params[vi]
        self.snapshot[:, self.cidx] %= params['LBOX']
        print("...done.\n")

    def create_mass_nsample_lut(self, params):
        '''
        Creates a lookup table for the number of samples per mass bin
        for a variable resolution grid in a regular StePS simulation.

        Parameters:
        -----------
        params : dict
            Dictionary containing the cosmological parameters.
        midx : int, optional; default=6
            Index of the mass column in the snapshot array.
        '''
        if self.mass_list is None or self.M_box is None:
            self.rescale_snapshot_mass(params)
        
        mass_nsample = np.c_[
                self.mass_list,
                np.uint32(np.cbrt(self.M_box / self.mass_list))
        ]
        mass_nsample_lut = np.zeros((params['NGRIDSAMPLES'], 2), dtype=np.uint32)
        d_nsample = np.uint32(np.ceil(len(self.mass_list) / params['NGRIDSAMPLES']))
        
        print("The generated Nsample list:")
        print("ID\tNsample\tMass(in 10e11Msol)")
        # Populate lookup table by starting with the outermost mass bin,
        # while skipping the innermost layer
        mass_nsample_sorted = sorted(mass_nsample, key=lambda x: x[0])
        for si in reversed(range(params['NGRIDSAMPLES']-1)):
            if si == params['NGRIDSAMPLES']-2:
                idx = 0
            else:
                idx = len(mass_nsample_sorted) - 1 - si*d_nsample
            mass_nsample_lut[si] = mass_nsample_sorted[idx]
            print(f"{si}\t{mass_nsample_lut[si, 0]}\t{mass_nsample_lut[si, 1]/1e11:.6f}")
        self.mass_nsample_lut = mass_nsample_lut