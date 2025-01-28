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
    def __init__(self, snapshot, silent=False):
        self.snapshot = snapshot
        self.N_part = snapshot.shape[0]  # Number of particles
        self.mass_list = None            # List of unique particle masses
        self.M_box = None                # Total mass in the box (in Msol)
        self.silent = silent

    def rescale_snapshot_mass(self, params, midx=6):
        '''
        Rescale the particle masses to fit the cosmological parameters.
        
        Parameters:
        -----------
        params : dict
            Dictionary containing the cosmological parameters.
        midx : int, optional; default=6
            Index of the mass column in the snapshot array.
        '''
        M_tot = np.sum(self.snapshot[:, midx])
        V_sim = 4.0*np.pi/3.0 * params['RSIM']**3
        omegam_box = (M_tot / V_sim) / params['RHO_CRIT']
        if np.isclose(omegam_box, params['OMEGAM'], rtol=1e-9):
            print(f"The cosmological Omega_m parameter, calculated from the particle masses: Omega_m={omegam_box:.6f}")
        else:
            self.snapshot[:, midx] = self.snapshot[:, midx] * params['OMEGAM'] / omegam_box
            print(f"The particle masses were rescaled to fit with the cosmological parameter Omega_m={params['OMEGAM']}")
        # Calculate mass statistics after rescaling
        self.mass_list = np.unique(self.snapshot[:, midx])
        print(f"Number of different masses:\t{len(self.mass_list)}")
        self.M_box = params['RHO_MEAN'] * params['LBOX']**3

    def periodic_shift(self, params, cidx=(1, 2, 3)):
        '''
        Periodically shift the input glass.
        
        Parameters:
        -----------
        params : dict
            Dictionary containing the cosmological parameters.
        cidx : tuple, optional; default=(1, 2, 3)
            Indices of the columns containing the x, y, and z coordinates.
        '''
        print("Periodically shifting the input glass...")
        for i, vi in enumerate(['VOIX', 'VOIY', 'VOIZ']):
            self.snapshot[:, cidx[i]] += params[vi]
        self.snapshot[:, cidx] = self.snapshot[:, cidx] % params['LBOX']
        print("...done.\n")

    def create_nsample_mass_lut(self, params, midx=6):
        '''
        Creates a lookup table  for the number of particles per mass bin.

        Parameters:
        -----------
        params : dict
            Dictionary containing the cosmological parameters.
        midx : int, optional; default=6
            Index of the mass column in the snapshot array.
        '''
        if self.mass_list is None or self.M_box is None:
            self.rescale_snapshot_mass(params, midx=midx)
        
        nsample_mass_counts = [
            [mass, np.uint32(np.cbrt(self.M_box / mass))] for mass in self.mass_list
        ]
        nsample_mass_lut = np.zeros((params['NGRIDSAMPLES'], 2), dtype=np.uint32)
        d_nsample = np.uint32(np.ceil(len(self.mass_list) / params['NGRIDSAMPLES']))
        
        print("The generated Nsample list:")
        print("ID\tNsample\tMass(in 10e11Msol)")
        # Populate lookup table
        nsample_mass_sorted = sorted(nsample_mass_counts, key=lambda x: x[0])
        for si in reversed(range(params['NGRIDSAMPLES']-1)):
            if si == params['NGRIDSAMPLES']-1:
                idx = 0
            else:
                idx = len(nsample_mass_sorted) - 1 - si*d_nsample
            nsample_mass_lut[si] = nsample_mass_sorted[idx]
            print(f"{si}\t{nsample_mass_lut[si, 0]}\t{nsample_mass_lut[si, 1]/1e11:.6f}")
        self.nsample_mass_lut = nsample_mass_lut