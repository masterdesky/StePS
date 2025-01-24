#*******************************************************************************#
#  StePS_IC.py - An initial condition generator for                             #
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2017-2024 Gabor Racz                                         #
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

# Functions for reading and writing particle data

import os
import h5py
import numpy as np

# import pygadgetreader
from astropy.units import solMass,Mpc,m,s

# from past.translation import autotranslate
# autotranslate(['glio'])
# import glio

#defining functions
def load_snapshot(filename, *, constant_res=False, double_precision=False,
                  silent=False, **kwargs):
    '''
    Loads a Gadget-format snapshot of a cosmological simulations from
    either an ASCII or HDF5 input file.

    Parameters:
    -----------
    filename: str
        Name of the input file
    constant_res: bool, optional; default: False
        If True, the snapshot has constant resolution
    silent: bool, optional; default: False
        If True, suppresses the output
    double_precision: bool, optional; default: False
        If True, uses double precision for the output
    '''
    float_dtype = np.float64 if double_precision else np.float32

    # ASCII snapshot
    if filename.lower().endswith('.dat'):
        if not silent:
            print(f"\tReading the input ASCII file {filename} ...")
        data = np.loadtxt(filename)
        particleIDs = np.arange(data.shape[0], dtype=np.uint64)
        coordinates = np.array(data[:, :3], dtype=float_dtype)
        velocities = np.array(data[:, 3:6], dtype=float_dtype)
        masses = np.array(data[:, 6], dtype=float_dtype)
        if not silent:
            print("\t...done.\n")
    # HDF5 snapshot
    elif filename.lower().endswith('.hdf5'):
        if not silent:
            print(f"\tReading the input HDF5 files ...")
        # Collect all filenames in the parent directory of `filename` that
        # have a '.hdf5' extension. If there are multiple files, sort them
        # by the integer value in the filename as in `filename.<int>.hdf5`.
        parent_dir = os.path.dirname(filename)
        filenames = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith('.hdf5')]
        if len(filenames) > 1:
            if not silent:
                print("\tSnapshot is stored in multiple files.")
            filenames.sort(key=lambda x: int(x.split('.')[-2]))
        # Read the particle data from each file and concatenate them in
        # the corresponding arrays
        particleIDs, coordinates, velocities, masses = [], [], [], []
        for hdf5_file in filenames:
            if not silent:
                print(f"\t\tOpening {hdf5_file} ...")
            with h5py.File(hdf5_file, 'r') as f:
                particleIDs.append(f['/PartType1/ParticleIDs'][:])
                coordinates.append(f['/PartType1/Coordinates'][:])
                velocities.append(f['/PartType1/Velocities'][:])
                if constant_res:
                    masses.append(f['/PartType1/Masses'][:])
                else:
                    masses.append(f['/PartType1/Masses'][:] * f['/Header'].attrs['MassTable'][1])
        if not silent:
            print("\t...done.\n")
        particleIDs = np.concatenate(particleIDs, dtype=np.uint64)
        coordinates = np.concatenate(coordinates, dtype=float_dtype)
        velocities = np.concatenate(velocities, dtype=float_dtype)
        masses = np.concatenate(masses, dtype=float_dtype)
    # (Assume) Gadget-format snapshot
    else:
        if not silent:
            print(f"\tReading the input Gadget file {filename} ...")
        particleIDs = pygadgetreader.readsnap(filename, 'IDs', 'dm')
        coordinates = pygadgetreader.readsnap(filename, 'pos', 'dm')
        velocities = pygadgetreader.readsnap(filename, 'vel', 'dm')
        masses = pygadgetreader.readsnap(filename, 'mass', 'dm')
        if not silent:
            print("\t...done.\n")
    return particleIDs, coordinates, velocities, masses


def Load_params_from_HDF5_snap(filename):
    if not filename.lower().endswith('.hdf5'):
        raise Exception('Error: input file {filename} is not in hdf5 format!')
    with h5py.File(filename, 'r') as f:
        Ntot = int(f['/Header'].attrs['NumPart_Total'][1])
        z = np.double(f['/Header'].attrs['Redshift'])
        Om = np.double(f['/Header'].attrs['Omega0'])
        Ol = np.double(f['/Header'].attrs['OmegaLambda'])
        H0 = np.double(f['/Header'].attrs['HubbleParam'])*100.0
    return z, Om, Ol, H0, Ntot


def writeHDF5snapshot(dataarray, outputfilename, Linearsize, Redshift, OmegaM, OmegaL, HubbleParam, precision):
    '''
    Function for writing out IC in hdf5 format.
    Parameters:
        dataarray - numpy array containing the particle data (coordinates, velocities, masses)
        outputfilename - name of the output file
        Linearsize - Linear size of the IC (simulation)
        Redshift - initial redshift
        OmegaM - Matter density
        OmegaL - Dark energy density
        HubbleParam - Hubble parameter
        precision - Floating point precision of the IC (0: 32bit; 1: 64bit)
    '''
    if np.int(precision) == 0:
        HDF5datatype = 'float32'
        npdatatype = np.float32
        print("Saving in 32bit HDF5 format.")
    if np.int(precision) == 1:
        HDF5datatype = 'double'
        npdatatype = np.float64
        print("Saving in 64bit HDF5 format.")
    N = len(dataarray)
    HDF5_snapshot = h5py.File(outputfilename, "w")
    #Creating the header
    header_group = HDF5_snapshot.create_group("/Header")
    #Writing the header attributes
    header_group.attrs['NumPart_ThisFile'] = np.array([0,N,0,0,0,0], dtype=np.uint32)
    header_group.attrs['NumPart_Total'] = np.array([0,N,0,0,0,0], dtype=np.uint32)
    header_group.attrs['NumPart_Total_HighWord'] = np.array([0,0,0,0,0,0], dtype=np.uint32)
    header_group.attrs['MassTable'] = np.array([0,0,0,0,0,0], dtype=npdatatype)
    header_group.attrs['Time'] = np.double(1.0/(Redshift+1))
    header_group.attrs['Redshift'] = np.double(Redshift)
    header_group.attrs['BoxSize'] = np.double(Linearsize)
    header_group.attrs['NumFilesPerSnapshot'] = np.int(1)
    header_group.attrs['Omega0'] = np.double(OmegaM)
    header_group.attrs['OmegaLambda'] = np.double(OmegaL)
    header_group.attrs['HubbleParam'] = np.double(HubbleParam)
    header_group.attrs['Flag_Sfr'] = np.int(0)
    header_group.attrs['Flag_Cooling'] = np.int(0)
    header_group.attrs['Flag_StellarAge'] = np.int(0)
    header_group.attrs['Flag_Metals'] = np.int(0)
    header_group.attrs['Flag_Feedback'] = np.int(0)
    header_group.attrs['Flag_Entropy_ICs'] = np.int(0)
    #Header created.
    #Creating datasets for the particle data
    particle_group = HDF5_snapshot.create_group("/PartType1")
    X = particle_group.create_dataset("Coordinates", (N, 3), dtype=HDF5datatype)
    V = particle_group.create_dataset("Velocities", (N, 3), dtype=HDF5datatype)
    IDs = particle_group.create_dataset("ParticleIDs", (N,), dtype='uint64')
    M = particle_group.create_dataset("Masses", (N,), dtype=HDF5datatype)
    #Saving the particle data
    X[:,:] = dataarray[:, 0:3]
    V[:,:] = dataarray[:, 3:6]
    M[:] = dataarray[:, 6]
    IDs[:] = np.arange(N, dtype=np.uint64)
    HDF5_snapshot.close()
    return

def ascii2gadget(infile, outfile, Lbox, H0, UNITLENGTH_IN_CM,
                 UNIT_D=3.0856775814671917e24):
    '''
    Converts a StePS ASCII snapshot to Gadget format.

    Parameters:
    -----------
    infile : str
        Name of the input StePS ASCII file.
    outfile : str
        Name of the output Gadget-format file.
    Lbox : float
        Box size in Mpc/h.
    H0 : float
        Hubble constant in km/s/Mpc.
    UNITLENGTH_IN_CM : float
        Unit length in cm.
    UNIT_D : float, optional; default: 3.0856775814671917e24
        Unit distance in cm in the StePS simulator.
    '''
    #Reading the input data
    particle_data = np.fromfile(infile, count=-1, sep='\t', dtype=np.float64)
    particle_data = particle_data.reshape(int(len(particle_data)/7),7)
    h = H0/100.0
    #Creating array of X coordinates and V velocities
    X = particle_data[:,0:3] * h * UNIT_D / UNITLENGTH_IN_CM
    V = particle_data[:,3:6] #velocities
    M = particle_data[:,6] # Masses
    Npart = len(X)
    del(particle_data)
    #Creating the Gadget-snapshot
    Gadget_snapshot = glio.GadgetSnapshot(outfile)
    Gadget_snapshot.header.npart = np.array([0,Npart,0,0,0,0], dtype=np.int32)
    Gadget_snapshot.header.mass = np.array([0.0,1.0,0.0,0.0,0.0,0.0], dtype=np.float64)
    Gadget_snapshot.header.time= np.array([0.0078125], dtype=np.float64)
    Gadget_snapshot.header.redshift= np.array([127.0], dtype=np.float64)
    Gadget_snapshot.header.flag_sfr= np.array([0], dtype=np.int32)
    Gadget_snapshot.header.flag_feedback= np.array([0], dtype=np.int32)
    Gadget_snapshot.header.npartTotal =  np.array([0,Npart,0,0,0,0], dtype=np.int32)
    Gadget_snapshot.header.flag_cooling = np.array([0], dtype=np.int32)
    Gadget_snapshot.header.num_files = np.array([1], dtype=np.int32)
    Gadget_snapshot.header.BoxSize = np.array([Lbox*h*UNIT_D/UNITLENGTH_IN_CM], dtype=np.float64)
    Gadget_snapshot.header.Omega0 =  np.array([1.0], dtype=np.float64)
    Gadget_snapshot.header.OmegaLambda =  np.array([0.0], dtype=np.float64)
    Gadget_snapshot.header.HubbleParam = np.array([h], dtype=np.float64)
    Gadget_snapshot.header.flag_stellarage = np.array([0], dtype=np.int32)
    Gadget_snapshot.header.flag_metals = np.array([0], dtype=np.int32)
    Gadget_snapshot.header.npartTotalHighWord = np.array([0,0,0,0,0,0], dtype=np.uint32)
    Gadget_snapshot.header.flag_entropy_instead_u = np.array([0], dtype=np.int32)
    Gadget_snapshot.header._padding = np.zeros(15,dtype=np.int32)
    Gadget_snapshot.ID[1] = np.array(range(0,Npart), dtype=np.uint32)
    Gadget_snapshot.pos[1] = np.array(X, dtype=np.float32)
    Gadget_snapshot.vel[1] = np.array(V, dtype=np.float32)
    Gadget_snapshot.save(outfile)
    del(X)
    del(V)
    del(M)
    return;
