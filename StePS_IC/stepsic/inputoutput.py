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

import os
import h5py
import numpy as np

import glio

class CosmoIO:
    '''
    TODO: : save functionality
    '''
    def __init__(self, silent=False):
        self.silent = silent

    def load_snapshot(self, fname, *, constant_res=False, double_precision=False):
        '''
        Loads a Gadget-format snapshot of a cosmological simulation from
        either an ASCII, HDF5 or Gadget-format input file.

        Parameters:
        -----------
        fname : str
            Path to the input file, containing a cosmological snapshot.
            The file can be in ASCII, HDF5, NPY or Gadget format.
        constant_res : bool, optional; default: False
            If True, the snapshot is assumed to have constant mass resolution.
        double_precision : bool, optional; default: False
            If True, the snapshot is assumed to have double precision.
        '''
        float_dtype = np.float64 if double_precision else np.float32

        ext = fname.lower().split('.')[-1]
        if ext == 'dat':
            return self._load_ascii_snapshot(fname, float_dtype)
        elif ext == 'hdf5':
            return self._load_hdf5_snapshot(fname, float_dtype, constant_res)
        elif ext == 'npy':
            return self._load_npy_snapshot(fname, float_dtype)
        else:   # assume Gadget binary (handles multi-part internally)
            return self._load_gadget_snapshot(fname)

    def _load_ascii_snapshot(self, fname, float_dtype):
        if not self.silent:
            print(f"\tReading the input ASCII file {fname} ...")
        data = np.loadtxt(fname)
        particleIDs = np.arange(data.shape[0], dtype=np.uint64)
        coordinates = np.array(data[:, :3], dtype=float_dtype)
        velocities = np.array(data[:, 3:6], dtype=float_dtype)
        masses = np.array(data[:, 6], dtype=float_dtype)
        if not self.silent:
            print("\t...done.\n")
        return np.c_[particleIDs, coordinates, velocities, masses]
    
    def _load_npy_snapshot(self, fname, float_dtype):
        if not self.silent:
            print(f"\tReading the input NPY file {fname} …")
        data = np.load(fname, allow_pickle=True)

        if isinstance(data, (dict, np.ndarray)) and hasattr(data, 'keys'):
            # if it is a dict-like container saved with allow_pickle=True
            particleIDs = np.asarray(data['ParticleIDs'], dtype=np.uint64)
            coordinates = np.asarray(data['Coordinates'], dtype=float_dtype)
            velocities = np.asarray(data['Velocities'], dtype=float_dtype)
            masses = np.asarray(data['Masses'], dtype=float_dtype)
        else:
            # if it is a plain array with expected column ordering
            data = np.asarray(data)
            if data.ndim != 2 or data.shape[1] < 10:
                raise ValueError("Unrecognised NPY snapshot format.")
            particleIDs = data[:, 0].astype(np.uint64, copy=False)
            coordinates = data[:, 1:4].astype(float_dtype, copy=False)
            velocities = data[:, 4:7].astype(float_dtype, copy=False)
            masses = data[:, 7].astype(float_dtype, copy=False)
        if not self.silent:
            print("\t…done.\n")
        return np.c_[particleIDs, coordinates, velocities, masses]

    def _load_hdf5_snapshot(self, fname, float_dtype, constant_res):
        if not self.silent:
            print(f"\tReading the input HDF5 files ...")
        input_path = os.path.dirname(fname)
        input_fnames = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.hdf5')]
        if len(input_fnames) > 1:
            if not self.silent:
                print("\tSnapshot is stored in multiple files.")
            input_fnames.sort(key=lambda x: int(x.split('.')[-2]))
        particleIDs, coordinates, velocities, masses = [], [], [], []
        for hdf5_file in input_fnames:
            if not self.silent:
                print(f"\t\tOpening {hdf5_file} ...")
            with h5py.File(hdf5_file, 'r') as f:
                particleIDs.append(f['/PartType1/ParticleIDs'][:])
                coordinates.append(f['/PartType1/Coordinates'][:])
                velocities.append(f['/PartType1/Velocities'][:])
                if constant_res:
                    masses.append(f['/PartType1/Masses'][:])
                else:
                    masses.append(f['/PartType1/Masses'][:] * f['/Header'].attrs['MassTable'][1])
        particleIDs = np.concatenate(particleIDs, dtype=np.uint64)
        coordinates = np.concatenate(coordinates, dtype=float_dtype)
        velocities = np.concatenate(velocities, dtype=float_dtype)
        masses = np.concatenate(masses, dtype=float_dtype)
        if not self.silent:
            print("\t...done.\n")
        return np.c_[particleIDs, coordinates, velocities, masses]

    def _load_gadget_snapshot(self, fname):
        if not self.silent:
            print(f"\tReading the input Gadget file {fname} ...")
        s = glio.GadgetSnapshot(fname)
        particleIDs = s.ID[1]
        coordinates = s.pos[1]
        velocities = s.vel[1]
        masses = s.mass[1]
        if not self.silent:
            print("\t...done.\n")
        return np.c_[particleIDs, coordinates, velocities, masses]


def Load_params_from_HDF5_snap(fname):
    '''TODO
    '''
    if not fname.lower().endswith('.hdf5'):
        raise Exception('Error: input file {fname} is not in hdf5 format!')
    with h5py.File(fname, 'r') as f:
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
