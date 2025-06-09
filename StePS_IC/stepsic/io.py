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
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from .data import CosmoData

# Gadget IO library for reading Gadget snapshots
# Download from https://www.github.com/masterdesky/glio
try:
    import glio
except ImportError as _err:
    glio = None
    # _GLIO_IMPORT_ERROR = _err

logger = logging.getLogger(__name__)


class UnsupportedFormatError(RuntimeError):
    '''Raised when CosmoIO encounters an unknown file format.'''


class CosmoIO:
    '''
    A stateless class for loading cosmological snapshots from various
    formats (ASCII, HDF5, NPY, Gadget) and returning them as a
    structured numpy array.
    '''
    @staticmethod
    def load_snapshot(
            fname: str,
            *,
            part_type: int = 1,
            constant_res: bool = False,
            dtype: np.dtype = np.float32
    ):
        '''
        Loads a Gadget-format snapshot of a cosmological simulation from
        either an ASCII, HDF5, NPY or Gadget-format input file.

        Parameters
        ----------
        fname : str
            Path to the *first* snapshot file on disk. All accompanying
            parts (e.g. ``snap_001``, ``snap_002``, etc.) are detected
            and concatenated automatically for Gadget or HDF5 inputs.
        part_type : int
            The particle type to load. For Gadget snapshots, this is the
            part type index (1 for gas, 2 for dark matter, etc.). For HDF5
            snapshots, this is the part type group (e.g. 'PartType1').
        constant_res : bool
            If True, the snapshot is assumed to have constant mass resolution.
        dtype : numpy.dtype
            The data type to use for the snapshot. If not specified, float32 is used.
        '''
        path = Path(fname).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        loader = CosmoIO._find_loader(path)
        return loader(
            path,
            part_type=part_type,
            constant_res=constant_res,
            dtype=dtype
        )
    
    _LOADER_MAP = {
        ".dat": lambda p, **kw: CosmoIO._load_ascii(p, **kw),
        ".txt": lambda p, **kw: CosmoIO._load_ascii(p, **kw),
        ".npy": lambda p, **kw: CosmoIO._load_npy(p, **kw),
        ".hdf5": lambda p, **kw: CosmoIO._load_hdf5(p, **kw),
        ".h5": lambda p, **kw: CosmoIO._load_hdf5(p, **kw)
    }

    @staticmethod
    def _find_loader(path: Path):
        '''Return the appropriate backend reader for ``path``.'''
        ext = path.suffix.lower()
        if ext in CosmoIO._LOADER_MAP:
            return CosmoIO._LOADER_MAP[ext]

        # Anything else we treat as Gadget if glio is present
        if glio is None:
            raise UnsupportedFormatError(
                f"'{path.name}' has an unsupported extension and glio is not installed."
            )
        return CosmoIO._load_gadget

    @staticmethod
    def _load_ascii(path: Path, **kwargs):
        '''Load a cosmological snapshot from an ASCII file.'''
        logger.info(f"Reading ASCII file: {path}")
        dtype = kwargs.get('dtype', np.float32)
        data = np.loadtxt(path)
        particleIDs = np.arange(data.shape[0], dtype=np.uint64)
        coordinates = np.array(data[:, 1:4], dtype=dtype)
        velocities = np.array(data[:, 4:7], dtype=dtype)
        masses = np.array(data[:, 7], dtype=dtype)
        logger.info("...done.")
        return particleIDs, coordinates, velocities, masses

    @staticmethod
    def _load_npy(path: Path, **kwargs):
        '''Load a cosmological snapshot from an NPY file.'''
        logger.info(f"Reading NPY file: {path}")
        dtype = kwargs.get('dtype', np.float32)
        data = np.load(path, allow_pickle=False)
        if isinstance(data, np.ndarray):
            # Fall back to ASCII loader for column‑major array layout
            return CosmoIO._load_ascii(path, **kwargs)
        if not isinstance(data, dict):
            raise ValueError(f"Could not interpret .npy snapshot '{path}'.")
        return (
            data["ParticleIDs"].astype(np.uint64, copy=False),
            data["Coordinates"].astype(dtype, copy=False),
            data["Velocities"].astype(dtype, copy=False),
            data["Masses"].astype(dtype, copy=False)
        )

    @staticmethod
    def _load_hdf5(path: Path, **kwargs):
        '''Load a cosmological snapshot from an HDF5 file.'''
        logger.info(f"Reading the input HDF5 files ...")
        part_type = kwargs.get('part_type', 1)
        dtype = kwargs.get('dtype', np.float32)
    
        stem = path.stem.split(".")[0]
        files = sorted(path.parent.glob(f"{stem}*.hdf5"))
        if not files:
            raise FileNotFoundError(path)
        particleIDs, coordinates, velocities, masses = [], [], [], []
        for f in files:
            logger.info(f"Opening {f}...")
            with h5py.File(f, "r") as hdf:
                particleIDs.append(hdf[f'/PartType{part_type}/ParticleIDs'][:])
                coordinates.append(hdf[f'/PartType{part_type}/Coordinates'][:])
                velocities.append(hdf[f'/PartType{part_type}/Velocities'][:])
                m = hdf[f'/PartType{part_type}/Masses'][:]
                if not kwargs.get('constant_res', False):
                    m *= hdf['/Header'].attrs['MassTable'][1]
                masses.append(m)
        particleIDs = np.concatenate(particleIDs, dtype=np.uint64)
        coordinates = np.concatenate(coordinates, dtype=dtype)
        velocities = np.concatenate(velocities, dtype=dtype)
        masses = np.concatenate(masses, dtype=dtype)
        logger.info("...done.")
        return particleIDs, coordinates, velocities, masses

    @staticmethod
    def _load_gadget(path: Path, **kwargs):
        '''Load a cosmological snapshot from a Gadget binary.'''
        logger.info(f"Reading Gadget file: {path}")
        part_type = kwargs.get('part_type', 1)
        s = glio.GadgetSnapshot(path)
        particleIDs = s.ID[part_type]
        coordinates = s.pos[part_type]
        velocities = s.vel[part_type]
        masses = s.mass[part_type]
        logger.info("...done.")
        return particleIDs, coordinates, velocities, masses


def load_hdf5_params(path: Path):
    '''TODO
    '''
    if not path.lower().endswith('.hdf5'):
        raise Exception('Error: input file {path} is not in hdf5 format!')
    with h5py.File(path, 'r') as f:
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
