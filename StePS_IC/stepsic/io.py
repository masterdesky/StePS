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
from typing import Dict, Any

import h5py
import numpy as np
from pathlib import Path

# Gadget IO library for reading Gadget snapshots
# Download from https://www.github.com/masterdesky/glio
try:
    import glio
except ImportError as _err:
    glio = None
    # _GLIO_IMPORT_ERROR = _err

import logging
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
        path: Path,
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
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        loader = CosmoIO._find_loader(path)
        return loader(
            path,
            part_type=part_type,
            constant_res=constant_res,
            dtype=dtype
        )
    
    @staticmethod
    def save_snapshot(path: Path, data: "CosmoData", fmt: str, **kwargs):
        '''
        Saves a cosmological snapshot to a file. Currently, only HDF5
        format is supported for writing.

        Parameters
        ----------
        fname : pathlib.Path
            Path to the output snapshot file.
        data : stepsic.CosmoData
            A structured array containing the particle data.
        fmt : str
            The file format to save the snapshot in.
        '''
        path = path.expanduser().resolve()
        saver = CosmoIO._find_saver(fmt)
        saver(path, data=data, **kwargs)

    _LOADER_MAP = {
        ".dat": lambda p, **kw: CosmoIO._load_ascii(p, **kw),
        ".txt": lambda p, **kw: CosmoIO._load_ascii(p, **kw),
        ".hdf5": lambda p, **kw: CosmoIO._load_hdf5(p, **kw),
        ".h5": lambda p, **kw: CosmoIO._load_hdf5(p, **kw)
    }
    _SAVER_MAP = {
        "ascii": lambda p, **kw: CosmoIO._save_ascii(p, **kw),
        "gadget": lambda p, **kw: CosmoIO._save_gadget(p, **kw),
        "hdf5": lambda p, **kw: CosmoIO._save_hdf5(p, **kw),
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
                f'`{path.name}` has an unsupported extension and glio is not installed.'
            )
        return CosmoIO._load_gadget
    
    @staticmethod
    def _find_saver(ext: str):
        '''Return the appropriate backend writer for the save method.'''
        if ext in CosmoIO._SAVER_MAP:
            return CosmoIO._SAVER_MAP[ext]
        raise UnsupportedFormatError('Only HDF5 files are supported.')

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
    def _load_hdf5(path: Path, **kwargs):
        '''Load a cosmological snapshot from an HDF5 file.'''
        logger.info(f"Reading the input HDF5 files ...")
        part_type = kwargs.get('part_type', 1)
        dtype = kwargs.get('dtype', np.float32)
    
        stem = path.stem.split('.')[0]
        files = sorted(path.parent.glob(f'{stem}*.hdf5'))
        if not files:
            raise FileNotFoundError(path)
        particleIDs, coordinates, velocities, masses = [], [], [], []
        for f in files:
            logger.info(f'Opening {f}...')
            with h5py.File(f, 'r') as hdf:
                particleIDs.append(hdf[f'/PartType{part_type}/ParticleIDs'][:])
                coordinates.append(hdf[f'/PartType{part_type}/Coordinates'][:])
                velocities.append(hdf[f'/PartType{part_type}/Velocities'][:])
                m = hdf[f'/PartType{part_type}/Masses'][:]
                mass_table = hdf['/Header'].attrs['MassTable'][part_type]
                if np.all(m == 0):
                    m = np.full_like(particleIDs, mass_table, dtype=dtype)
                if kwargs.get('constant_res', True):
                    m *= mass_table
                masses.append(m)
        particleIDs = np.concatenate(particleIDs, dtype=np.uint64)
        coordinates = np.concatenate(coordinates, dtype=dtype)
        velocities = np.concatenate(velocities, dtype=dtype)
        masses = np.concatenate(masses, dtype=dtype)
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
        return particleIDs, coordinates, velocities, masses
    
    @staticmethod
    def _save_ascii(path: Path, data: "CosmoData", **kwargs):
        path = path.with_suffix('.dat')
        '''Save a cosmological snapshot to an ASCII file.'''
        raise NotImplementedError
    
    @staticmethod
    def _save_gadget(path: Path, data: "CosmoData", **kwargs):
        raise NotImplementedError

    @staticmethod
    def _save_hdf5(path: Path, data: "CosmoData", **kwargs):
        '''
        Saves data of a single particle type in GADGET-format HDF5 file.

        Compatible with GADGET, GIZMO and StePS.

        Parameters
        ----------
        path : Path
            The file path to save the snapshot.
        data : stepsic.CosmoData
            The cosmological data to save.

        .. Optional Parameters :
        header arguments
            Additional parameters for the snapshot header. TODO.
        part_type : int
            The particle type to save (e.g., 1 for dark matter in Gadget).
        dtype : numpy.dtype
            The data type to use for the snapshot positions, velocities
            and masses.
        '''
        path = path.with_suffix('.hdf5')
        with h5py.File(path, 'w') as hdf_file:
            part_type = kwargs.get('part_type', 1)
            dtype = kwargs.get('dtype', np.float32)

            h = hdf_file.create_group("/Header")
            num_part_array = np.zeros(6, dtype=np.uint32)
            num_part_array[part_type] = data.N_part
            h.attrs['NumPart_ThisFile'] = num_part_array
            h.attrs['NumPart_Total'] = num_part_array
            h.attrs['NumPart_Total_HighWord'] = np.zeros(6, dtype=np.uint32)
            h.attrs['MassTable'] = np.zeros(6, dtype=dtype)
            h.attrs['Time'] = 1.0 / (kwargs.get('Redshift', 0) + 1.0)
            h.attrs['Redshift'] = float(kwargs.get('Redshift', 0.0))
            h.attrs['BoxSize'] = float(kwargs.get('BoxSize', 0.0))
            h.attrs['NumFilesPerSnapshot'] = kwargs.get('NumFilesPerSnapshot', 1)
            h.attrs['Omega0'] = float(kwargs.get('Omega0', 0.0))
            h.attrs['OmegaLambda'] = float(kwargs.get('OmegaLambda', 0.0))
            h.attrs['HubbleParam'] = float(kwargs.get('HubbleParam', 0.0))
            h.attrs['Flag_Sfr'] = float(kwargs.get('Flag_Sfr', 0.0))
            h.attrs['Flag_Cooling'] = float(kwargs.get('Flag_Cooling', 0.0))
            h.attrs['Flag_StellarAge'] = float(kwargs.get('Flag_StellarAge', 0.0))
            h.attrs['Flag_Metals'] = float(kwargs.get('Flag_Metals', 0.0))
            h.attrs['Flag_Feedback'] = float(kwargs.get('Flag_Feedback', 0.0))
            h.attrs['Flag_Entropy_ICs'] = float(kwargs.get('Flag_Entropy_ICs', 0.0))

            p = hdf_file.create_group(f"/PartType{part_type}")
            p.create_dataset('ParticleIDs', data=data.id)
            p.create_dataset('Coordinates', data=data.pos, dtype=dtype)
            p.create_dataset('Velocities', data=data.vel, dtype=dtype)
            p.create_dataset('Masses', data=data.mass, dtype=dtype)
