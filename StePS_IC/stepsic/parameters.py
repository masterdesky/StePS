#*******************************************************************************#
#  stepsic - An initial condition generator for                                 #
#            STEreographically Projected cosmological Simulations               #
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

import importlib.resources
from pathlib import Path
import toml
import numpy as np
from textwrap import dedent

from dataclasses import dataclass, field

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CosmoParameters:
    def __init__(self, filepath):

        with open(filepath, 'r') as file:
            self.P = toml.load(file)
        self._load_default_cosmology(parameter_set=self.P['COSMOLOGY'])
        self._process_cosmo_params()
        self._process_ic_params()
    
    def _check_exists(self, name):
        '''Check if the parameter exists in the parameter list.'''
        if name not in self.P:
            raise KeyError(f'{name} is not defined in the parameters!')
        return self.P[name]

    def _check_path(self, name, makedir=False):
        '''Check if the parameter is a valid file path.'''
        value = self.P[name]
        if not isinstance(value, (str, Path)):
            raise TypeError(f'{name} must be a string or a Path object!')
        path = Path(value)
        if not path.exists():
            if makedir:
                path.mkdir(parents=True)
            else:
                raise FileNotFoundError(f'{name} does not exist: {path}')
        return path

    def _check_string(self, name):
        '''Check if the parameter is a string.'''
        value = self.P[name]
        if not isinstance(value, str):
            raise TypeError(f'{name} must be a string!')
        return value.strip().lower()

    def _check_scalar(self, name, dtype=np.float64):
        '''Check if the parameter is a scalar value.'''
        value = self.P[name]
        if not np.isscalar(value):
            raise TypeError(f'{name} must be a scalar!')
        return dtype(value)

    def _check_array_or_scalar(self, name, length=3, dtype=np.float64):
        '''Check if the parameter is a scalar or a container of a given length.'''
        value = self.P[name]
        if not (np.isscalar(value) or (hasattr(value, '__len__') and len(value) == length)):
            raise TypeError(f'{name} must be a scalar or a container of {length} elements!')
        if not np.issubdtype(np.asarray(value).dtype, np.number):
            raise TypeError(f'{name} must be a scalar or numeric array!')
        return np.broadcast_to(value, (length,)).astype(dtype)

    def _load_default_cosmology(self, *,
            parameter_set='Planck2018EE+BAO+SN', value_type='best'):
        '''TODO'''
        if value_type not in ['best', 'mean']:
            raise ValueError("The 'value_type' must be either 'best' or 'mean'!")

        with importlib.resources.path('stepsic.config', 'cosmology.toml') as filepath:
            with open(filepath, 'r') as f:
                config_data = toml.load(f)

        if parameter_set not in config_data:
            raise KeyError(f"Parameter set '{parameter_set}' not found in the TOML data.")

        config_data = config_data[parameter_set]

        for key, value in config_data.items():
            if isinstance(value, dict):
                # Values with best fit/68% limit values
                self.P.setdefault(key, value[value_type])
                if value_type == 'mean' and key not in self.P:
                    self.P[f'{key}_err'] = value['error']
            else:
                # Single value parameters
                self.P.setdefault(key, value)

    def _process_cosmo_params(self):
        '''
        Print all cosmological parameters found in the parameter list.
        '''
        self.P['H'] = self.P['H0'] / 100.0
        self.P['OMMH2'] = self.P['OMEGA_M'] * self.P['H']**2
        self.P['OMBH2'] = self.P['OMEGA_B'] * self.P['H']**2
        self.P['OMEGA_NU'] = self.P.get('MNU', 0.06) / 93.14 / self.P['H']**2
        self.P['OMEGA_C'] = self.P['OMEGA_M'] - self.P['OMEGA_B'] - self.P['OMEGA_NU']
        self.P['OMCH2'] = self.P['OMEGA_C'] * self.P['H']**2
        self.P['OMEGA_K'] = 1.0 - self.P['OMEGA_M'] - self.P['OMEGA_L']
        self.P['SCALE'] = 1.0 / (self.P['REDSHIFT'] + 1.0)

        # Print cosmological parameters
        text = dedent(f'''
        Cosmological Parameters:
        ------------------------
        H0:                {self.P['H0']:.3f} km/s/Mpc
        Omega_m:           {self.P['OMEGA_M']:.6f}
        Omega_m h^2:       {self.P['OMMH2']:.6f}
        Omega_c h^2:       {self.P['OMCH2']:.6f}
        Omega_L:           {self.P['OMEGA_L']:.6f}
        Omega_k:           {self.P['OMEGA_K']:.6f}
        Omega_b:           {self.P['OMEGA_B']:.6f}
        Omega_b h^2:       {self.P['OMBH2']:.6f}
        Sigma8:            {self.P['SIGMA8']:.3f}
        Redshift:          {self.P['REDSHIFT']:.3f}
        Scale factor:      {self.P['SCALE']:.6f}
        ''')
        log.info(text)
    
        SPECTRUM = self._check_string('SPECTRUM')
        if SPECTRUM not in ['input', 'camb']:
            raise ValueError(f'Error: unknown spectrum type `{SPECTRUM}`!\nExiting.')
        if SPECTRUM == 'input':
            _ = self._check_path('FILE_WITH_INPUT_SPECTRUM')
        _ = self._check_scalar('INPUTSPECTRUM_UNITLENGTH_IN_CM')

    def _process_ic_params(self):
        '''Process initial condition parameters from the config file.'''
        _ = self._check_array_or_scalar('NMESH', length=3, dtype=int)
        _ = self._check_array_or_scalar('LBOX', length=3)
        _ = self._check_array_or_scalar('PERIODIC', length=3, dtype=int)
        _ = self._check_array_or_scalar('COI', length=3)
        _ = self._check_scalar('LPTORDER', dtype=int)

        _ = self._check_scalar('R_3D')
        _ = self._check_scalar('D_4D')
        _ = self._check_scalar('NRBINS', dtype=int)

        GEOMETRY = self._check_string('GEOMETRY')
        if GEOMETRY not in ['cylindrical', 'spherical', 'cubical']:
            raise ValueError(f'Error: unknown geometry `{GEOMETRY}`!\nExiting.')
        BIN_MODE = self._check_string('BIN_MODE')
        if BIN_MODE not in ['omega', 'volume']:
            raise ValueError(f'Error: unknown binning mode `{BIN_MODE}`!\nExiting.')

        TYPE = self._check_string('TYPE')
        if TYPE not in ['grid', 'random', 'glass']:
            raise ValueError(f'Error: unknown IC pattern `{TYPE}`!\nExiting.')

        if TYPE == 'glass':
            _ = self._check_exists('INPUT_GLASS')
            _ = self._check_path('INPUT_GLASS')
        _ = self._check_string('IC_PREFIX')
        _ = self._check_path('IC_DIR', makedir=True)
        IC_FORMAT = self._check_string('IC_FORMAT')
        if IC_FORMAT not in ['ascii', 'npy', 'hdf5', 'gadget']:
            raise ValueError(f'Error: unknown output file format `{IC_FORMAT}`!\nExiting.')

        _ = self._check_scalar('NPART')
        _ = self._check_scalar('SEED', dtype=int)

        # Print initial condition parameters
        text = dedent(f'''
        IC parameters
        -------------
        Random seed:                   {self.P['SEED']:d}
        Mesh size:                     {self.P['NMESH']} voxels
        Box size:                      {self.P['LBOX']} Mpc/h
        Periodicity along x-y-z axis:  {self.P['PERIODIC']}
        Center of interest:            {self.P['COI']} Mpc/h
        Euclidean simulation radius:   {self.P['R_3D']} Mpc/h
        Compact. simulation diameter:  {self.P['D_4D']} Mpc/h
        Number of grid samples:        {self.P['NGRIDSAMPLES']:d}
        Glass input file:              {self.P['INPUT_GLASS']}
        IC output directory:           {self.P['IC_DIR']}
        IC name prefix:                {self.P['IC_PREFIX']}
        Comoving IC:                   {self.P['COMOVING']}
        Counter phase simulation:      {self.P['COUNTER']:.2f}
        Phase shift:                   {self.P['PHASE_SHIFT']:.2f} degrees
        ''')
        log.info(text)
        # Supplementary log messages and operations
        _ = self._check_scalar('UNIT_L_IN_CM')
        _ = self._check_scalar('UNIT_M_IN_G')
        _ = self._check_scalar('UNIT_V_IN_CM_PER_S')
        # check of boolean
        if self.P['COMOVING'] not in (0, 1):
            raise ValueError('Error: the COMOVING parameter should be 1 or 0!\nExiting.')