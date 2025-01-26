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
import yaml
import numpy as np
from textwrap import dedent


class CosmoParameters:
    def __init__(self):
        self.params = None

    def load_parameters(self, filepath):
        '''
        Load the parameter file and store the parameters in a dictionary.

        Parameters:
        -----------
        filepath : str
            Path to the parameter file.
        '''
        with open(filepath, 'r') as file:
            self.params = yaml.safe_load(file)
        self._process_cosmo_params()
        self._process_ic_params()
        self._process_icgen_parameters()
        return self.params

    def _process_cosmo_params(self):
        '''
        Print all cosmological parameters found in the parameter list.

        Parameters:
        -----------
        self.params : dict
            Dictionary containing the parameter list of a StePS simulation.
        '''
        self.params['OMMH2'] = self.params['OMEGAM'] * (self.params['H0']/100.0)**2
        self.params['OMCH2'] = (self.params['OMEGAM'] - self.params['OMEGAB']) * (self.params['H0']/100.0)**2
        self.params['OMK']   = 1.0 - self.params['OMEGAM'] - self.params['OMEGAL']
        self.params['OMBH2'] = self.params['OMEGAB'] * (self.params['H0']/100.0)**2
        self.params['SCALE'] = 1.0/(self.params['REDSHIFT']+1.0)

        # Print cosmological parameters
        text = dedent(f'''
        Cosmological Parameters:
        ------------------------
        Omega_m:            {self.params['OMEGAM']:.6f}      (Ommh2={self.params['OMMH2']:.6f}; Omch2={self.params['OMCH2']:.6f})
        Omega_lambda:       {self.params['OMEGAL']:.6f}
        Omega_k:            {self.params['OMK']:.6f}
        Omega_b:            {self.params['OMEGAB']:.6f}      (Ombh2={self.params['OMBH2']:.6f})
        H0:                 {self.params['H0']:.3f} km/s/Mpc
        Redshift:           {self.params['REDSHIFT']:.3f}    (a={self.params['SCALE']:.6f})
        Sigma8:             {self.params['SIGMA8']:.3f}
        Dark energy model:  {self.params['DARKENERGYMODEL']}
        ''')
        print(text)
        # Supplementary log messages and operations
        if self.params['DARKENERGYMODEL'] == 'Lambda':
            print('\n')
        elif not self.params['USECAMBINPUTSPECTRUM']:
            raise ValueError('Error: For non-standard dark energy parametrization '\
                            'USECAMBINPUTSPECTRUM has to be set True!\nExiting.\n')
        elif self.params['DARKENERGYMODEL'] == 'w0':
            print(f'w = {self.params['DARKENERGYPARAMS'][0]:.3f}\n')
        elif self.params['DARKENERGYMODEL'] == 'CPL':
            print(f'w0 = {self.params['DARKENERGYPARAMS'][0]:.3f}\n\
                    wa = {self.params['DARKENERGYPARAMS'][1]:.3f}\n')
        else:
            raise ValueError('Error: unkown dark energy parametrization!\nExiting.\n')
        self.params['INPUTSPECTRUM_UNITLENGTH_IN_CM'] = \
                np.float64(self.params['INPUTSPECTRUM_UNITLENGTH_IN_CM'])

def _process_ic_params(self):
    '''
    Print all initial condition parameters found in the parameter list.

    Parameters:
    -----------
    self.params : dict
        Dictionary containing the parameter list of a StePS simulation.
    '''

    # Print initial condition parameters
    text = dedent(f'''
    IC parameters:
    --------------
    Lbox:                           {self.params['LBOX']:.3f} Mpc
    Rsim:                           {self.params['RSIM']:.3f} Mpc
    VOI_x:                          {self.params['VOIX']:.3f} Mpc
    VOI_y:                          {self.params['VOIY']:.3f} Mpc
    VOI_z:                          {self.params['VOIZ']:.3f} Mpc
    Seed:                           {self.params['SEED']:d}
    Spheremode:                     {self.params['SPHEREMODE']:d}
    WhichSpectrum:                  {self.params['WHICHSPECTRUM']:d}
    FileWithInputSpectrum:          {self.params['FILEWITHINPUTSPECTRUM']}
    InputSpectrum_UnitLength_in_cm: {self.params['INPUTSPECTRUM_UNITLENGTH_IN_CM']:.3e}
    ReNormalizeInputSpectrum:       {self.params['RENORMALIZEINPUTSPECTRUM']:d}
    ShapeGamma:                     {self.params['SHAPEGAMMA']:.3f}
    PrimordialIndex:                {self.params['PRIMORDIALINDEX']:.3f}
    Ngrid samples:                  {self.params['NGRIDSAMPLES']:d}
    GlassFile:                      {self.params['GLASSFILE']}
    OutDir:                         {self.params['OUTDIR']}
    FileBase:                       {self.params['FILEBASE']}
    Comoving IC:                    {self.params['COMOVINGIC']}
    Number of MPI tasks:            {self.params['MPITASKS']:d}
    H0 independent units:           {self.params['HINDEPENDENTUNITS']:d}
    ''')
    print(text)
    # Supplementary log messages and operations
    self.params['UNITLENGTH_IN_CM'] = np.float64(self.params['UNITLENGTH_IN_CM'])
    self.params['UNITMASS_IN_G'] = np.float64(self.params['UNITMASS_IN_G'])
    self.params['UNITVELOCITY_IN_CM_PER_S'] = np.float64(self.params['UNITVELOCITY_IN_CM_PER_S'])
    if self.params['COMOVINGIC'] not in (0, 1):
        raise ValueError('Error: the COMOVINGIC parameter should be 1 or 0!\nExiting.')

def _process_icgen_parameters(self):
    '''
    Print all parameters related to the initial conditions generation.

    Parameters:
    -----------
    self.params : dict
        Dictionary containing the parameter list of a StePS simulation.
    '''

    # --- Validate IC Generator Type ---
    generator_map = {
        0: '2LPTic',
        1: 'NgenIC',
        2: 'L-genIC'
    }
    icgen_type = self.params.get('ICGENERATORTYPE')
    if icgen_type not in generator_map:
        raise ValueError(f"Error: unknown IC generator type: {icgen_type}\nExiting.")
    generator_str = generator_map[icgen_type]

    # --- Validate Executable ---
    if not os.path.exists(self.params['EXECUTABLE']):
        raise FileNotFoundError(f"Error: the executable '{self.params['EXECUTABLE']}' "\
                                 "does not exist.\nExiting.")

    # --- Validate Binning Mode ---
    bin_mode_map = {
        0: 'Constant size binning in the "omega" compact coordinate.',
        1: 'Constant shell volumes in the compact space.'
    }
    bin_mode = self.params.get('BIN_MODE')
    if bin_mode not in bin_mode_map:
        raise ValueError(f"Error: unknown binning mode {bin_mode}!\nExiting.")
    bin_mode_str = bin_mode_map[bin_mode]

    # --- Validate Output Format & Precision ---
    output_format_map = {
        0: ("ASCII",   "(N/A for ASCII)"),
        1: ("Gadget",  "(N/A for Gadget)"),
        2: ("HDF5",    None),  # This one requires a precision lookup
    }
    precision_map = {
        0: "32-bit",
        1: "64-bit"
    }
    output_format = self.params.get('OUTPUTFORMAT')
    if output_format not in output_format_map:
        raise ValueError(f"Error: unknown OUTPUTFORMAT value {output_format}!\nExiting.")
    output_format_str, fixed_precision_str = output_format_map[output_format]

    if fixed_precision_str is not None:
        # This format ignores OUTPUTPRECISION or uses a fixed notion
        output_precision_str = fixed_precision_str
    else:
        # For formats that require an actual precision check (e.g. HDF5)
        output_precision = self.params.get('OUTPUTPRECISION')
        if output_precision not in precision_map:
            raise ValueError(f"Error: unknown OUTPUTPRECISION value {output_precision}!\nExiting.")
        output_precision_str = precision_map[output_precision]

    # --- Phase Shift ---
    if self.params.get('PHASE_SHIFT_ENABLED', 0) == 1:
        phase_shift_str = f"{self.params['PHASE_SHIFT']:.2f} degrees"
    else:
        phase_shift_str = "Disabled"

    # --- Local Execution ---
    # The original code used 1 for local, 0 or 2 for remote.
    local_execution = self.params.get('LOCAL_EXECUTION')
    if local_execution == 1:
        execution_str = "local"
    elif local_execution in (0, 2):
        execution_str = "remote"
    else:
        raise ValueError(f"Error: unknown LOCAL_EXECUTION value {local_execution}!\nExiting.")

    text = dedent(f"""
    IC generator parameters:
    ------------------------
    IC generator:                   {generator_str}
    Executable:                     {self.params['EXECUTABLE']}
    Binning mode:                   {bin_mode_str}
    Output format:                  {output_format_str}
    Output precision:               {output_precision_str}
    Phase shift:                    {phase_shift_str}
    Execution mode:                 {execution_str}
    """)
    print(text)
