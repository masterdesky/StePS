#!/usr/bin/env python3

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
import sys
import time
import yaml
import numpy as np
from subprocess import call
from textwrap import dedent

import astropy.units as u
from astropy.cosmology import LambdaCDM, wCDM, w0waCDM, z_at_value

from pynverse import inversefunc

from write_ICparamfile import *
from inputoutput import *
from powerspec import *

_VERSION = 'v2.0'
_YEAR = '2017-2025'

#StePS internal units
UNIT_T=47.14829951063323      #Unit time in Gy
UNIT_V=20.738652969925447     #Unit velocity in km/s
UNIT_D=3.0856775814671917e24  #=1Mpc Unit distance in cm


#Basic function for the stereographic projection
def calculate_rlimits_i(i, d_s, N_r_bin, last_cell_size):
    r_i = d_s*np.tan((i)*np.pi/(2.0*(N_r_bin+last_cell_size)))
    return r_i
def calculate_rlimits_i_cvol(i, d_s, N_r_bin, R_sim):
    '''
    Calculates the lower limit of the i-th bin for the constant volume binning in the
    non-compact space (constant volume in the compact space)

    Parameters:
    -----------
    i : int
        The ID of the boundary
    d_s : float
        The diameter of the 4D sphere
    N_r_bin : int
        Number of the radial bins
    R_sim : float
        The radius of the simulation volume in real space
    '''
    omega_max = 2.0*np.arctan(R_sim/d_s)
    V_unit_bin = (2.0*omega_max-np.sin(2.0*omega_max))/N_r_bin
    V_unit_to_i = i*V_unit_bin
    #inverting numerically the x-sin(x) function
    func = lambda x: x-np.sin(x)
    omega_i = inversefunc(func, y_values=V_unit_to_i)/2.0
    r_i = d_s*np.tan(omega_i/2)
    return r_i
def calculate_r_i(r_func, i, d_s, N_r_bin, last_cell_size):
    ll = r_func(i, d_s, N_r_bin, last_cell_size)    # lower limit
    ul = r_func(i+1, d_s, N_r_bin, last_cell_size)  # upper limit
    #simple assumption with "conical frustum"
    r_i = 0.25 * (ul-ll) * (ll*ll + 2*ll*ul + 3*ul*ul) / (ll*ll + ll*ul + ul*ul) + ll
    return r_i

# Functions for cosmological perturbation theory
def zeldovich(x, Lbox, overdensity_field, growth_rate, h):
    '''
    Perform the Zel'dovich approximation to compute particle positions and velocities.
    
    Parameters:
    x (ndarray): Initial unperturbed particle positions (N, 3).
    overdensity_field (ndarray): Target overdensity field (real field).
    growth_rate (float): Time derivative of the growth factor D(t) in km/s/Mpc units.
    
    Returns:
    positions (ndarray): Updated particle positions (N, 3).
    velocities (ndarray): Particle velocities (N, 3).
    '''
    # TODO: Köbös rácson elmozdulásmező
    # 3D rácspontok (ezek) között kiinterpolálom ezt a mezőt
    # Interpoláció választása CIC
    nres = overdensity_field.shape[0]
    kk = np.fft.fftfreq(nres) * 2*np.pi/Lbox * nres
    ks = np.fft.rfftfreq(nres) * 2*np.pi/Lbox * nres
    kvec = np.array(np.meshgrid(kk, kk, ks))
    kmod = np.sqrt(np.sum(kvec**2, axis=0))
    delta_k = np.fft.rfftn(overdensity_field)
    xpert = np.zeros(x.shape, dtype=np.float32)
    v = np.zeros(x.shape, dtype=np.float32)
    for i, xi in zip(range(3), ('x', 'y', 'z')):
        psi_i = np.zeros_like(kmod, dtype=complex)
        mask = kmod > 0.0
        psi_i[mask] = -1j * kvec[i, mask] / kmod[mask]**2 * delta_k[mask]
        disp_field = np.fft.irfftn(psi_i)
        max_disp = np.max(disp_field)
        print(f"Maximal {xi} displacement: {max_disp*1000} kpc/h; in units of mean particle separation: {max_disp*nres/Lbox}")
        xpert[i, ...] = x[i, ...] + disp_field
        v[i, ...] = disp_field*growth_rate
    #Periodic wrapping
    xpert = np.fmod(xpert+Lbox, Lbox)
    #Converting the velocities from km/s/h to km/s.
    v /= h 
    return xpert, v

def header(N1:int = 97, N2:int = 66):
    art = dedent(f'''
    |      _____ _       _____   _____    _____ _____
    |     / ____| |     |  __ \ / ____|  |_   _/ ____|
    |    | (___ | |_ ___| |__) | (___      | || |       _ __  _   _
    |     \___ \| __/ _ \  ___/ \___ \     | || |      | '_ \| | | |
    |     ____) | ||  __/ |     ____) |____| || |____ _| |_) | |_| |
    |    |_____/ \__\___|_|    |_______________\_____(_) .__/ \__, |
    |                                                  | |     __/ |
    |                                                  |_|    |___/
    | StePS_IC.py {_VERSION}
    |  (an IC generator python script for STEreographically Projected cosmological Simulations)
    ''')
    cop = dedent(f'''
    | Copyright (C) ({_YEAR}) Gabor Racz
    | \tJet Propulsion Laboratory, California Institute of Technology | Pasadena, CA, USA
    | \tDepartment of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary
    | \tDepartment of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA
    ''')
    war = dedent(f'''
    | StePS_IC.py comes with ABSOLUTELY NO WARRANTY.
    | This is free software, and you are welcome to redistribute it
    | under certain conditions. See the LICENSE file for details.
    ''')
    # Define horizontal borders: +-- ... --+
    b  = lambda N: f'+{"-"*(N-2)}+'
    # Converts multiline string to list of lines
    ls = lambda s: s.strip().expandtabs(4).splitlines()
    # Pad RHS of all lines with spaces to get them equally `N` chars wide
    T  = lambda s, N: '\n'.join([f"{l}{' '*(N-1-len(l))}|" for l in ls(s)])

    print(f'{b(N1)}\n{T(art, N1)}\n{b(N1)}\n{T(cop, N1)}\n{b(N1)}')
    print(f'\n{b(N2)}\n{T(war, N2)}\n{b(N2)}')

def process_cosmo_params(params):
    '''
    Print all cosmological parameters found in the parameter list.

    Parameters:
    -----------
    params : dict
        Dictionary containing the parameter list.
    '''
    params['OMMH2'] = params['OMEGAM'] * (params['H0']/100.0)**2
    params['OMCH2'] = (params['OMEGAM'] - params['OMEGAB']) * (params['H0']/100.0)**2
    params['OMK']   = 1.0 - params['OMEGAM'] - params['OMEGAL']
    params['OMBH2'] = params['OMEGAB'] * (params['H0']/100.0)**2
    params['SCALE'] = 1.0/(params['REDSHIFT']+1.0)

    # Print cosmological parameters
    text = dedent(f'''
    Cosmological Parameters:
    ------------------------
    Omega_m:            {params['OMEGAM']:.6f}      (Ommh2={params['OMMH2']:.6f}; Omch2={params['OMCH2']:.6f})
    Omega_lambda:       {params['OMEGAL']:.6f}
    Omega_k:            {params['OMK']:.6f}
    Omega_b:            {params['OMEGAB']:.6f}      (Ombh2={params['OMBH2']:.6f})
    H0:                 {params['H0']:.3f} km/s/Mpc
    Redshift:           {params['REDSHIFT']:.3f}    (a={params['SCALE']:.6f})
    Sigma8:             {params['SIGMA8']:.3f}
    Dark energy model:  {params['DARKENERGYMODEL']}
    ''')
    print(text)
    # Supplementary log messages and operations
    if params['DARKENERGYMODEL'] == 'Lambda':
        print('\n')
    elif params['USECAMBINPUTSPECTRUM'] == False:
        print('Error: For non-standard dark energy parametrization USECAMBINPUTSPECTRUM has to be set True!\nExiting.\n')
        sys.exit(2)
    elif params['DARKENERGYMODEL'] == 'w0':
        print(f'w = {params['DARKENERGYPARAMS'][0]:.3f}\n')
    elif params['DARKENERGYMODEL'] == 'CPL':
        print(f'w0 = {params['DARKENERGYPARAMS'][0]:.3f}\nwa = {params['DARKENERGYPARAMS'][1]:.3f}\n')
    else:
        print('Error: unkown dark energy parametrization!\nExiting.\n')
        sys.exit(2)
    params['INPUTSPECTRUM_UNITLENGTH_IN_CM'] = np.float64(params['INPUTSPECTRUM_UNITLENGTH_IN_CM'])
    return params

def process_ic_params(params):
    '''
    Print all initial condition parameters found in the parameter list.

    Parameters:
    -----------
    params : dict
        Dictionary containing the parameter list.
    '''

    # Print initial condition parameters
    text = dedent(f'''
    IC parameters:
    --------------
    Lbox:                           {params['LBOX']:.3f} Mpc
    Rsim:                           {params['RSIM']:.3f} Mpc
    VOI_x:                          {params['VOIX']:.3f} Mpc
    VOI_y:                          {params['VOIY']:.3f} Mpc
    VOI_z:                          {params['VOIZ']:.3f} Mpc
    Seed:                           {params['SEED']:d}
    Spheremode:                     {params['SPHEREMODE']:d}
    WhichSpectrum:                  {params['WHICHSPECTRUM']:d}
    FileWithInputSpectrum:          {params['FILEWITHINPUTSPECTRUM']}
    InputSpectrum_UnitLength_in_cm: {params['INPUTSPECTRUM_UNITLENGTH_IN_CM']:.3e}
    ReNormalizeInputSpectrum:       {params['RENORMALIZEINPUTSPECTRUM']:d}
    ShapeGamma:                     {params['SHAPEGAMMA']:.3f}
    PrimordialIndex:                {params['PRIMORDIALINDEX']:.3f}
    Ngrid samples:                  {params['NGRIDSAMPLES']:d}
    GlassFile:                      {params['GLASSFILE']}
    OutDir:                         {params['OUTDIR']}
    FileBase:                       {params['FILEBASE']}
    Comoving IC:                    {params['COMOVINGIC']}
    Number of MPI tasks:            {params['MPITASKS']:d}
    H0 independent units:           {params['HINDEPENDENTUNITS']:d}
    ''')
    print(text)
    # Supplementary log messages and operations
    params['UNITLENGTH_IN_CM'] = np.float64(params['UNITLENGTH_IN_CM'])
    params['UNITMASS_IN_G'] = np.float64(params['UNITMASS_IN_G'])
    params['UNITVELOCITY_IN_CM_PER_S'] = np.float64(params['UNITVELOCITY_IN_CM_PER_S'])
    if params['COMOVINGIC'] != 0 and params['COMOVINGIC'] != 1:
        print('Error: the COMOVINGIC parameter should be 1 or 0!\nExiting.\n')
        sys.exit(2)
    return params

def process_icgen_parameters(params):
    '''
    Print all parameters related to the initial conditions generation.

    Parameters:
    -----------
    params : dict
        Dictionary containing the parameter list.
    '''

    # --- Validate IC Generator Type ---
    generator_map = {
        0: '2LPTic',
        1: 'NgenIC',
        2: 'L-genIC'
    }
    icgen_type = params.get('ICGENERATORTYPE')
    if icgen_type not in generator_map:
        print(f"Error: unknown IC generator type: {icgen_type}\nExiting.")
        sys.exit(2)
    generator_str = generator_map[icgen_type]

    # --- Validate Executable ---
    if not os.path.exists(params['EXECUTABLE']):
        print(f"Error: the executable '{params['EXECUTABLE']}' does not exist.\nExiting.")
        sys.exit(2)

    # --- Validate Binning Mode ---
    bin_mode_map = {
        0: 'Constant size binning in the "omega" compact coordinate.',
        1: 'Constant shell volumes in the compact space.'
    }
    bin_mode = params.get('BIN_MODE')
    if bin_mode not in bin_mode_map:
        print(f"Error: unknown binning mode {bin_mode}!\nExiting.")
        sys.exit(2)
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
    output_format = params.get('OUTPUTFORMAT')
    if output_format not in output_format_map:
        print(f"Error: unknown OUTPUTFORMAT ({output_format})!\nExiting.")
        sys.exit(2)
    output_format_str, fixed_precision_str = output_format_map[output_format]

    if fixed_precision_str is not None:
        # This format ignores OUTPUTPRECISION or uses a fixed notion
        output_precision_str = fixed_precision_str
    else:
        # For formats that require an actual precision check (e.g. HDF5)
        output_precision = params.get('OUTPUTPRECISION')
        if output_precision not in precision_map:
            print(f"Error: unknown OUTPUTPRECISION ({output_precision}) for HDF5!\nExiting.")
            sys.exit(2)
        output_precision_str = precision_map[output_precision]

    # --- Phase Shift ---
    if params.get('PHASE_SHIFT_ENABLED', 0) == 1:
        phase_shift_str = f"{params['PHASE_SHIFT']:.2f} degrees"
    else:
        phase_shift_str = "Disabled"

    # --- Local Execution ---
    # The original code used 1 for local, 0 or 2 for remote.
    local_execution = params.get('LOCAL_EXECUTION')
    if local_execution == 1:
        execution_str = "local"
    elif local_execution in (0, 2):
        execution_str = "remote"
    else:
        print(f"Error: unknown LOCAL_EXECUTION value {local_execution}!\nExiting.")
        sys.exit(2)

    text = dedent(f"""
    IC generator parameters:
    ------------------------
    IC generator:                   {generator_str}
    Executable:                     {params['EXECUTABLE']}
    Binning mode:                   {bin_mode_str}
    Output format:                  {output_format_str}
    Output precision:               {output_precision_str}
    Phase shift:                    {phase_shift_str}
    Execution mode:                 {execution_str}
    """)
    print(text)
    return params

def generate_camb(params):
    '''
    Setting the initial power spectrum with CAMB.

    Parameters:
    -----------
    params : dict
        Dictionary containing the parameter list.
    '''
    params['RENORMALIZEINPUTSPECTRUM'] = 0
    
    print('Calculating input spectrum with CAMB...')
    kmin    = 1.0*np.pi/params['LBOX']
    kmax    = 100.0
    npoints = 2048
    kh, pk  = get_CAMB_Linear_SPECTRUM(
        H0=params['H0'], ombh2=params['OMBH2'], omch2=params['OMCH2'], omk=params['OMK'],
        ns=params['PRIMORDIALINDEX'], redshift=params['REDSHIFT'],
        kmin=kmin, kmax=kmax, npoints=npoints, sigma8=params['SIGMA8'],
        DE=params['DARKENERGYMODEL'], DE_params=params['DARKENERGYPARAMS'])
    Pk3 = np.vstack((np.log10(kh), np.log10(pk*kh**3/(2*np.pi**2)))).T
    np.savetxt(params['FILEWITHINPUTSPECTRUM'], Pk3)
    print('...done')
    return params


def main():
    start = time.time()
    header(N1=97, N2=66)
    # Reading in input parameterfile in yaml format
    if len(sys.argv) != 2:
        print('Error: missing yaml file!')
        print('Usage: ./StePS_IC.py <input yaml file>\nExiting.')
        sys.exit(2)
    with open(sys.argv[1], 'r') as f:
        print(f'Reading the {sys.argv[1]} paramfile...\n')
        params = yaml.safe_load(f)
    # Processing simulation parameters
    process_cosmo_params(params)
    process_ic_params(params)
    process_icgen_parameters(params)
    # Calculating the density from the cosmological parameters
    params['RHO_CRIT'] = 3*params['H0']**2/(8*np.pi)/UNIT_V/UNIT_V
    params['RHO_MEAN'] = params['OMEGAM']*params['RHO_CRIT']
    # Generating the initial power spectrum with CAMB
    if params['USECAMBINPUTSPECTRUM']:
        generate_camb(params)
    # Loading the input initial condition, which will potentially be
    # a cosmological glass, created by another cosmological IC generator
    _, GC, _, GM = load_snapshot(params['GLASSFILE'])
    input_glass = np.vstack((np.hstack((GC, np.zeros_like(GC, dtype=np.float64)).T, GM))).T
    # End of the script
    print(f'The IC building took {(time.time() - start):.4f} s.')

if __name__ == "__main__":
    main()




#Loading the input glass:
print("Loading the %s input glass file..." % params['GLASSFILE'])
glasscoords,glassmasses = Load_snapshot(params['GLASSFILE'])
input_glass = np.vstack((np.hstack((glasscoords,np.zeros(glasscoords.shape,dtype=np.double))).T,glassmasses)).T
Npart = len(input_glass)
del(glasscoords)
del(glassmasses)
print("...done.")
#Calculating the total mass, and the average density
M_tot = np.sum(input_glass[:,6])
V_sim = 4.0*np.pi/3.0*params['RSIM']**3
OmegaM_mean_input = (M_tot/V_sim)/rho_crit
if np.absolute(OmegaM_mean_input/params['OMEGAM']-1.0) < 1e-9:
    print("\nThe cosmological Omega_m parameter, calculated from the particle masses:\tOmega_m=%f\n" % (OmegaM_mean_input))
else:
    input_glass[:,6] = input_glass[:,6] * params['OMEGAM'] / OmegaM_mean_input
    print("\nThe particle masses were rescaled to fit with the cosmological parameter Omega_m=%f\n" % (params['OMEGAM']))
original_glass = np.copy(input_glass)
#Calculating the Mass list:
Mass_list = np.unique(input_glass[:,6])
print("Number of different masses:\t%i\n" % len(Mass_list))
#Periodically shifting the input glass:
print("Periodically shifting the input glass...")
input_glass[:,0] = input_glass[:,0]+params['VOIX']
input_glass[:,1] = input_glass[:,1]+params['VOIY']
input_glass[:,2] = input_glass[:,2]+params['VOIZ']
for i in range(0, Npart):
    for k in range(0,3):
        if input_glass[i,k]<0:
            input_glass[i,k] += params['LBOX']
        if input_glass[i,k]>params['LBOX']:
            input_glass[i,k] -= params['LBOX']
print("...done.\n")
#Converting the input glass to gadget format:
if params['LOCAL_EXECUTION'] < 2:
    print("Converting the input glass to Gadget format...")
    output_glassfile = params['OUTDIR'] + params['FILEBASE'] + "_Glass_tmp.dat"
    np.savetxt(output_glassfile, input_glass, delimiter='\t')
    gadget_glassfile = params['OUTDIR'] + params['FILEBASE'] + "_GLASS"
    ascii2gadget(output_glassfile, gadget_glassfile, params['LBOX'], params['H0'], params['UNITLENGTH_IN_CM'])
    call(["rm", "-f", output_glassfile])
    print("...done.\n")
M_tot_box = rho_mean*params['LBOX']**3
if params['NMESH'] == 0:
    #Calculating the Nsample-Mass function:
    Nsample_func = np.zeros((len(Mass_list),2))
    Nsample_func[:,0] = Mass_list[:]
    Nsample_func[:,1] = np.uint32(np.cbrt(M_tot_box/Mass_list[:]))
    Nsample_tab = np.zeros(params['NGRIDSAMPLES'],dtype=np.uint32)
    Mass_tab = np.zeros(params['NGRIDSAMPLES'],dtype=np.uint32)
    delta_Nsample = np.uint32(np.ceil(len(Mass_list)/params['NGRIDSAMPLES']))
    print("The generated Nsample list:")
    print("ID\tNsample\tMass(in 10e11Msol)")
    Nsample_tab[len(Nsample_tab)-1] = Nsample_func[0,1]
    print("%i\t%i\t%e" % (len(Nsample_tab)-1, Nsample_tab[len(Nsample_tab)-1], Nsample_func[0,0]))
    for i in range(len(Nsample_tab)-2, -1, -1):
        Nsample_tab[i] = Nsample_func[len(Nsample_func)-1-i*delta_Nsample,1]
        Mass_tab[i] = Nsample_func[len(Nsample_func)-1-i*delta_Nsample,0]
        print("%i\t%i\t%e" % (i, Nsample_tab[i], Mass_tab[i]))
    #generating paramfiles
    paramfile_name=len(Nsample_tab)*[None]
    if params['LOCAL_EXECUTION'] < 2:
        for i in range(0,len(Nsample_tab)):
            paramfile_name[i] = params['OUTDIR'] + params['FILEBASE'] + "_%i" % i + ".param"
            if params['ICGENERATORTYPE'] == 0:
                Write_2LPTic_paramfile(paramfile_name[i], Nsample_tab[i], Nsample_tab[i], params['LBOX']*UNIT_D/params['UNITLENGTH_IN_CM']*(params['H0']/100.0), params['FILEBASE'] + "_%i" % i, params['OUTDIR'], gadget_glassfile, params['OMEGAM'], params['OMEGAL'], params['OMEGAB'], params['H0']/100.0, params['REDSHIFT'], params['SIGMA8'], params['SPHEREMODE'], params['WHICHSPECTRUM'], params['FILEWITHINPUTSPECTRUM'], params['SHAPEGAMMA'], params['PRIMORDIALINDEX'], params['SEED'],params['UNITLENGTH_IN_CM'],params['UNITMASS_IN_G'],params['UNITVELOCITY_IN_CM_PER_S'],params['INPUTSPECTRUM_UNITLENGTH_IN_CM'],params['PHASE_SHIFT_ENABLED'],params['PHASE_SHIFT'], params['FIXED_AMPLITUDES_ENABLED'], params['FIXED_AMPLITUDES'], renormalizeinputspectrum)
            elif params['ICGENERATORTYPE'] == 1:
                Write_NgenIC_paramfile(paramfile_name[i], Nsample_tab[i], Nsample_tab[i], params['LBOX']*UNIT_D/params['UNITLENGTH_IN_CM']*(params['H0']/100.0), params['FILEBASE'] + "_%i" % i, params['OUTDIR'], gadget_glassfile, params['OMEGAM'], params['OMEGAL'], params['OMEGAB'], params['H0']/100.0, params['REDSHIFT'], params['SIGMA8'], params['SPHEREMODE'], params['WHICHSPECTRUM'], params['FILEWITHINPUTSPECTRUM'], renormalizeinputspectrum, params['SHAPEGAMMA'], params['PRIMORDIALINDEX'], params['SEED'],params['UNITLENGTH_IN_CM'],params['UNITMASS_IN_G'],params['UNITVELOCITY_IN_CM_PER_S'],params['INPUTSPECTRUM_UNITLENGTH_IN_CM'],params['PHASE_SHIFT_ENABLED'],params['PHASE_SHIFT'], params['FIXED_AMPLITUDES_ENABLED'], params['FIXED_AMPLITUDES'])
            elif params['ICGENERATORTYPE'] == 2:
                Write_LgenIC_paramfile(paramfile_name[i], Nsample_tab[i], Nsample_tab[i], params['LBOX']*UNIT_D/params['UNITLENGTH_IN_CM']*(params['H0']/100.0), params['FILEBASE'] + "_%i" % i, params['OUTDIR'], gadget_glassfile, params['OMEGAM'], params['OMEGAL'], params['OMEGAB'], params['H0']/100.0, params['REDSHIFT'], params['SIGMA8'], params['SPHEREMODE'], params['WHICHSPECTRUM'], params['FILEWITHINPUTSPECTRUM'], params['SHAPEGAMMA'], params['PRIMORDIALINDEX'], params['SEED'],params['UNITLENGTH_IN_CM'],params['UNITMASS_IN_G'],params['UNITVELOCITY_IN_CM_PER_S'],params['INPUTSPECTRUM_UNITLENGTH_IN_CM'],params['PHASE_SHIFT_ENABLED'],params['PHASE_SHIFT'], params['FIXED_AMPLITUDES_ENABLED'], params['FIXED_AMPLITUDES'])
            else:
                print("Error: unkown IC generator!\nExiting.\n")
                sys.exit(2)
    if params['LOCAL_EXECUTION'] == 2:
        for i in range(0,len(Nsample_tab)):
            paramfile_name[i] = params['OUTDIR'] + params['FILEBASE'] + "_%i" % i + ".param"
    #generating the ICs
    if params['LOCAL_EXECUTION'] == 1:
        for i in range(0,len(Nsample_tab)):
            print("\n--------------------------------\nExecuting:\n " + "mpirun " + params['EXECUTABLE'] + " " + paramfile_name[i] + "\nNsample=%i\n\n" % Nsample_tab[i])
            print("\t(Estimated memory equirement:\t%.3fGb)\n" % ((16*Nsample_tab[i]**3+Npart*8*6)/1024**3))
            call(["mpirun", "-n", str(params['MPITASKS']), params['EXECUTABLE'], paramfile_name[i]])
    if params['LOCAL_EXECUTION'] == 0:
        for i in range(0,len(Nsample_tab)):
            print("\nCall the IC generator by:\n\t$" + "mpirun " + "-np " + str(params['MPITASKS']) + " " +str(params['EXECUTABLE']) + " " + str(paramfile_name[i]))
            print("\t(Estimated memory equirement:\t%.3fGb)\n" % ((16*Nsample_tab[i]**3+Npart*8*6)/1024**3))
        print("Then restart this script with LOCAL_EXECUTION option set to 2.\nExiting...")
        exit()
    #Calculating the displacement field for every NSAMPLE:
    print("Calculating the displacement and velocity field for every Nsample...")
    Disp_field = np.zeros( ( params['NGRIDSAMPLES'], Npart, 3), dtype=np.float32)
    Vel_field = np.zeros( ( params['NGRIDSAMPLES'], Npart, 3), dtype=np.float32)
    for i in range(0,len(Nsample_tab)):
        print("    i=%i\tNsample=%i" % (i,Nsample_tab[i]))
        X_tmp = np.zeros( (Npart,3), dtype=np.float32)
        V_tmp = np.zeros( (Npart,3), dtype=np.float32)
        if params['MPITASKS'] == 1:
            #reading only 1 gadget file
            if params['ICGENERATORTYPE'] == 2:
                filename = params['OUTDIR'] + params['FILEBASE'] + "_%i" % i + ".0"
            else:
                filename = params['OUTDIR'] + params['FILEBASE'] + "_%i" % i
            if exists(filename):
                print("    Loading the " + filename + " file...")
                snapshot = glio.GadgetSnapshot(filename)
                snapshot.load()
                X_tmp = snapshot.pos[1] / (params['H0'] / 100.0) * params['UNITLENGTH_IN_CM']/UNIT_D
                V_tmp = snapshot.vel[1]
            else:
                print("Error: the " + filename + " file does not exist. Exiting...")
                exit(-3)
        else:
            #reading multiple gadget file
            for j in range(0,params['MPITASKS']):
                filename = params['OUTDIR'] + params['FILEBASE'] + "_%i" % i + ".%i" % j
                if exists(filename):
                    print("    Loading the " + filename + " file...")
                    snapshot = glio.GadgetSnapshot(filename)
                    snapshot.load()
                    N_in_this_file=snapshot.header.npart[1]
                    for k in range(0,N_in_this_file):
                        #The IDs are shifted with 1
                        index_of_this_particle=snapshot.ID[1][k]-1
                        X_tmp[index_of_this_particle] = snapshot.pos[1][k] / (params['H0'] / 100.0) * params['UNITLENGTH_IN_CM']/UNIT_D
                        V_tmp[index_of_this_particle] = snapshot.vel[1][k]
                else:
                    print("Error: the " + filename + " file does not exist. Exiting...")
                    exit(-3)

        print("    ...done.\n    Calculating the displacement field...")
        Disp_field[i,:,:] = X_tmp-input_glass[:,0:3]
        for j in range(0,Npart):
            for k in range(0,3):
                if np.absolute(Disp_field[i,j,k]) >= params['LBOX']/2.0:
                    if Disp_field[i,j,k]>0:
                        Disp_field[i,j,k] -= params['LBOX']
                    else:
                        Disp_field[i,j,k] += params['LBOX']
        print("    Average displacement: %f Mpc" %  np.mean(np.sqrt(Disp_field[i,:,0]**2 + Disp_field[i,:,1]**2 + Disp_field[i,:,2]**2)))
        print("    Maximal displacement: %f Mpc" % np.max(np.sqrt(Disp_field[i,:,0]**2 + Disp_field[i,:,1]**2 + Disp_field[i,:,2]**2)))
        print("    ...done.\n    Calculating the velocity field...")
        Vel_field[i,:,:] = V_tmp #saving in km/s
        print("    Average velocity: %f km/s" % (np.mean(np.sqrt(Vel_field[i,:,0]**2 + Vel_field[i,:,1]**2 + Vel_field[i,:,2]**2))))
        print("    ...done.\n")
    print("...done\n")
    del(V_tmp)
    del(X_tmp)
    print("Interpolating between the different Nsamples and generating the final IC...")
    IC = np.zeros((Npart,7))
    IC[:,6] = original_glass[:,6] #masses
    for i in range(0,Npart):
        for k in range(0,3):
            #interpolation in the coordinate-space
            IC[i,k] = original_glass[i,k] + np.interp(IC[i,6],Mass_tab,Disp_field[:,i,k])
            #interpolation in the velocity-space
            IC[i,k+3] = original_glass[i,k+3] + np.interp(IC[i,6],Mass_tab,Vel_field[:,i,k])
    print("...done\n")
else:
    #In this case, the script only generates one displacement field
    if params['LOCAL_EXECUTION'] < 2:
        paramfile_name = params['OUTDIR'] + params['FILEBASE'] + ".param"
        Nsample = np.uint32(np.ceil(np.cbrt(M_tot_box/np.min(Mass_list))))
        if Nsample > params['NMESH']:
            print("Warning: Nsample (=%i) > Nmesh (=%i). Setting Nsample to %i." % (Nsample, params['NMESH'], params['NMESH']))
            Nsample = params['NMESH']
        if params['ICGENERATORTYPE'] == 0:
            Write_2LPTic_paramfile(paramfile_name, params['NMESH'], Nsample, params['LBOX']*UNIT_D/params['UNITLENGTH_IN_CM']*(params['H0']/100.0), params['FILEBASE'], params['OUTDIR'], gadget_glassfile, params['OMEGAM'], params['OMEGAL'], params['OMEGAB'], params['H0']/100.0, params['REDSHIFT'], params['SIGMA8'], params['SPHEREMODE'], params['WHICHSPECTRUM'], params['FILEWITHINPUTSPECTRUM'], params['SHAPEGAMMA'], params['PRIMORDIALINDEX'], params['SEED'],params['UNITLENGTH_IN_CM'],params['UNITMASS_IN_G'],params['UNITVELOCITY_IN_CM_PER_S'],params['INPUTSPECTRUM_UNITLENGTH_IN_CM'], params['PHASE_SHIFT_ENABLED'], params['PHASE_SHIFT'], params['FIXED_AMPLITUDES_ENABLED'], params['FIXED_AMPLITUDES'], renormalizeinputspectrum)
        elif params['ICGENERATORTYPE'] == 1:
            Write_NgenIC_paramfile(paramfile_name, params['NMESH'], Nsample, params['LBOX']*UNIT_D/params['UNITLENGTH_IN_CM']*(params['H0']/100.0), params['FILEBASE'], params['OUTDIR'], gadget_glassfile, params['OMEGAM'], params['OMEGAL'], params['OMEGAB'], params['H0']/100.0, params['REDSHIFT'], params['SIGMA8'], params['SPHEREMODE'], params['WHICHSPECTRUM'], params['FILEWITHINPUTSPECTRUM'], renormalizeinputspectrum, params['SHAPEGAMMA'], params['PRIMORDIALINDEX'], params['SEED'],params['UNITLENGTH_IN_CM'],params['UNITMASS_IN_G'],params['UNITVELOCITY_IN_CM_PER_S'],params['INPUTSPECTRUM_UNITLENGTH_IN_CM'],params['PHASE_SHIFT_ENABLED'],params['PHASE_SHIFT'], params['FIXED_AMPLITUDES_ENABLED'], params['FIXED_AMPLITUDES'])
        elif params['ICGENERATORTYPE'] == 2:
            Write_LgenIC_paramfile(paramfile_name, params['NMESH'], Nsample, params['LBOX']*UNIT_D/params['UNITLENGTH_IN_CM']*(params['H0']/100.0), params['FILEBASE'], params['OUTDIR'], gadget_glassfile, params['OMEGAM'], params['OMEGAL'], params['OMEGAB'], params['H0']/100.0, params['REDSHIFT'], params['SIGMA8'], params['SPHEREMODE'], params['WHICHSPECTRUM'], params['FILEWITHINPUTSPECTRUM'], params['SHAPEGAMMA'], params['PRIMORDIALINDEX'], params['SEED'],params['UNITLENGTH_IN_CM'],params['UNITMASS_IN_G'],params['UNITVELOCITY_IN_CM_PER_S'],params['INPUTSPECTRUM_UNITLENGTH_IN_CM'],params['PHASE_SHIFT_ENABLED'],params['PHASE_SHIFT'], params['FIXED_AMPLITUDES_ENABLED'], params['FIXED_AMPLITUDES'])
        else:
            print("Error: unkown IC generator!\nExiting.\n")
            sys.exit(2)
    if params['LOCAL_EXECUTION'] == 2:
        paramfile_name = params['OUTDIR'] + params['FILEBASE'] + ".param"
    if params['LOCAL_EXECUTION'] == 1:
        print("\n--------------------------------\nExecuting:\n " + "mpirun " + params['EXECUTABLE'] + " " + paramfile_name + "\nNsample=%i\n\n" % Nsample)
        print("(Estimated memory equirement:\t%.3fGb)" % ((16*Nsample**3+Npart*8*6)/1024**3))
        call(["mpirun", "-np", str(params['MPITASKS']), params['EXECUTABLE'], paramfile_name])
    if params['LOCAL_EXECUTION'] == 0:
        print("\nCall the IC generator by:\n\t$" + "mpirun " + "-np " + str(params['MPITASKS']) + " " +str(params['EXECUTABLE']) + " " + str(paramfile_name))
        print("(Estimated memory equirement:\t%.3fGb)" % ((16*Nsample**3+Npart*8*6)/1024**3))
        print("Then restart this script with LOCAL_EXECUTION option set to 2.\nExiting...")
        exit()
    #Calculating the displacement field:
    print("Calculating the displacement and velocity field...")
    Disp_field = np.zeros( (Npart, 3), dtype=np.float32)
    Vel_field = np.zeros( (Npart, 3), dtype=np.float32)
    if params['MPITASKS'] == 1:
        #reading only 1 gadget file
        if params['ICGENERATORTYPE'] == 2:
            print("    Loading the " + params['OUTDIR'] + params['FILEBASE'] + ".0" + " file...")
            snapshot = glio.GadgetSnapshot(params['OUTDIR'] + params['FILEBASE'] + ".0")
        else:
            print("    Loading the " + params['OUTDIR'] + params['FILEBASE'] + " file...")
            snapshot = glio.GadgetSnapshot(params['OUTDIR'] + params['FILEBASE'])
            snapshot.load()
            Disp_field = snapshot.pos[1] / (params['H0'] / 100.0) * params['UNITLENGTH_IN_CM']/UNIT_D
            Vel_field = snapshot.vel[1]
    else:
        #reading multiple gadget file
        for j in range(0,params['MPITASKS']):
            filename = params['OUTDIR'] + params['FILEBASE'] + ".%i" % j
            if exists(filename):
                print("    Loading the " + filename + " file...")
                snapshot = glio.GadgetSnapshot(filename)
                snapshot.load()
                N_in_this_file=snapshot.header.npart[1]
                for k in range(0,N_in_this_file):
                    #The IDs are shifted with 1
                    index_of_this_particle=snapshot.ID[1][k]-1
                    Disp_field[index_of_this_particle] = snapshot.pos[1][k] / (params['H0'] / 100.0) * params['UNITLENGTH_IN_CM']/UNIT_D
                    Vel_field[index_of_this_particle] = snapshot.vel[1][k]
    print("    ...done.\n    Calculating the displacement field...")
    Disp_field[:,0:3] = Disp_field[:,0:3]-input_glass[:,0:3]
    for j in range(0,Npart):
        for k in range(0,3):
            if np.absolute(Disp_field[j,k]) >= params['LBOX']/2.0:
                if Disp_field[j,k]>0:
                    Disp_field[j,k] -= params['LBOX']
                else:
                    Disp_field[j,k] += params['LBOX']
    print("    Average displacement: %f Mpc" %  np.mean(np.sqrt(Disp_field[:,0]**2 + Disp_field[:,1]**2 + Disp_field[:,2]**2)))
    print("    Maximal displacement: %f Mpc" % np.max(np.sqrt(Disp_field[:,0]**2 + Disp_field[:,1]**2 + Disp_field[:,2]**2)))
    print("    Average velocity: %f km/s" % (np.mean(np.sqrt(Vel_field[:,0]**2 + Vel_field[:,1]**2 + Vel_field[:,2]**2))))
    print("...done.\n")
    IC = np.zeros((Npart,7))
    IC[:,6] = original_glass[:,6] #masses
    IC[:,0:3] = original_glass[:,0:3] + Disp_field[:,:] #coordinates
    IC[:,3:6] = Vel_field[:,:] #velocities
    print("...done\n")

if params['COMOVINGIC'] == 0:
    print("Rescaling the IC and adding the Hubble flow for non-comoving simulation...")
    a_start = 1.0/(params['REDSHIFT']+1.0)
    Hubble_start = params['H0']*np.sqrt(np.power(a_start, -3.0)*params['OMEGAM'] + params['OMEGAL'] + np.power(a_start, -2.0)*(1-params['OMEGAM']-params['OMEGAL']))
    print("Initial Hubble parameter: %f km/s/Mpc" % Hubble_start)
    for i in range(0,Npart):
        for k in range(0,3):
            IC[i,k] = IC[i,k] * a_start
            IC[i,k+3] = IC[i,k+3] * np.sqrt(a_start)
            IC[i,k+3] = IC[i,k+3] + IC[i,k]*Hubble_start
    print("...done\n")

if params['HINDEPENDENTUNITS'] == 1:
    print("Converting the IC to H0 independent units...")
    #converting the IC to /h units
    h = params['H0'] / 100.0
    #coordinates
    IC[:,0:3] *= h
    #masses
    IC[:,6] *= h
    print("...done\n")


if params['OUTPUTFORMAT'] == 0:
    outputfilename = params['OUTDIR'] + params['FILEBASE'] + ".dat"
if params['OUTPUTFORMAT'] == 2:
    outputfilename = params['OUTDIR'] + params['FILEBASE'] + ".hdf5"
print("Saving %s IC file..." % outputfilename)
if params['OUTPUTFORMAT'] == 0:
    np.savetxt(outputfilename, IC, delimiter='\t')
if params['OUTPUTFORMAT'] == 2:
    writeHDF5snapshot(IC, outputfilename, np.double(2.0*params['RSIM']), params['REDSHIFT'], params['OMEGAM'], params['OMEGAL'], params['H0']/100.0, params['OUTPUTPRECISION'])
print("...done\n")
print("Calculating redshifts for the spherical shells...")
#calculating the comoving distances of the particles
shell_limits = np.zeros(np.uint64(params['NRBINS'])+np.uint64(1), dtype=np.float64)
z_list = np.zeros(np.uint64(params['NRBINS']), dtype=np.float64)
if params['BIN_MODE'] == 0:
    last_cell_size = params['NRBINS']*np.pi/(2*np.arctan(params['RSIM']/params['D_S']))-params['NRBINS']
i = np.arange(params['NRBINS'])
if params['BIN_MODE'] == 0:
    r_list = calculate_r_i(calculate_rlimits_i, i, params['D_S'], params['NRBINS'], last_cell_size)
    if params['HINDEPENDENTUNITS'] == 1:
        r_list *= h
if params['BIN_MODE'] == 1:
    r_list = calculate_r_i(calculate_rlimits_i_cvol, i, params['D_S'], params['NRBINS'], params['RSIM'])
    if params['HINDEPENDENTUNITS'] == 1:
        r_list *= h
del(i)
i = np.arange(params['NRBINS']+1)
if params['BIN_MODE'] == 0:
    shell_limits = calculate_rlimits_i(i, params['D_S'], params['NRBINS'], last_cell_size)
    if params['HINDEPENDENTUNITS'] == 1:
        shell_limits *= h
if params['BIN_MODE'] == 1:
    shell_limits = calculate_rlimits_i_cvol(i, params['D_S'], params['NRBINS'], params['RSIM'])
    if params['HINDEPENDENTUNITS'] == 1:
        shell_limits *= h
#calculating redshift-comoving distance function for the redshift cone
if params['DARKENERGYMODEL'] == 'Lambda':
    #LCDM model
    cosmo = LambdaCDM(H0=params['H0'], Om0=params['OMEGAM'], Ode0=params['OMEGAL'])
elif params['DARKENERGYMODEL'] == 'w0':
    #wCDM model
    cosmo = wCDM(H0=params['H0'], Om0=params['OMEGAM'], Ode0=params['OMEGAL'],w0=params['DARKENERGYPARAMS'][0])
elif params['DARKENERGYMODEL'] == 'CPL':
    #w0waCDM model
    cosmo = w0waCDM(H0=params['H0'], Om0=params['OMEGAM'], Ode0=params['OMEGAL'],w0=params['DARKENERGYPARAMS'][0],wa=params['DARKENERGYPARAMS'][1])
for i in range(0,len(z_list)):
    z_list[i] = z_at_value(cosmo.comoving_distance, r_list[i]*u.Mpc)
if params['OUTPUTFORMAT'] == 0:
    outputfilename = params['OUTDIR'] + params['FILEBASE'] + ".dat_zbins"
if params['OUTPUTFORMAT'] == 2:
    outputfilename = params['OUTDIR'] + params['FILEBASE'] + ".hdf5_zbins"
np.savetxt(outputfilename, z_list)
if params['OUTPUTFORMAT'] == 0:
    outputfilename = params['OUTDIR'] + params['FILEBASE'] + ".dat_zbins_rlimits"
if params['OUTPUTFORMAT'] == 2:
    outputfilename = params['OUTDIR'] + params['FILEBASE'] + ".hdf5_zbins_rlimits"
np.savetxt(outputfilename, shell_limits)
print("...done\n")
print("Deleting the temporary files...")

if params['LOCAL_EXECUTION'] > 0:
    if params['NMESH'] == 0:
        for i in range(0,len(Nsample_tab)):
            call(["rm", "-f", paramfile_name[i]])
            if params['MPITASKS'] == 1:
                if params['ICGENERATORTYPE'] == 2:
                    call(["rm", "-f", (params['OUTDIR'] + params['FILEBASE'] + "_%i" % i + ".0")])
                    call(["rm", "-f", (params['OUTDIR'] + "inputspec_" + params['FILEBASE'] + "_%i" % i + ".txt")])
                else:
                    call(["rm", "-f", (params['OUTDIR'] + params['FILEBASE'] + "_%i" % i)])
                    call(["rm", "-f", (params['OUTDIR'] + "inputspec_" + params['FILEBASE'] + "_%i" % i + ".txt")])
            else:
                for j in range(0,params['MPITASKS']):
                    call(["rm", "-f", (params['OUTDIR'] + params['FILEBASE'] + "_%i" % i + ".%i" % j)])
                    call(["rm", "-f", (params['OUTDIR'] + "inputspec_" + params['FILEBASE'] + "_%i" % i + ".txt")])
    else:
        call(["rm", "-f", paramfile_name])
        if params['MPITASKS'] == 1:
            if params['ICGENERATORTYPE'] == 2:
                call(["rm", "-f", (params['OUTDIR'] + params['FILEBASE'] + ".0")])
                call(["rm", "-f", (params['OUTDIR'] + "inputspec_" + params['FILEBASE'] + ".txt")])
            else:
                call(["rm", "-f", (params['OUTDIR'] + params['FILEBASE'])])
                call(["rm", "-f", (params['OUTDIR'] + "inputspec_" + params['FILEBASE'] + ".txt")])
        else:
            for j in range(0,params['MPITASKS']):
                call(["rm", "-f", (params['OUTDIR'] + params['FILEBASE'] + ".%i" % j)])
                call(["rm", "-f", (params['OUTDIR'] + "inputspec_" + params['FILEBASE'] + ".txt")])
call(["rm", "-f", (params['OUTDIR'] + params['FILEBASE'] + "_GLASS")])
print("...done.\n")

