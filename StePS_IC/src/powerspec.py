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

import camb
from colossus.cosmology import cosmology


def linear_growth_function(z, H0, omega_m, omega_b, omega_l, s8, ns, DE, DE_params,
                           silent=True):
    '''
    Calculate the linear growth factor D(a) normalized to 1 at a=1.

    For w0 and CPL dark energy, we use the Colossus implementation of
    Eq. (11) from Linder & Jenkins (2003).
    See paper at https://arxiv.org/pdf/astro-ph/0305286.pdf.

    Returns:
    --------
    Dlin : float
        The linear growth factor at a=1.
    '''
    # Calculating the curvature
    omega_k = 1.0 - omega_m - omega_l
    if np.abs(omega_k) <= 1e-5:
        flat = True
        print('Flat cosmology.')
    else:
        flat = False
        print(f'Non-flat cosmology; {omega_k =}, {omega_m =:.4f}, {omega_l =:.4f}')
    # Zero CMB temperature can cause issues in colossus. This small value
    # shouldn't cause any significant errors at late times in relevant
    # cosmologies. (Non-zero `omega_r` will be implemented in the future.)
    T_cmb = 0.001
    
    params = {  # Common cosmological parameters for all models
        'flat': flat, 'H0': H0, 'Om0': omega_m, 'Ob0': omega_b,
        'sigma8': s8, 'ns': ns, 'Tcmb0': T_cmb
    }
    if DE == 'Lambda':
        if not flat:
            params.update({'Ode0': omega_l})
        cosmo = cosmology.setCosmology('LCDM', **params)
    elif DE == 'w0':
        params.update({'de_model': 'w0', 'w0': DE_params[0]})
        if not flat:
            params.update({'Ode0': omega_l})
        cosmo = cosmology.setCosmology('wCDM', **params)
    elif DE == 'CPL':
        params.update({'de_model': 'w0wa', 'w0': DE_params[0], 'wa': DE_params[1]})
        if not flat:
            params.update({'Ode0': omega_l})
        cosmo = cosmology.setCosmology('w0waCDM', **params)
    Dlin = cosmo.growthFactorUnnormalized(z) / cosmo.growthFactorUnnormalized(0.0)
    if not silent:
        print(f'Initial normalized linear growth: D(z={z:.2f})/D(z=0) = {Dlin:.2e}')
    return Dlin

def linear_spectrum_camb(
        z=127, H0=73.0, ombh2=0.024, omch2=0.1092445, omega_k=0.0, ns=1.0,
        kmin=0.01, kmax=1.0, npoints=1024, sigma8=None, DE='Lambda', DE_params=None):
    '''
    Calculate the linear matter power spectrum using CAMB. The output is
    normalized to the linear growth factor at z=0. The default cosmological
    parameters are from the Planck 2018 results.

    Parameters:
    -----------
    z : float
        Redshift at which the linear power spectrum is calculated.
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter.
    omch2 : float
        Physical cold dark matter density parameter.
    omega_k : float
        Curvature density parameter.
    ns : float
        Scalar spectral index.
    kmin : float
        Minimum wavenumber in h/Mpc.
    kmax : float
        Maximum wavenumber in h/Mpc.
    npoints : int
        Number of wavenumber points.
    sigma8 : float; default=None
        Rescale the linear power spectrum to the given sigma8 value.
    DE : str
        Dark energy model. Options are 'Lambda', 'w0', 'CPL'.
    DE_params : list or array-like
        Dark energy model parameters. For 'Lambda' and 'w0', it is a single
        element list containing the dark energy equation of state parameter w.
        For 'CPL', it is a two element list containing w and wa.
    '''
    params = camb.CAMBparams()
    params.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omega_k)
    if DE == 'w0':
        params.DarkEnergy = camb.dark_energy.DarkEnergyFluid()
        params.DarkEnergy.set_params(w=DE_params[0])
    elif DE == 'CPL':
        params.DarkEnergy = camb.dark_energy.DarkEnergyFluid()
        params.DarkEnergy.set_params(w=DE_params[0], wa=DE_params[1])
    
    # Calculate sigma_8 at z=0
    params.NonLinear = camb.model.NonLinear_none
    params.InitPower.set_params(ns=ns)
    params.set_matter_power(redshifts=[0.0], kmax=kmax)
    results = camb.get_results(params)
    s8 = results.get_sigma8()
    print(f'Original sigma_8: {s8}')

    # Calculating normalized linear growth function
    omega_m = (ombh2+omch2) / ((H0/100.0)**2)  # Non-relativistic matter density
    omega_b = ombh2 / ((H0/100.0)**2)          # Baryonic matter density
    omega_l = 1.0 - omega_k - omega_m          # Dark energy density
    DzD0 = linear_growth_function(
                z, H0, omega_m, omega_b, omega_l, s8, ns, DE, DE_params)
    
    # Calculating the linear P(k) at z=0
    if sigma8 is None:
        params.InitPower.set_params(
                As=2e-09*(sigma8/s8)**2, ns=ns, nrun=0, nrunrun=0.0, r=0.0,
                nt=None, ntrun=0.0, pivot_scalar=0.05, pivot_tensor=0.05,
                parameterization='tensor_param_rpivot')
        params.set_matter_power(redshifts=[0.0], kmax=kmax)
        results = camb.get_results(params)
        s8 = results.get_sigma8()
        print(f'Rescaled sigma_8: {s8}')
    kh, _, pk = results.get_matter_power_spectrum(
                                minkh=kmin, maxkh=kmax, npoints=npoints)
    return kh, pk[0]*DzD0**2