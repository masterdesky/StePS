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


def hubble_a(a, H0, omega_m, omega_l):
    r'''
    Computes the Hubble parameter $H(a)$ at scale factor $a$.

    The Hubble parameter is given by

    .. math::
        H(a) = H_0\,\sqrt{\Omega_m\,a^3 + (1 - \Omega_m - \Omega_\Lambda)\,a^2 + \Omega_\Lambda},

    where $a$ is the scale factor normalized to 1 at present. $H_0$ is
    the Hubble constant, $\Omega_m$ is the present-day matter density
    parameter and $\Omega_\Lambda$ is the present-day dark energy density
    parameter.

    Parameters
    ----------
    a : float
        Scale factor (normalized to 1 at present).
    H0 : float
        Hubble constant in km/s/Mpc.
    omega_m : float
        Present-day matter density parameter.
    omega_l : float
        Present-day dark energy density parameter.

    Returns
    -------
    float
        The Hubble parameter $H(a)$ evaluated at scale factor $a$.
    '''
    return H0 * np.sqrt(omega_m / a**3 + (1 - omega_m - omega_l) / a**2 + omega_l)

def F_omega(a, omega_m, omega_l):
    r'''
    Computes the linear growth rate factor for first-order Lagrangian
    perturbation.

    This function returns the factor $F_\omega(a)$, defined by

    .. math::
        F_\omega(a) = \left[\Omega(a)\right]^{0.6},

    where the effective matter density parameter $\Omega(a)$ is computed
    as

    .. math::
        \Omega(a) = \frac{\omega_m}{\omega_m + a\,(1 - \omega_m - \omega_l) + \omega_l\,a^3}.

    $F_\omega$ approximates the logarithmic derivative of the linear
    growth factor $D_1$ with respect to the scale factor $a$, i.e.

    .. math::
        f \equiv \frac{d\ln(D_1)}{d\ln(a)}.


    Parameters
    ----------
    a : float
        Scale factor (normalized to 1 at present).
    omega_m : float
        Present-day matter density parameter.
    omega_l : float
        Present-day dark energy density parameter.

    Returns
    -------
    float
        The linear growth rate $F_\omega$ evaluated at scale factor $a$.
    '''
    omega_a = omega_m / (omega_m + a * (1 - omega_m - omega_l) + a**3 * omega_l)
    return np.power(omega_a, 3.0/5.0)

def F2_omega(a, omega_m, omega_l):
    r'''
    Computes the second-order growth rate factor for second-order
    Lagrangian perturbation theory corrections.

    This function returns the factor $F2_\omega(a)$, defined by

    .. math::
        F2_\omega(a) = 2\,\left[\Omega(a)\right]^{\frac{4}{7}},

    where the effective matter density parameter $\Omega(a)$ is computed
    as

    .. math::
        \Omega(a) = \frac{\omega_m}{\omega_m + a\,(1 - \omega_m - \omega_l) + \omega_l\,a^3}.

    $F2_\omega$ is used in second-order Lagrangian perturbation theory
    to scale the second-order displacement field and its time derivative,
    thereby accounting for non-linear corrections to the growth of structure.

    Parameters
    ----------
    a : float
        Scale factor (normalized to 1 at present).
    omega_m : float
        Present-day matter density parameter.
    omega_l : float
        Present-day dark energy density parameter.

    Returns
    -------
    float
        The second-order growth rate $F2_\omega$ evaluated at scale
        factor $a$.
    '''
    omega_a = omega_m / (omega_m + a * (1 - omega_m - omega_l) + a**3 * omega_l)
    return 2 * np.power(omega_a, 4.0/7.0)

def linear_growth_function(
        z, H0, omega_m, omega_b, omega_l, s8, ns, de_model, de_params, silent=True):
    '''
    Calculate the linear growth factor $D_1 (a)$ normalized to $1$
    at $a = 1$.

    For $w_0$ and CPL dark energy, we use the Colossus implementation
    of Eq. (11) from Linder & Jenkins (2003).
    See paper at https://arxiv.org/pdf/astro-ph/0305286.pdf.

    Parameters:
    -----------
    z : float
        Redshift at which the linear growth factor is calculated.
    H0 : float
        Hubble constant in km/s/Mpc.
    omega_m : float
        Present-day matter density parameter.
    omega_b : float
        Present-day baryonic matter density parameter.
    omega_l : float
        Present-day dark energy density parameter.
    s8 : float
        RMS matter fluctuation amplitude at 8 Mpc/h.
    ns : float
        Scalar spectral index of the primordial power spectrum.
    de_model : str
        Dark energy model. Possible values are "Lambda", "w0", or "CPL").
    de_params : list or array-like
        Dark energy parameters (one element for "w0", two for "CPL").
    silent : bool, optional
        If True, suppress console output.

    Returns:
    --------
    D1 : float
        The linear growth factor at $a = 1$.
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
    # should not cause any significant errors at late times in relevant
    # cosmologies. TODO: Implement a non-zero `omega_r`
    T_cmb = 0.001
    
    params = {  # Common cosmological parameters for all models
        'flat': flat, 'H0': H0, 'Om0': omega_m, 'Ob0': omega_b,
        'sigma8': s8, 'ns': ns, 'Tcmb0': T_cmb
    }
    if de_model == 'Lambda':
        if not flat:
            params.update({'Ode0': omega_l})
        cosmo = cosmology.setCosmology('LCDM', **params)
    elif de_model == 'w0':
        params.update({'de_model': 'w0', 'w0': de_params[0]})
        if not flat:
            params.update({'Ode0': omega_l})
        cosmo = cosmology.setCosmology('wCDM', **params)
    elif de_model == 'CPL':
        params.update({'de_model': 'w0wa', 'w0': de_params[0], 'wa': de_params[1]})
        if not flat:
            params.update({'Ode0': omega_l})
        cosmo = cosmology.setCosmology('w0waCDM', **params)
    else:
        raise ValueError('Invalid dark energy model. Options are "Lambda", "w0", "CPL".')
    D1 = cosmo.growthFactorUnnormalized(z) / cosmo.growthFactorUnnormalized(0.0)
    if not silent:
        print(f'Initial normalized linear growth: D(z={z:.2f})/D(z=0) = {D1:.2e}')
    return D1

def camb_linear_spectrum(
        z=127, H0=73.0, ombh2=0.024, omch2=0.1092445, omega_k=0.0,
        As=2.0e-09, ns=1.0, kmin=0.01, kmax=1.0, npoints=1024,
        sigma8=None, de_model='Lambda', de_params=None):
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
    As : float
        Scalar amplitude of the primordial power spectrum.
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
    de_model : str
        Dark energy model. Options are 'Lambda', 'w0', 'CPL'.
    de_params : list or array-like
        Dark energy model parameters. For 'Lambda' and 'w0', it is a
        single element list containing the dark energy equation of state
        parameter $w$. For 'CPL', it is a two element list containing
        $w$ and $w_a$.
    '''
    params = camb.CAMBparams()
    params.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omega_k)
    if de_model == 'w0':
        assert len(de_params) == 1, 'de_params must be a single element list.'
        params.DarkEnergy = camb.dark_energy.DarkEnergyFluid()
        params.DarkEnergy.set_params(w=de_params[0])
    elif de_model == 'CPL':
        assert len(de_params) == 2, 'de_params must be a two element list.'
        params.DarkEnergy = camb.dark_energy.DarkEnergyFluid()
        params.DarkEnergy.set_params(w=de_params[0], wa=de_params[1])
    
    # Calculate sigma_8 at z=0
    params.InitPower.set_params(As=As, ns=ns)
    params.set_matter_power(redshifts=[0.0], kmax=kmax)
    params.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(params)
    s8 = results.get_sigma8()
    print(f'Original sigma_8: {s8}')

    # Calculating normalized linear growth function
    omega_m = (ombh2+omch2) / ((H0/100.0)**2)  # Non-relativistic matter density
    omega_b = ombh2 / ((H0/100.0)**2)          # Baryonic matter density
    omega_l = 1.0 - omega_k - omega_m          # Dark energy density
    DzD0 = linear_growth_function(
                z, H0, omega_m, omega_b, omega_l, s8, ns, de_model, de_params)
    
    # Calculating the linear P(k)
    if sigma8 is None:
        # Rescale the power spectrum to the desired sigma_8 value
        As_rescaled = As*(sigma8/s8)**2
        params.InitPower.set_params(
                As=As_rescaled, ns=ns, nrun=0, nrunrun=0.0, r=0.0,
                nt=None, ntrun=0.0, pivot_scalar=0.05, pivot_tensor=0.05,
                parameterization='tensor_param_rpivot')
        params.set_matter_power(redshifts=[z], kmax=kmax)
        results = camb.get_results(params)
        s8 = results.get_sigma8()
        print(f'sigma_8 at z={z}: {s8}')
    kh, _, pk = results.get_matter_power_spectrum(
                                minkh=kmin, maxkh=kmax, npoints=npoints)
    return kh, pk[0]*DzD0**2

def generate_camb(params):
    '''
    Setting the initial power spectrum with CAMB.

    Parameters:
    -----------
    params : dict
        Dictionary containing the parameter list of a StePS simulation.
    '''    
    print('Calculating input spectrum with CAMB...')
    kmin    = 1.0*np.pi/params['LBOX']
    kmax    = 100.0
    npoints = 2048
    kh, pk  = camb_linear_spectrum(
        H0=params['H0'], ombh2=params['OMBH2'], omch2=params['OMCH2'], omk=params['OMK'],
        ns=params['PRIMORDIALINDEX'], redshift=params['REDSHIFT'],
        kmin=kmin, kmax=kmax, npoints=npoints, sigma8=params['SIGMA8'],
        DE=params['DARKENERGYMODEL'], DE_params=params['DARKENERGYPARAMS'])
    Pk3 = np.vstack((np.log10(kh), np.log10(pk*kh**3/(2*np.pi**2)))).T
    np.savetxt(params['FILEWITHINPUTSPECTRUM'], Pk3)
    print('...done.')