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
    Computes the Hubble parameter :math:`H(a)` at scale factor :math:`a`.

    The Hubble parameter is given by

    .. math::
        H(a) = H_0\,\sqrt{\Omega_m\,a^3 + (1 - \Omega_m - \Omega_\Lambda)\,a^2 + \Omega_\Lambda},

    where :math:`a` is the scale factor normalized to 1 at present. :math:`H_0`
    is the Hubble constant, :math:`\Omega_m` is the present-day matter
    density parameter and :math:`\Omega_\Lambda` is the present-day dark
    energy density parameter.

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
        The Hubble parameter :math:`H(a)`.
    '''
    return H0 * np.sqrt(omega_m / a**3 + (1 - omega_m - omega_l) / a**2 + omega_l)


def F_omega(a, omega_m, omega_l):
    r'''
    Computes the linear growth rate factor for first-order Lagrangian
    perturbation.

    This function returns the factor :math:`F_\omega(a)`, defined by

    .. math::
        F_\omega(a) = \left[\Omega(a)\right]^{0.6},

    where the effective matter density parameter :math:`\Omega(a)` is computed
    as

    .. math::
        \Omega(a) = \frac{\omega_m}{\omega_m + a\,(1 - \omega_m - \omega_l) + \omega_l\,a^3}.

    :math:`F_\omega` approximates the logarithmic derivative of the linear
    growth factor :math:`D_1` with respect to the scale factor :math:`a`, i.e.

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
        The linear growth rate :math:`F_\omega(a)`.
    '''
    omega_a = omega_m / (omega_m + a * (1 - omega_m - omega_l) + a**3 * omega_l)
    return np.power(omega_a, 5.0/9.0)  # Bernardeau et al. 2001, eq. 101a


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
    return 2 * np.power(omega_a, 6.0/11.0)  # Bernardeau et al. 2001, eq. 101


def D1_z(z, H0, omega_m, omega_b, omega_l, sigma8, ns, de_model, de_params, silent=True):
    '''
    Calculate the linear growth factor $D_1 (z)$ normalized to $1$
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
    sigma8 : float
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
        print(f'Non-flat cosmology; {omega_k = }, {omega_m = :.4f}, {omega_l = :.4f}')
    # Zero CMB temperature can cause issues in colossus. This small value
    # should not cause any significant errors at late times in relevant
    # cosmologies. TODO: Implement a non-zero `omega_r`
    T_cmb = 0.001
    
    params = {  # Common cosmological parameters for all models
        'flat': flat, 'H0': H0, 'Om0': omega_m, 'Ob0': omega_b,
        'sigma8': sigma8, 'ns': ns, 'Tcmb0': T_cmb
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
        print(f'Normalized linear growth factor D1(z={z:.2f})/D1(z=0) = {D1:.2e}')
    return D1


def init_cosmology(
        H0=67.4, ombh2=0.0224, omch2=0.120, omega_k=0.0,
        de_model='Lambda', de_params=None,
        nonlinear=False, halofit_version='mead2020'):
    '''
    Initialize a CAMB cosmology object with specified parameters. The
    default cosmological parameters are from the Planck 2018 results.
    
    Parameters:
    -----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter.
    omch2 : float
        Physical cold dark matter density parameter.
    omega_k : float
        Curvature density parameter.
    kmax : float
        Maximum wavenumber in h/Mpc.
    de_model : str
        Dark energy model. Options are 'Lambda', 'w0', 'CPL'.
    de_params : list or array-like
        Dark energy model parameters. For 'Lambda' and 'w0', it is a
        single element list containing the dark energy equation of state
        parameter $w$. For 'CPL', it is a two element list containing
        $w$ and $w_a$.
    nonlinear : bool; default=False
        If True, include non-linear corrections using Halofit.
    halofit_version : str; default='mead2020'
        Version of the Halofit model to use for non-linear corrections.
        Check ``camb.nonlinear.Halofit`` for available models.
    '''
    camb_params = camb.CAMBparams()
    camb_params.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omega_k)
    if de_model == 'w0':
        if not (isinstance(de_params, (list, tuple)) and len(de_params) == 1):
            raise ValueError("`de_params` must be a 1-element list for 'w0'.")
        camb_params.DarkEnergy = camb.dark_energy.DarkEnergyFluid()
        camb_params.DarkEnergy.set_params(w=de_params[0])
    elif de_model == 'CPL':
        if not (isinstance(de_params, (list, tuple)) and len(de_params) == 2):
            raise ValueError("`de_params` must be a 2-element list for 'CPL'.")
        camb_params.DarkEnergy = camb.dark_energy.DarkEnergyFluid()
        camb_params.DarkEnergy.set_params(w=de_params[0], wa=de_params[1])
    
    if nonlinear:
        camb_params.NonLinear = camb.model.NonLinear_both
        camb_params.NonLinearModel = camb.nonlinear.Halofit()
        camb_params.NonLinearModel.set_params(halofit_version=halofit_version)
    else:
        camb_params.NonLinear = camb.model.NonLinear_none

    return camb_params


def calculate_sigma8(
        camb_params, *, z=0, As=2.097e-09, ns=0.965, kmax=1.0, silent=True):
    r'''
    Calculate the RMS matter fluctuation amplitude $\sigma_8$ at given
    redshift $z$.
    
    This function uses the CAMB package to compute the matter power
    spectrum and extract the $\sigma_8$ value. The default cosmological
    parameters are from the Planck 2018 results.

    Parameters:
    -----------
    camb_params : camb.CAMBparams
        CAMB parameters object initialized with cosmological parameters.
    z : float; default=0
        Target redshift.
    As : float
        Comoving curvature power at $k = 0.05\,\mathrm{Mpc}^{-1}$.
        This is the amplitude of the primordial power spectrum at large
        scales, typically set to match the observed $\sigma_8$.
    ns : float
        Scalar spectral index.
    kmax : float
        Maximum wavenumber in $h^{-1}\,\mathrm{Mpc}$.
    silent : bool
        If True, suppress console output.

    Returns:
    --------
    float
        The RMS matter fluctuation amplitude $\sigma_8$ at redshift $z$.
    '''
    # Calculate sigma8 at z=0 for the input As amplitude
    camb_params.InitPower.set_params(As=As, ns=ns)
    camb_params.set_matter_power(redshifts=[z], kmax=kmax)
    results = camb.get_results(camb_params)
    sigma8 = results.get_sigma8()[0]
    
    if not silent:
        print(f'RMS matter fluctuation amplitude {sigma8 = :.4f} (from {As = :.3e})')
    return sigma8


def camb_spectrum(
        camb_params, *, z=127, As=2.097e-09, ns=0.965, kmin=0.01, kmax=1.0,
        npoints=1024, sigma8_init: float = None, silent=True):
    r'''
    Calculate the matter power spectrum using CAMB. The default
    cosmological parameters are from the Planck 2018 results.

    Parameters:
    -----------
    z : float
        Redshift at which the linear power spectrum is calculated.
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
    sigma8_init : float; default=None
        Rescale the matter power spectrum to the given $\sigma_8$ value.
    silent : bool; default=True
        If True, suppress console output.
    '''
    # Optional rescaling of the `As` amplitude to match a desired sigma8
    if sigma8_init is not None:
        sigma8 = calculate_sigma8(
                camb_params, z=0, As=As, ns=ns, kmax=kmax, silent=silent)

        As *= (sigma8_init / sigma8)**2
        camb_params.InitPower.set_params(As=As, ns=ns)
        camb_params.set_matter_power(redshifts=[0.0], kmax=kmax)
        results = camb.get_results(camb_params)
        sigma8 = results.get_sigma8()[0]

        if not silent:
            print(f'Rescaled matter fluctuation amplitude {sigma8 = :.4f} (from {As = :.3e})')

    # Calculating P(k) at redshift `z`
    camb_params.set_matter_power(redshifts=[z], kmax=kmax)
    results = camb.get_results(camb_params)
    kh, _, pk = results.get_matter_power_spectrum(
                                minkh=kmin, maxkh=kmax, npoints=npoints)
    return kh, pk[0]#*DzD0**2


def generate_camb(params, silent=True):
    '''
    Setting the initial power spectrum with CAMB.

    Parameters:
    -----------
    params : dict
        Dictionary containing the parameter list of a StePS simulation.
    silent : bool; 
        If True, suppress console output.
    '''    
    print('Calculating input spectrum with CAMB...')
    camb_params = init_cosmology(
        H0=params['H0'], ombh2=params['OMBH2'], omch2=params['OMCH2'], omk=params['OMK'],
        de_model=params['DARKENERGYMODEL'], de_params=params['DARKENERGYPARAMS'],
        nonlinear=params['NONLINEAR'], halofit_version=params['HALOFITVERSION'])
    kmin    = 1.0*np.pi/params['LBOX']
    kmax    = 100.0
    npoints = 2048
    kh, pk  = camb_spectrum(
        camb_params, z=params['REDSHIFT'], kmin=kmin, kmax=kmax, npoints=npoints,
        sigma8_init=params['SIGMA8'], silent=silent)
    Pk3 = np.vstack((np.log10(kh), np.log10(pk*kh**3/(2*np.pi**2)))).T
    np.savetxt(params['FILEWITHINPUTSPECTRUM'], Pk3)
    print('...done.')