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
from scipy.interpolate import RegularGridInterpolator


def interpolate_field(x, field, Lbox):
    r'''
    Interpolate a grid-based field onto particle positions using periodic
    boundaries.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Particle positions in the simulation box.
    field : ndarray
        The grid-based field (e.g., a displacement field) defined on a
        regular grid.
    Lbox : float
        The simulation box size.

    Returns
    -------
    interp_values : ndarray of shape (N,)
        Field values interpolated at the particle positions.
    '''
    nmesh = field.shape[0]
    grid = np.linspace(0, Lbox, nmesh, endpoint=False)
    interpolator = RegularGridInterpolator(
        (grid, grid, grid),
        field,
        method='linear',
        bounds_error=False,
        fill_value=None  # Extrapolate using periodic wrapping if needed
    )
    return interpolator(np.mod(x, Lbox))


def fourier_grid(nmesh, Lbox, hermitian=False):
    r'''
    Construct a 3D Fourier space grid for use in cosmological perturbation
    calculations.

    This function generates a three-dimensional array of wavevector
    components (`kvec`) and computes the corresponding magnitude (`kmod`)
    for a cubic grid with `nmesh` points per side within a simulation box
    of size `Lbox`. These arrays are essential for performing discrete
    Fourier transforms (FFTs) of simulation fields.

    The grid is constructed using the FFT frequencies:
    
    - For the first two dimensions, the full set of FFT frequencies is
      computed using ``np.fft.fftfreq``.
    - For the third dimension, if the input field is real-valued (i.e.
      if Hermitian symmetry is assumed), the reduced set of frequencies
      is computed using ``np.fft.rfftfreq``.
    
    When ``hermitian`` is True, the output grid reflects the storage
    scheme of a real FFT, and the shape of ``kvec`` is
    \((3, \texttt{nmesh}, \texttt{nmesh}, \texttt{nmesh}//2+1)\). Otherwise,
    a full grid with shape \((3, \texttt{nmesh}, \texttt{nmesh}, \texttt{nmesh})\)
    is returned.

    Parameters
    ----------
    nmesh : int
        The number of grid points along each dimension of the mesh.
    Lbox : float
        The physical size of the simulation box (e.g., in Mpc/h).
    hermitian : bool, optional
        If True, assume the field has Hermitian symmetry (i.e., it is
        real-valued) and use the reduced FFT along the last dimension.
        The default is False.

    Returns
    -------
    kvec : ndarray
        A three-dimensional array of wavevector components with shape:
          - \((3, nmesh, nmesh, nmesh//2+1)\) if ``hermitian`` is True.
          - \((3, nmesh, nmesh, nmesh)\) if ``hermitian`` is False.
        Each sub-array corresponds to the \(x\), \(y\), or \(z\) component
        of the wavevector.
    kmod : ndarray
        The magnitude of the wavevector at each grid point, computed as
        \(\|\mathbf{k}\| = \sqrt{k_x^2 + k_y^2 + k_z^2}\).

    Examples
    --------
    >>> import numpy as np
    >>> kvec, kmod = fourier_grid(64, 100.0, hermitian=True)
    >>> kvec.shape
    (3, 64, 64, 33)
    >>> kmod.shape
    (64, 64, 33)
    '''
    kk = np.fft.fftfreq(nmesh) * 2 * np.pi / Lbox * nmesh
    ks = np.fft.rfftfreq(nmesh) * 2 * np.pi / Lbox * nmesh if hermitian else kk
    kvec = np.array(np.meshgrid(kk, kk, ks, indexing='ij'))
    kmod = np.linalg.norm(kvec, axis=0)
    return kvec, kmod


def zeldovich(x, Lbox, overdensity_field, dD1, h):
    r'''
    Apply the Zel'dovich approximation to generate perturbed particle
    positions and velocities.

    This function implements the Zel'dovich approximation—a first-order
    Lagrangian perturbation theory commonly used to set up initial
    conditions for cosmological N-body simulations. In this approximation,
    particles are displaced from their initial (Lagrangian) positions
    \(\mathbf{q}\) to their Eulerian positions \(\mathbf{x}\) via the
    displacement field \(\mathbf{\Psi}\):

    .. math::

        \mathbf{x}(\mathbf{q}, t) = \mathbf{q} + D(t) \, \mathbf{\Psi}(\mathbf{q}),

    where:
    
      - \(D(t)\) is the linear growth factor.
      - \(\mathbf{\Psi}(\mathbf{q})\) is the displacement field computed
        from the density perturbations.

    The displacement field is determined by solving the linearized Poisson
    equation in Fourier space. For each spatial component \(i\), the
    Fourier-space displacement is computed as

    .. math::

        \Psi_i(\mathbf{k}) = -i \, \frac{k_i}{|\mathbf{k}|^2} \, \delta(\mathbf{k})
        \quad \text{for } |\mathbf{k}| > 0,

    where:
    
      - \(\delta(\mathbf{k})\) is the Fourier transform of the real-space
        overdensity field,
      - \(k_i\) is the \(i\)-th component of the wavevector,
      - \(|\mathbf{k}|\) is the magnitude of the wavevector.

    The function performs the following key steps:

    1. **Fourier Transform Setup:**
       - It determines the resolution of the FFT grid from the shape of
         the input ``overdensity_field``.
       - It constructs the wavevector arrays by calling the utility
         function ``fourier_grid()``. For a real field, the grid is built
         using ``np.fft.fftfreq`` for the first two dimensions and
         ``np.fft.rfftfreq`` for the third, yielding a k-space array of
         shape \((3, \texttt{nmesh}, \texttt{nmesh}, \texttt{nmesh}//2+1)\).
    
    2. **Computing the Displacement Field:**
       - The overdensity field is Fourier-transformed (using ``np.fft.rfftn``)
         to obtain \(\delta(\mathbf{k})\).
       - For each spatial axis \(i\), the Fourier-space displacement
         \(\Psi_i(\mathbf{k})\) is computed with a mask to avoid division
         by zero at \( |\mathbf{k}| = 0 \).
       - An inverse FFT (``np.fft.irfftn``) converts the Fourier-space
         displacement field back to real space.
    
    3. **Interpolation and Particle Update:**
       - The grid-based displacement field is interpolated to the actual
         particle positions using a suitable interpolation routine (e.g.,
         a trilinear interpolator).
       - Particle positions are updated as

         .. math::

             \mathbf{x}_{\text{pert}} = \mathbf{x} + \mathbf{\Psi},

         and the velocities are computed as

         .. math::

             \mathbf{v} = \dot{D}(t) \, \mathbf{\Psi},
         
         where \(\dot{D}(t)\) is provided as ``growth_rate``.
       - Periodic boundary conditions are enforced by wrapping the updated
         positions, and the velocities are rescaled by the dimensionless
         Hubble parameter ``h``.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Initial unperturbed particle positions (Lagrangian coordinates),
        typically in Mpc/h.
    Lbox : float
        The size of the simulation box in Mpc/h. Periodic boundary
        conditions are assumed.
    overdensity_field : ndarray of shape (M, M, M)
        Real-space overdensity field used to generate the displacement
        field. 
    dD1 : float
        The time derivative of the linear growth factor, \(\dot{D}(t)\),
        in km/s/Mpc. This is used to compute the particle velocities.
    h : float
        Dimensionless Hubble parameter, \(h = H_0/100\), where \(H_0\) is
        the Hubble constant in km/s/Mpc.

    Returns
    -------
    xpert : ndarray of shape (N, 3)
        Perturbed particle positions (Eulerian coordinates) after adding
        the displacement field. Positions are wrapped periodically within
        the simulation box.
    v : ndarray of shape (N, 3)
        Particle velocities in km/s, computed as

        .. math::

            \mathbf{v} = \dot{D}(t) \, \mathbf{\Psi}.

    Examples
    --------
    >>> import numpy as np
    >>> # Particle pos., box size, overdensity field, growth rate, and H0
    >>> x = np.random.rand(1000, 3).astype(np.float32) * 100.0  # Pos. in Mpc/h
    >>> Lbox = 100.0  # Mpc/h
    >>> overdensity_field = np.random.randn(64, 64, 64).astype(np.float32)
    >>> growth_rate = 10.0  # km/s/Mpc
    >>> h = 70.0 / 100  # Dimensionless Hubble parameter
    >>> xpert, v = zeldovich(x, Lbox, overdensity_field, growth_rate, h)
    '''
    nmesh = overdensity_field.shape[0]
    kvec, kmod = fourier_grid(nmesh, Lbox, hermitian=True)
    delta_k = np.fft.rfftn(overdensity_field)
    xpert = np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x, dtype=np.float32)
    for i, xi in enumerate(('x', 'y', 'z')):
        psi_i = np.zeros_like(kmod, dtype=complex)
        mask = kmod > 0.0
        psi_i[mask] = -1j * kvec[i, mask] / (kmod[mask] ** 2) * delta_k[mask]
        disp_field = np.fft.irfftn(psi_i)
        max_disp = np.max(np.abs(disp_field))
        print(f"Maximal '{xi}' displacement: {max_disp*1000:.3f} kpc/h; "
              f"in units of mean particle separation: {max_disp * nmesh / Lbox:.3f}")
        disp_field_interp = interpolate_field(x, Lbox, disp_field)
        xpert[:, i] = x[:, i] + disp_field_interp
        v[:, i] = disp_field_interp * dD1
    return np.mod(xpert, Lbox), v / h


def twolpt(x, overdensity_field, Lbox, dD1, dD2, h):
    '''
    Perform second-order Lagrangian perturbation theory.
    '''
    nmesh = overdensity_field.shape[0]
    kvec, kmod = fourier_grid(nmesh, Lbox, hermitian=True)
    delta_k = np.fft.rfftn(overdensity_field)
    xpert = np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x, dtype=np.float32)

    #...

    for i, xi in enumerate(('x', 'y', 'z')):
        # ...
        xpert[:, i] = x[:, i] + psi_1i

    return np.mod(xpert, Lbox), v / h
