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


def interpolate(x, Lbox, disp_field):
    '''
    Perform trilinear interpolation of the displacement field onto
    particle positions.
    
    Parameters:
    -----------
    x : ndarray of shape (N, 3)
        Particle positions.
    Lbox : float
        Box size in Mpc/h.
    disp_field : ndarray of shape (N, N, N)
        Displacement field.
    
    Returns:
    --------
    disp_field_interp : ndarray of shape (N, 3)
        Interpolated displacement field at particle positions.
    '''
    nmesh = disp_field.shape[0]

    u, v, w = x.T / Lbox * nmesh

    i = np.floor(u).astype(int)
    j = np.floor(v).astype(int)
    k = np.floor(w).astype(int)

    u -= i
    v -= j
    w -= k

    i = i % nmesh
    j = j % nmesh
    k = k % nmesh

    ii = (i + 1) % nmesh
    jj = (j + 1) % nmesh
    kk = (k + 1) % nmesh

    # Calculate the trilinear interpolation coefficients for each
    # of the 8 vertices of the cube the particle is in
    f1 = (1 - u) * (1 - v) * (1 - w)
    f2 = (1 - u) * (1 - v) * w
    f3 = (1 - u) * v * (1 - w)
    f4 = (1 - u) * v * w
    f5 = u * (1 - v) * (1 - w)
    f6 = u * (1 - v) * w
    f7 = u * v * (1 - w)
    f8 = u * v * w

    disp_field_interp = (
        disp_field[i, j, k] * f1[:, None] +
        disp_field[i, j, kk] * f2[:, None] +
        disp_field[i, jj, k] * f3[:, None] +
        disp_field[i, jj, kk] * f4[:, None] +
        disp_field[ii, j, k] * f5[:, None] +
        disp_field[ii, j, kk] * f6[:, None] +
        disp_field[ii, jj, k] * f7[:, None] +
        disp_field[ii, jj, kk] * f8[:, None]
    )
    return disp_field_interp

def zeldovich(x, Lbox, overdensity_field, growth_rate, h):
    r'''
    Apply the Zel'dovich approximation to generate perturbed particle
    positions and velocities.

    This function implements the Zel'dovich approximation, a first-order
    Lagrangian perturbation theory used to set up initial conditions for
    cosmological N-body simulations. In this approximation, particles
    are displaced from their initial (Lagrangian) positions, \(\mathbf{q}\),
    to Eulerian positions, \(\mathbf{x}\), according to

    .. math::

        \mathbf{x}(\mathbf{q}, t) = \mathbf{q} + D(t) \, \mathbf{\Psi}(\mathbf{q}),

    where:
    
    - \(D(t)\) is the linear growth factor,
    - \(\mathbf{\Psi}(\mathbf{q})\) is the displacement field computed
      from the density perturbations.

    The displacement field is obtained by solving the linearized Poisson
    equation in Fourier space. Specifically, for each component \(i\) the
    displacement field in Fourier space is

    .. math::

        \Psi_i(\mathbf{k}) = -i \, \frac{k_i}{|\mathbf{k}|^2} \, \delta(\mathbf{k})
        \quad \text{for } |\mathbf{k}| > 0,

    where:
    
    - \(\delta(\mathbf{k})\) is the Fourier transform of the real-space
      overdensity field,
    - \(k_i\) is the \(i\)-th component of the wavevector,
    - \(|\mathbf{k}|\) is the magnitude of the wavevector.

    The inverse Fourier transform then yields the displacement field
    \(\mathbf{\Psi}(\mathbf{x})\) in real space. Particle velocities are
    computed as

    .. math::

        \mathbf{v} = \dot{D}(t) \, \mathbf{\Psi},

    where \(\dot{D}(t)\) (provided as ``growth_rate``) is the time
    derivative of the growth factor.

    The Fourier transform is performed on a grid defined by
    ``overdensity_field``. For a real-valued field, Hermitian symmetry
    allows one to use a reduced FFT along one dimension. Here, the first
    two dimensions use ``np.fft.fftfreq`` and the third uses
    ``np.fft.rfftfreq``, resulting in a k-space vector array, ``kvec``,
    with shape \((3, \texttt{nmesh}, \texttt{nmesh}, \texttt{nmesh}//2+1)\).

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
    growth_rate : float
        The time derivative of the linear growth factor, \(\dot{D}(t)\),
        in km/s/Mpc. This is used to compute the particle velocities.
    h : float
        Dimensionless Hubble parameter, \(h = H_0 / 100\), where \(H_0\)
        is the value of the Hubble constant in km/s/Mpc.

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

    Notes
    -----
    The function follows these key steps:

    1. **Fourier Transform Setup:**
       - The resolution of the FFT grid is determined by the shape of
         ``overdensity_field``.
       - Wavevector arrays are constructed:
         - ``kk`` is generated using ``np.fft.fftfreq`` for the first
           two dimensions.
         - ``ks`` is generated using ``np.fft.rfftfreq`` for the third
           dimension.
       - These arrays are combined via ``np.meshgrid`` to form the 3D
         k-space vector array ``kvec``, which has a reduced size in the
         last dimension due to the use of the real FFT.

    2. **Computing the Displacement Field:**
       - The overdensity field is transformed to Fourier space using
       ``np.fft.rfftn``, yielding \(\delta(\mathbf{k})\).
       - For each spatial component (x, y, z), the Fourier-space
         displacement is computed as:

         .. math::

             \Psi_i(\mathbf{k}) = -i \, \frac{k_i}{|\mathbf{k}|^2} \, \delta(\mathbf{k}),

         with a mask applied to avoid division by zero for \( |\mathbf{k}| = 0 \).
       - The inverse FFT (``np.fft.irfftn``) converts the displacement
         field back to real space.

    3. **Interpolation and Particle Update:**
       - The grid-based displacement field is interpolated to the actual
         particle positions.
       - Particle positions are updated according to

         .. math::

             \mathbf{x}_{\text{pert}} = \mathbf{x} + \mathbf{\Psi},

         and the velocities are computed as

         .. math::

             \mathbf{v} = \dot{D}(t) \, \mathbf{\Psi}.
         
       - Periodic boundary conditions are enforced by wrapping the
         positions, and velocities are rescaled by the dimensionless
         Hubble constant ``h`` for unit consistency.

    4. **Fourier Domain Considerations:**
       - The use of separate ``kk`` and ``ks`` arrays is due to the
         Hermitian symmetry inherent in the Fourier transform of real
         fields: full frequency information is needed for the first two
         dimensions, while only the non-negative frequencies are stored
         for the third dimension.

    Examples
    --------
    >>> import numpy as np
    >>> # Define particle positions, box size, overdensity field, growth rate, and Hubble constant
    >>> x = np.random.rand(1000, 3).astype(np.float32) * 100.0  # Example positions in Mpc/h
    >>> Lbox = 100.0  # Mpc/h
    >>> overdensity_field = np.random.randn(64, 64, 64).astype(np.float32)
    >>> growth_rate = 10.0  # km/s/Mpc
    >>> h = 70.0 / 100  # dimensionless Hubble parameter
    >>> xpert, v = zeldovich(x, Lbox, overdensity_field, growth_rate, h)
    '''
    # Nagyskálás módusok mindig ugyanazok legyenek ugyanarra a seedre
    # Feltöltése a módusoknak nagyobbtól kisebbek irányába
    nmesh = overdensity_field.shape[0]
    kk = np.fft.fftfreq(nmesh) * 2*np.pi/Lbox * nmesh
    ks = np.fft.rfftfreq(nmesh) * 2*np.pi/Lbox * nmesh  # Hermitian symmetry
    kvec = np.array(np.meshgrid(kk, kk, ks))
    kmod = np.linalg.norm(kvec, axis=0)
    delta_k = np.fft.rfftn(overdensity_field)
    xpert = np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x, dtype=np.float32)
    for i, xi in enumerate(('x', 'y', 'z')):
        psi_i = np.zeros_like(kmod, dtype=complex)
        mask = kmod > 0.0
        psi_i[mask] = -1j * kvec[i, mask] / kmod[mask]**2 * delta_k[mask]
        disp_field = np.fft.irfftn(psi_i)
        max_disp = np.max(disp_field)
        print(f'Maximal \'{xi}\' displacement: {max_disp*1000} kpc/h; '\
              f'in units of mean particle separation: {max_disp * nmesh/Lbox}')
        disp_field_interp = interpolate(x, Lbox, disp_field)
        xpert[:, i] = x[:, i] + disp_field_interp
        v[:, i] = disp_field_interp * growth_rate
    return xpert%Lbox, v/h

def second_order_lpt():
    '''
    Perform second-order Lagrangian perturbation theory.
    '''
    pass