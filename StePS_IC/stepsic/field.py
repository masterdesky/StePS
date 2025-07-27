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

import numpy as np
from scipy.interpolate import RegularGridInterpolator, CubicSpline

from stepsic.random import RNG

import logging
log = logging.getLogger(__name__)


def interpolate_field(x, field, dk, method='linear'):
    r'''
    Interpolate a grid-based field onto particle positions using periodic
    boundaries.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Particle positions in the simulation box.
    field : ndarray
        The grid-based field (e.g. a displacement field) defined on a
        regular grid.
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    method : str
        The interpolation method to use. This can be 'linear', 'nearest',
        or 'cubic'. The default is 'linear'.

    Returns
    -------
    interp_values : ndarray of shape (N,)
        Field values interpolated at the particle positions.
    '''
    nvox = field.shape
    mesh = tuple(np.arange(dk/2, n*dk+dk/2, dk) for n in nvox)
    interpolator = RegularGridInterpolator(
        points=mesh,
        values=field,
        method=method,
        bounds_error=False,
        fill_value=None  # Extrapolate using periodic wrapping if needed
    )
    return interpolator(x)


def create_grid(nvox, dk):
    '''
    Create a regular grid for the simulation box.

    Parameters
    ----------
    nvox : tuple of int
        The number of voxels in each dimension of the simulation box.
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.

    Returns
    -------
    particles : ndarray of shape (N, 3)
        The particle positions in the simulation box, where N is the total
        number of particles (voxels).
    coords : ndarray of shape (3, Nx, Ny, Nz)
        The grid coordinates in each dimension, where Nx, Ny, Nz are the
        number of voxels in each dimension.
    '''
    mesh = [np.arange(-(n-1)*dk/2, n*dk/2, dk) for n in nvox]
    xx, yy, zz = np.meshgrid(*mesh, indexing='ij')
    particles = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    return particles, np.array((xx, yy, zz))


def create_particles(npart: int, Lbox, seed=None):
    r'''
    Create a set of particles uniformly distributed in a cubic box.

    Parameters
    ----------
    npart : int
        The number of particles to create.
    Lbox : float or list of float
        Length of the box in each dimension [Lx, Ly, Lz] or a single
        float value for a cubic box.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    particles : ndarray of shape (npart, 3)
        Particle positions in the simulation box.
    '''
    rng = RNG(seed=seed)
    return rng.uniform(size=(npart, 3), seed=seed) * np.array(Lbox)


def cubic_voxels(nmesh, Lbox):
    '''
    Defines a rectangular cuboid mesh with the specified number of
    voxels in each dimensions, ensuring that the voxels are cubic.
    The function calculates the number of voxels in each dimension
    (Nx, Ny, Nz) based on the shortest dimension of the cuboid and
    scales the other dimensions accordingly.

    Parameters
    ----------
    nmesh : int
        Number of voxels in the shortest dimension.
    Lbox : float or list of float
        Length of the box in each dimension [Lx, Ly, Lz] or a single
        float value for a cubic box.

    Returns
    -------
    nvox : tuple of int
        The number of voxels in each dimension of the grid (Nx, Ny, Nz).
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    '''
    nvox = np.ceil(Lbox / (np.min(Lbox) / nmesh)).astype(int)
    nvox = (nvox + nvox % 2).astype(int)  # Ensure even number of voxels
    dk = np.min(Lbox) / np.min(nvox)
    log.info('mesh: Nx={}, Ny={}, Nz={}; step size: {}'.format(*nvox, dk))
    return nvox, dk


def fourier_grid(nvox, dk, hermitian=False):
    r'''
    Construct a 3D Fourier space grid.

    This function generates a three-dimensional array of wavevector
    components (``kvec``) and computes the corresponding magnitude
    (``kmod``) for a cubic grid with ``nmesh`` points per side within
    a box of size ``Lbox``.

    The grid is then constructed using the FFT frequencies:
    - For the first two dimensions, the full set of FFT frequencies is
      computed using ``np.fft.fftfreq``.
    - For the third dimension, if the input field is real-valued (i.e.
      if Hermitian symmetry is assumed), the reduced set of frequencies
      is computed using ``np.fft.rfftfreq``.

    Parameters
    ----------
    nvox : tuple of int
        The number of voxels in each dimension of the grid (Nx, Ny, Nz).
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    hermitian : bool
        If True, assume the field has Hermitian symmetry (i.e., it is
        real-valued) and use the reduced FFT along the last dimension.

    Returns
    -------
    kvec : ndarray
        A three-dimensional array of wavevector components with shape:
          - :math:`(3, {N_x}, {N_y}, {N_z}//2+1)` if ``hermitian`` is True.
          - :math:`(3, {N_x}, {N_y}, {N_z})` if ``hermitian`` is False.
        Each sub-array corresponds to the ``x``, ``y``, or ``z`` component
        of the wavevector.
    
    kmod : ndarray
        The magnitude of the wavevector at each grid point, computed as
        :math:`\|\mathbf{k}\| = \sqrt{k_x^2 + k_y^2 + k_z^2}`.
    '''
    kx = np.fft.fftfreq(nvox[0]) * 2 * np.pi / dk
    ky = np.fft.fftfreq(nvox[1]) * 2 * np.pi / dk
    if hermitian:
        kz = np.fft.rfftfreq(nvox[2]) * 2 * np.pi / dk
    else:
        kz = np.fft.fftfreq(nvox[2]) * 2 * np.pi / dk
    kvec = np.array(np.meshgrid(kx, ky, kz, indexing='ij'))
    kmod = np.linalg.norm(kvec, axis=0)
    return kvec, kmod


def white_noise(nvox, seed=None):
    r'''
    Return a complex Gaussian array :math:`W(k)` on the ``rfftn()`` grid
    `(Nx, Ny, Nz//2+1)`, obeying Hermitian constraints that guarantee
    :math:`\delta(x)` reconstructed with ``irfftn()`` is real.

    The field is generated in the real space.

    Parameters
    ----------
    nvox : tuple of int
        Number of voxels in each dimension (Nx, Ny, Nz).
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    seed : int or None, optional
        Random seed for reproducibility. If None, uses the default RNG.

    Returns
    -------
    w_k : ndarray
        3D array of white noise values.
    '''
    rng = RNG(seed=seed)
    w_k = np.fft.rfftn(rng.normal(size=nvox, seed=seed))
    w_k[0, 0, 0] = 0.0  # set DC=0 (mean density) as we only need fluctuations
    return w_k


def generate_delta_k(kh, pk, nvox, dk, *, field=None, seed=None):
    r'''
    Generates the Fourier modes of an arbitrary input field from a
    given power spectrum.

    Parameters
    ----------
    kh : ndarray
        1D array of wavenumbers (k), in h/Mpc.
    pk : ndarray
        1D array of the matter power spectrum P(k) at the initial redshift.
    nvox : tuple of int
        The number of voxels in each dimension of the grid (Nx, Ny, Nz).
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    field : ndarray
        Fourier transform of the real-valued overdensity field
        :math:`\delta(\mathbf{x})` defined on a regular grid, where
        :math:`\mathbf{x}` are the comoving coordinates.
    seed : int
        The seed for the random number generator.

    Returns
    -------
    delta_k : ndarray
        A 3D complex-valued array of shape (Nx, Ny, Nz//2+1) representing
        the Fourier modes of the overdensity field.
    '''
    _, kmod = fourier_grid(nvox, dk, hermitian=True)
    
    # interpolate the power spectrum in log-log space
    spline = CubicSpline(np.log(kh), np.log(pk), extrapolate=True)
    pk_grid = np.zeros_like(kmod, dtype=float)
    mask = kmod > 0
    if np.any(mask):
        ktarget_log = np.log(kmod[mask])
        pk_grid[mask] = np.exp(spline(ktarget_log))

    if field is None:
        field = white_noise(size=nvox, seed=seed)

    # Sirko 2005; Bagla & Padmanabhan 1997; Klypin & Holtzman 1997
    return field * np.sqrt(pk_grid / dk**3)