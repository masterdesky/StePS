#*******************************************************************************#
#  StePS_IC.py - An initial condition generator for                             #
#     STEreographically Projected cosmological Simulations                      #
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
from scipy.interpolate import RegularGridInterpolator


def interpolate_field(x, field, Lbox, method='linear'):
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
    method : str, optional
        The interpolation method to use. This can be 'linear', 'nearest',
        or 'cubic'. The default is 'linear'.

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
        method=method,
        bounds_error=False,
        fill_value=None  # Extrapolate using periodic wrapping if needed
    )
    return interpolator(np.mod(x, Lbox))


def make_density_field(nmesh, Lbox, seed=None):
    r'''TODO
    Generate a Gaussian random overdensity field on a regular grid with
    periodic boundaries, sampled from $P(k)$ at that resolution.

    Parameters
    ----------
    nmesh : int
        The number of grid points along each dimension of the mesh.
    Lbox : float
        The size of the simulation box in Mpc/h.
    '''
    return 

def compute_density_field(x, nmesh, Lbox):
    r'''
    Compute the density field from particle positions using a
    cloud-in-cell (CIC) method.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Particle positions in the simulation box.
    nmesh : int
        The number of grid points along each dimension of the mesh.
    Lbox : float
        The size of the simulation box in Mpc/h.

    Returns
    -------
    density_field : ndarray of shape (nmesh, nmesh, nmesh)
        The computed density field on a regular grid.
    '''
    density_field = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)
    for i in range(x.shape[0]):
        xi = np.floor(x[i] / Lbox * nmesh).astype(int) % nmesh
        density_field[xi[0], xi[1], xi[2]] += 1.0
    return density_field


def compute_overdensity(density_field):
    r'''
    Compute the overdensity field $\delta(\mathbf{x})$ from the given
    density field.

    Parameters
    ----------
    density_field : ndarray
        The density field (e.g., particle counts or mass density on a
        grid).

    Returns
    -------
    overdensity_field : ndarray
        The overdensity field, defined as
        $\delta(\mathbf{x}) = \frac{\rho(\mathbf{x}) - \bar{\rho}}{\bar{\rho}}$.
    '''
    mean_density = np.mean(density_field)
    return (density_field - mean_density) / mean_density


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
    $(3, {\rm nmesh}, {\rm nmesh}, {\rm nmesh}//2+1)$. Otherwise,
    a full grid with shape $(3, {\rm nmesh}, {\rm nmesh}, {\rm nmesh})$
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
          - $(3, {\mr nmesh}, {\rm nmesh}, {\rm nmesh}//2+1)$ if
          ``hermitian`` is True.
          - $(3, {\rm nmesh}, {\rm nmesh}, {\rm nmesh})$ if
          ``hermitian`` is False.
        Each sub-array corresponds to the $x$, $y$, or $z$ component
        of the wavevector.
    kmod : ndarray
        The magnitude of the wavevector at each grid point, computed as
        $\|\mathbf{k}\| = \sqrt{k_x^2 + k_y^2 + k_z^2}$.

    Examples
    --------
    >>> import numpy as np
    >>> kvec, kmod = fourier_grid(64, 100.0, hermitian=True)
    >>> kvec.shape
    (3, 64, 64, 33)
    >>> kmod.shape
    (64, 64, 33)
    '''
    kk = np.fft.fftfreq(nmesh) * 2*np.pi / Lbox * nmesh
    ks = np.fft.rfftfreq(nmesh) * 2*np.pi / Lbox * nmesh if hermitian else kk
    kvec = np.array(np.meshgrid(kk, kk, ks, indexing='ij'))
    kmod = np.linalg.norm(kvec, axis=0)
    return kvec, kmod


def zeldovich(x, Lbox, density_field, dD1, h, counter=False):
    r'''
    Apply the Zel'dovich approximation to generate perturbed particle
    positions and velocities.

    This function implements the Zel'dovich approximation—a first-order
    Lagrangian perturbation theory commonly used to set up initial
    conditions for cosmological N-body simulations. In this approximation,
    particles are displaced from their initial (Lagrangian) positions
    $\mathbf{q}$ to their Eulerian positions $\mathbf{x}$ via the
    displacement field $\mathbf{\Psi}$:

    .. math::
        \mathbf{x}(\mathbf{q}, t) =
            \mathbf{q} + D(t) \, \mathbf{\Psi}(\mathbf{q}),

    where:
      - $D(t)$ is the linear growth factor,
      - $\mathbf{\Psi}(\mathbf{q})$ is the displacement field computed
        from the density perturbations.

    The displacement field is determined by solving the linearized Poisson
    equation in Fourier space. For each spatial component $i$, the
    Fourier-space displacement is computed as

    .. math::
        \Psi_i(\mathbf{k}) =
            -i \, \frac{k_i}{|\mathbf{k}|^2} \, \delta(\mathbf{k})
        \quad \text{for } |\mathbf{k}| > 0,

    where:
      - $\delta(\mathbf{k})$ is the Fourier transform of the computed
        overdensity field,
      - $k_i$ is the $i$-th component of the wavevector,
      - $|\mathbf{k}|$ is the magnitude of the wavevector.

    **Overview of the Implementation:**

    1. **Fourier grid setup:**
       - A 3D Fourier space grid is constructed (via ``fourier_grid()``)
         for a mesh of size $n_{\rm mesh}$ in a box of size $L_{\rm box}$.
         For a real field, the third dimension is handled using a reduced
         FFT (Hermitian symmetry) so that the k-space array has shape
         $(3, n_{\rm mesh}, n_{\rm mesh}, n_{\rm mesh}//2+1)$.
    
    2. **Computing the displacement field:**
       - The overdensity field is Fourier-transformed (using ``np.fft.rfftn``)
         to obtain $\delta(\mathbf{k})$.
       - For each spatial axis $i$, the Fourier-space displacement
         $\Psi_i(\mathbf{k})$ is computed with a mask to avoid division
         by zero at $ |\mathbf{k}| = 0 $.
       - An inverse FFT (``np.fft.irfftn``) converts the Fourier-space
         displacement field back to real space.
    
    3. **Interpolation and particle update:**
       - The grid-based displacement field is interpolated to the actual
         particle positions using a suitable interpolation routine (here,
         a trilinear interpolator).
       - Particle positions are updated as

         .. math::
             \mathbf{x}_{\text{pert}} = \mathbf{x} + \mathbf{\Psi},

         and the velocities are computed as

         .. math::
             \mathbf{v} = \dot{D}(t) \, \mathbf{\Psi},
         
         where $\dot{D}(t)$ is provided as ``dD1``.
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
    density_field : ndarray of shape (M, M, M)
        Real-space density field from which the overdensity field is
        computed.
    dD1 : float
        The time derivative of the linear growth factor, $\dot{D}(t)$,
        in km/s/Mpc. This is used to compute the particle velocities.
    h : float
        Dimensionless Hubble parameter, $h = H_0/100$, where $H_0$
        is the Hubble constant in km/s/Mpc.
    counter : bool, optional
        If True, apply a $\pi$ radian global phase shift to the density
        field before computing the displacement field. This is useful
        to run counter-phased simulations. The default is False.

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
    >>> # Box size, particle coordinates, density field, growth rate, and H0
    >>> Lbox = 100.0  # Mpc/h
    >>> x = np.random.rand(1000, 3).astype(np.float32) * Lbox  # Coords in Mpc/h
    >>> density_field = np.random.randn(64, 64, 64).astype(np.float32)
    >>> dD1 = 10.0  # km/s/Mpc (time derivative of the growth factor)
    >>> h = 67.6 / 100  # Dimensionless Hubble parameter
    >>> xpert, v = zeldovich(x, Lbox, density_field, dD1, h)
    
    Notes
    -----
    - The interpolation step is needed since the displacement field is
      computed on a regular grid, but particle positions are generally
      not located exactly on grid points.
    '''
    nmesh = density_field.shape[0]
    overdensity_field = compute_overdensity(density_field)
    kvec, kmod = fourier_grid(nmesh, Lbox, hermitian=True)
    delta_k = np.fft.rfftn(overdensity_field)
    delta_k = delta_k * np.exp(1j * np.pi) if counter else delta_k
    mask = kmod > 0.0  # Avoid division by zero at k=0
    xpert = np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x, dtype=np.float32)
    s = (nmesh, nmesh, nmesh)  # Shape of the displacement field along each axis

    for i, xi in enumerate(('x', 'y', 'z')):
        psi1_ki = np.zeros_like(kmod, dtype=complex)
        psi1_ki[mask] = -1j * kvec[i, mask] / (kmod[mask] ** 2) * delta_k[mask]
        disp_field = np.fft.irfftn(psi1_ki, s=s)
        max_disp = np.max(np.abs(disp_field))
        print(f"Maximal '{xi}' displacement: {max_disp*1000:.3f} kpc/h; "
              f"in units of mean particle separation: {max_disp * nmesh / Lbox:.3f}")
        disp_field_interp = interpolate_field(x, Lbox, disp_field)
        xpert[:, i] = x[:, i] + disp_field_interp
        v[:, i] = disp_field_interp * dD1
    return np.mod(xpert, Lbox), v / h


def twolpt(x, density_field, Lbox, dD1, dD2, h, counter=False):
    r'''
    Perform second-order Lagrangian perturbation theory (2LPT) to generate
    perturbed particle positions and velocities.

    In 2LPT the Eulerian position $\mathbf{x}$ is given by

    .. math::
        \mathbf{x}(\mathbf{q}, t) =
            \mathbf{q}
            + D_1(t)\,\mathbf{\Psi}^{(1)}(\mathbf{q})
            + D_2(t)\,\mathbf{\Psi}^{(2)}(\mathbf{q}),

    where:
      - $\mathbf{q}$ denotes the initial (Lagrangian) positions,
      - $D_1(t)$ is the linear growth factor (first order),
      - $D_2(t)$ is the second-order growth factor,
      - $\mathbf{\Psi}^{(1)}$ is the first-order displacement field,
      - $\mathbf{\Psi}^{(2)}$ is the second-order displacement field.

    In many implementations the displacement fields are normalized so
    that the computed displacements already include the growth factors.
    Here, it is done in a way that $D_1$ and $D_2$ are normalized to
    $1$ at $a=1$. In our convention, the first-order displacement
    (computed as in the Zel'dovich approximation) is taken to represent
    $D_1\,\mathbf{\Psi}^{(1)}$, and the second-order displacement will
    represent $D_2\,\mathbf{\Psi}^{(2)}$.
    
    The velocities are then given by

    .. math::
        \mathbf{v} =
            \dot{D}_1\,\mathbf{\Psi}^{(1)} + \dot{D}_2\,\mathbf{\Psi}^{(2)},
    
    where $\dot{D}_1$ and $\dot{D}_2$ (passed as ``dD1`` and ``dD2``)
    are the time derivatives of the growth factors.

    **Overview of the implementation:**

    1. **Fourier grid setup:**
       - A 3D Fourier space grid is constructed (via ``fourier_grid()``)
         for a mesh of size $n_{\rm mesh}$ in a box of size $L_{\rm box}$.
         For a real field, the third dimension is handled using a reduced
         FFT (Hermitian symmetry) so that the k-space array has shape
         $(3, n_{\rm mesh}, n_{\rm mesh}, n_{\rm mesh}//2+1)$.

    2. **First-order displacement field ($\mathbf{\Psi}^{(1)}$):**
       - The overdensity field is Fourier-transformed using ``np.fft.rfftn``
         to obtain $\delta(\mathbf{k})$.
       - For each spatial component $i$, the Fourier-space displacement
         is computed as

         .. math::
             \Psi^{(1)}_i(\mathbf{k}) =
                -i\,\frac{k_i}{|\mathbf{k}|^2}\,\delta(\mathbf{k}),
             \quad |\mathbf{k}| > 0.
         
       - An inverse FFT (``np.fft.irfftn``) yields the real-space first-
         order displacement field, which already includes the factor $D_1$.

    3. **Second-Order Source Term:**
       - To compute the second-order displacement field, we first need
         to construct a quadratic source term $S(\mathbf{x})$ from the
         spatial derivatives of the first-order displacement.
       - Following standard 2LPT theory, one may compute

         .. math::
             S(\mathbf{x}) =
               \frac{\partial \Psi^{(1)}_x}{\partial x}\,\frac{\partial \Psi^{(1)}_y}{\partial y}
             + \frac{\partial \Psi^{(1)}_x}{\partial x}\,\frac{\partial \Psi^{(1)}_z}{\partial z}
             + \frac{\partial \Psi^{(1)}_y}{\partial y}\,\frac{\partial \Psi^{(1)}_z}{\partial z}
             - \left[
                 \left(\frac{\partial \Psi^{(1)}_x}{\partial y}\right)^2
               + \left(\frac{\partial \Psi^{(1)}_x}{\partial z}\right)^2
               + \left(\frac{\partial \Psi^{(1)}_y}{\partial z}\right)^2
               \right].
         
       - In the code we denote:
         - $dPxx = \frac{\partial \Psi^{(1)}_x}{\partial x}$,
         - $dPxy = \frac{\partial \Psi^{(1)}_x}{\partial y}$,
         - $dPxz = \frac{\partial \Psi^{(1)}_x}{\partial z}$,
         - $dPyy = \frac{\partial \Psi^{(1)}_y}{\partial y}$,
         - $dPyz = \frac{\partial \Psi^{(1)}_y}{\partial z}$,
         - $dPzz = \frac{\partial \Psi^{(1)}_z}{\partial z}$,
         
         and then set

         .. math::
             S(\mathbf{x}) =
                 dPxx\,dPyy + dPxx\,dPzz + dPyy\,dPzz - (dPxy^2 + dPxz^2 + dPyz^2).
         
       - The source $S(\mathbf{x})$ is Fourier-transformed to $S(\mathbf{k})$.

    4. **Second-Order Displacement Field ($\mathbf{\Psi}^{(2)}$):**
       - The Poisson equation in Fourier space is solved for the second-
         order potential:

         .. math::
             \phi^{(2)}(\mathbf{k}) = -\frac{S(\mathbf{k})}{|\mathbf{k}|^2},
         
         with the $k=0$ mode appropriately masked.
       - The second-order displacement in Fourier space is then given by

         .. math::
             \Psi^{(2)}_i(\mathbf{k}) = i\,k_i\,\phi^{(2)}(\mathbf{k}),
         
         and an inverse FFT yields the real-space second-order displacement
         field.
       - In an Einstein-de Sitter Universe, the second-order growth factor
         is $D_2 = -\frac{3}{7}\,D_1^2$. Thus, the final update is
         performed as

         .. math::
             \mathbf{x}_{\rm pert} =
                \mathbf{q} + \mathbf{\Psi}^{(1)} - \frac{3}{7}\,\mathbf{\Psi}^{(2)},
         
         and similarly for the velocity field:

         .. math::
              \mathbf{v} =
                             \dot{D}_1\,\mathbf{\Psi}^{(1)}
              - \frac{3}{7}\,\dot{D}_2\,\mathbf{\Psi}^{(2)}.
    
    5. **Interpolation and Particle Update:**
       - Both the first- and second-order displacement fields are defined
         on the FFT grid. They are interpolated to the particle positions
         (which generally do not lie exactly on grid points) using a
         trilinear (or similar) interpolation routine.
       - Finally, positions are updated and wrapped periodically, and
         velocities are rescaled by the dimensionless Hubble parameter $h$.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Initial unperturbed particle positions (Lagrangian coordinates),
        typically in Mpc/h.
    density_field : ndarray of shape (M, M, M)
        Real-space density field from which the overdensity field is
        computed.
    Lbox : float
        The size of the simulation box in Mpc/h. Periodic boundary
        conditions are assumed.
    dD1 : float
        The time derivative of the linear growth factor, $\dot{D}_1$
        (in km/s/Mpc), used to compute the first-order velocity contribution.
    dD2 : float
        The time derivative of the second-order growth factor, $\dot{D}_2$
        (in km/s/Mpc), used to compute the second-order velocity contribution.
    h : float
        Dimensionless Hubble parameter, $h = H_0/100$, where $H_0$
        is the Hubble constant in km/s/Mpc.
    counter : bool, optional
        If True, apply a $\pi$ radian global phase shift to the density
        field before computing the displacement field. This is useful
        to run counter-phased simulations. The default is False. 
        

    Returns
    -------
    xpert : ndarray of shape (N, 3)
        Perturbed particle positions (Eulerian coordinates) after
        applying 2LPT. Positions are wrapped periodically within the
        simulation box.
    v : ndarray of shape (N, 3)
        Particle velocities in km/s, computed as

        .. math::
            \mathbf{v} =  
                           \dot{D}_1\,\mathbf{\Psi}^{(1)}
            - \frac{3}{7}\,\dot{D}_2\,\mathbf{\Psi}^{(2)}.

    Examples
    --------
    >>> import numpy as np
    >>> # Box size, particle coordinates, density field, growth rates, and H0
    >>> Lbox = 100.0  # Mpc/h
    >>> x = np.random.rand(1000, 3).astype(np.float32) * Lbox  # Coords in Mpc/h
    >>> density_field = np.random.randn(64, 64, 64).astype(np.float32)
    >>> dD1 = 10.0   # km/s/Mpc (first-order time derivative)
    >>> dD2 = 1.0    # km/s/Mpc (second-order time derivative)
    >>> h = 67.6 / 100  # Dimensionless Hubble parameter
    >>> xpert, v = twolpt(x, density_field, Lbox, dD1, dD2, h)
    
    Notes
    -----
    - The second-order source term is computed from specific derivatives
      of the first-order displacement field, following the standard 2LPT
      formulation. A canonical $3/7$ factor is applied to the first-
      order term to obtain the second-order growth factor.
    - The interpolation step is needed since the displacement field is
      computed on a regular grid, but particle positions are generally
      not located exactly on grid points.
    '''
    nmesh = density_field.shape[0]
    overdensity_field = compute_overdensity(density_field)
    kvec, kmod = fourier_grid(nmesh, Lbox, hermitian=True)
    delta_k = np.fft.rfftn(overdensity_field)
    delta_k = delta_k * np.exp(1j * np.pi) if counter else delta_k
    mask = kmod > 0.0  # Avoid division by zero at k=0
    xpert = np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x, dtype=np.float32)
    s = (nmesh, nmesh, nmesh) # Shape of the displacement field along each axis

    # ------------------------------
    # 1. First-order displacement (Psi^(1))
    # ------------------------------
    psi1_k = []  # Store Fourier-space first-order displacement for each axis
    disp1 = []   # Real-space first-order displacement fields
    for i, xi in enumerate(('x', 'y', 'z')):
        psi1_ki = np.zeros_like(kmod, dtype=complex)
        psi1_ki[mask] = -1j * kvec[i, mask] / (kmod[mask]**2) * delta_k[mask]
        psi1_k.append(psi1_ki)
        disp_field = np.fft.irfftn(psi1_ki, s=s)
        max_disp = np.max(np.abs(disp_field))
        print(f"Maximal '{xi}' 1st-order displacement: {max_disp*1000:.3f} kpc/h; "
              f"in units of mean particle separation: {max_disp * nmesh / Lbox:.3f}")
        disp1.append(disp_field)
    disp1 = np.array(disp1)  # Shape: (3, nmesh, nmesh, nmesh)

    # ------------------------------
    # 2. Compute derivatives of Psi^(1) for the second-order source
    # ------------------------------
    # The derivative of the i-th component of the first-order displacement
    # Psi^(1) with respect to the j-th coordinate in Fourier space is given by
    #
    #     d[Psi^(1)_i]/dx_j = irfftn(1j * kvec[j] * psi1_k[i])
    #
    dPxx = np.fft.irfftn(1j * kvec[0] * psi1_k[0], s=s)  # d(Psi_x)/dx
    dPxy = np.fft.irfftn(1j * kvec[1] * psi1_k[0], s=s)  # d(Psi_x)/dy
    dPxz = np.fft.irfftn(1j * kvec[2] * psi1_k[0], s=s)  # d(Psi_x)/dz
    # --
    dPyy = np.fft.irfftn(1j * kvec[1] * psi1_k[1], s=s)  # d(Psi_y)/dy
    dPyz = np.fft.irfftn(1j * kvec[2] * psi1_k[1], s=s)  # d(Psi_y)/dz
    # --
    dPzz = np.fft.irfftn(1j * kvec[2] * psi1_k[2], s=s)  # d(Psi_z)/dz

    # Compute the quadratic source S(x) and its Fourier transform S(k)
    S = dPxx * dPyy + dPxx * dPzz + dPyy * dPzz - (dPxy**2 + dPxz**2 + dPyz**2)
    S_k = np.fft.rfftn(S)

    # ------------------------------
    # 3. Second-order displacement (Psi^(2))
    # ------------------------------
    # Solve the Poisson equation in Fourier space:
    # phi2(k) = -S(k) / |k|^2
    phi2_k = np.zeros_like(S_k, dtype=complex)
    phi2_k[mask] = -S_k[mask] / (kmod[mask]**2)

    # Compute the second-order displacement field:
    # Psi2_i(k) = i * k_i * phi2(k), then inverse FFT to obtain real space.
    disp2 = []  # Real-space second-order displacement fields
    for i, xi in enumerate(('x', 'y', 'z')):
        psi2_k = np.zeros_like(S_k, dtype=complex)
        psi2_k[mask] = 1j * kvec[i, mask] * phi2_k[mask]
        disp_field = np.fft.irfftn(psi2_k, s=s)
        max_disp = np.max(np.abs(disp_field))
        print(f"Maximal '{xi}' 2nd-order displacement: {max_disp*1000:.3f} kpc/h; "
              f"in units of mean particle separation: {max_disp * nmesh / Lbox:.3f}")
        disp2.append(disp_field)
    disp2 = np.array(disp2)  # Shape: (3, nmesh, nmesh, nmesh)

    # ------------------------------
    # 4. Interpolate and update particle positions and velocities
    # ------------------------------
    # For each spatial axis, interpolate the displacement fields (both
    # first- and second-order) from the grid to the particle positions.
    for i in range(3):
        disp1_interp = interpolate_field(x, Lbox, disp1[i])
        disp2_interp = interpolate_field(x, Lbox, disp2[i])
        xpert[:, i] = x[:, i] + disp1_interp - (3.0/7.0) * disp2_interp
        v[:, i] = disp1_interp * dD1 - (3.0/7.0) * disp2_interp * dD2
    return np.mod(xpert, Lbox), v / h