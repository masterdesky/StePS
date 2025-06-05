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
        The grid-based field (e.g. a displacement field) defined on a
        regular grid.
    Lbox : float
        The simulation box size.
    method : str
        The interpolation method to use. This can be 'linear', 'nearest',
        or 'cubic'.

    Returns
    -------
    interp_values : ndarray of shape (N,)
        Field values interpolated at the particle positions.
    '''
    if not isinstance(Lbox, (list, tuple)):
        Lbox = (Lbox,) * 3
    nvox = field.shape
    dk = np.array(Lbox) / np.array(nvox)  # Assuming cubical voxels
    mesh = [np.linspace(di/2, L-di/2, n, endpoint=True) for L, n, di in zip(Lbox, nvox, dk)]
    interpolator = RegularGridInterpolator(
        tuple(mesh),
        field,
        method=method,
        bounds_error=False,
        fill_value=None  # Extrapolate using periodic wrapping if needed
    )
    return interpolator(np.mod(x, Lbox))


def make_density_field():
    r'''TODO
    Generate a Gaussian random overdensity field on a regular grid with
    periodic boundaries, sampled from $P(k)$ at that resolution.

    Parameters
    ----------

    Returns
    -------
    density_field : ndarray of shape (Nx, Ny, Nz)
        The generated density field on a regular grid.
    '''
    raise NotImplementedError


def compute_density_field():
    r'''TODO
    Compute the density field from particle positions using a ... method.

    Parameters
    ----------

    Returns
    -------
    density_field : ndarray of shape (Nx, Ny, Nz)
        The computed density field on a regular grid.
    '''
    raise NotImplementedError


def compute_overdensity(density_field):
    r'''
    Compute the overdensity field :math:`\delta(\mathbf{x})` from the
    given density field.

    Parameters
    ----------
    density_field : ndarray
        The density field (e.g., particle counts or mass density on a
        grid).

    Returns
    -------
    overdensity_field : ndarray
        The overdensity field, defined as
        .. math::
            \delta(\mathbf{x}) = \frac{\rho(\mathbf{x}) - \bar{\rho}}{\bar{\rho}}.
    '''
    mean_density = np.mean(density_field)
    return (density_field - mean_density) / mean_density


def cubic_voxels(nmesh, Lbox, silent=False):
    '''
    Defines a rectangular cuboid mesh with the specified number of
    voxels in each dimensions, ensuring that the voxels are cubic.
    The function calculates the number of voxels in each dimension
    (Nx, Ny, Nz) based on the shortest dimension of the cuboid and
    scales the other dimensions accordingly.

    Parameters
    ----------
    nmesh : int
        The number of voxels along the shortest dimension of the mesh.
    Lbox : float or list of float
        The size of the simulation box in comoving Mpc/h. If a single
        float is provided, it is assumed to be a cubic box with equal
        dimensions. If a list is provided, it should contain three
        values representing the box size in each dimension (Lx, Ly, Lz).
    silent : bool
        If True, suppresses output messages.

    Returns
    -------
    dk : float
        The uniform step size in each dimension, calculated as the length
        of the shortest dimension divided by the number of voxels in
        that dimension.
    mesh : tuple of int
        A tuple containing the number of voxels in each dimension (Nx, Ny, Nz).
    '''
    if not isinstance(Lbox, (list, tuple)):
        Lbox = (Lbox,) * 3
    ref_L = np.min(Lbox)
    mesh = np.ceil(Lbox / (ref_L / nmesh)).astype(int)
    mesh = (mesh + mesh % 2).astype(int)  # Ensure even number of voxels
    if not silent:
        print('Mesh: Nx={}, Ny={}, Nz={}'.format(*mesh))
    dk = ref_L / mesh[Lbox.index(ref_L)]
    if not silent:
        print(f'Step size: {dk}')
    return dk, mesh


def fourier_grid(nmesh, Lbox, hermitian=False, silent=False):
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
    
    When ``hermitian`` is True, the output grid reflects the storage
    scheme of a real FFT, and the shape of ``kvec`` is
    :math:`(3, {\rm nmesh}, {\rm nmesh}, {\rm nmesh}//2+1)`. Otherwise,
    a full grid with shape :math:`(3, {\rm nmesh}, {\rm nmesh}, {\rm nmesh})`
    is returned.

    Parameters
    ----------
    nmesh : int
        The number of voxels along the shortest dimension of the mesh.
    Lbox : float or list of float
        The size of the simulation box in comoving Mpc/h. If a single
        float is provided, it is assumed to be a cubic box with equal
        dimensions. If a list is provided, it should contain three
        values representing the box size in each dimension (Lx, Ly, Lz).
    hermitian : bool, optional
        If True, assume the field has Hermitian symmetry (i.e., it is
        real-valued) and use the reduced FFT along the last dimension.
        The default is False.

    Returns
    -------
    kvec : ndarray
        A three-dimensional array of wavevector components with shape:
          - :math:`(3, {\rm nmesh}, {\rm nmesh}, {\rm nmesh}//2+1)` if
          ``hermitian`` is True.
          - :math:`(3, {\rm nmesh}, {\rm nmesh}, {\rm nmesh})` if
          ``hermitian`` is False.
        Each sub-array corresponds to the ``x``, ``y``, or ``z`` component
        of the wavevector.
    kmod : ndarray
        The magnitude of the wavevector at each grid point, computed as
        :math:`\|\mathbf{k}\| = \sqrt{k_x^2 + k_y^2 + k_z^2}`.
    '''
    dk, nvox = cubic_voxels(nmesh, Lbox, silent=silent)
    
    kx = np.fft.fftfreq(nvox[0]) * 2*np.pi / dk
    ky = np.fft.fftfreq(nvox[1]) * 2*np.pi / dk
    if hermitian:
        kz = np.fft.rfftfreq(nvox[2]) * 2*np.pi / dk
    else:
        kz = np.fft.fftfreq(nvox[2]) * 2*np.pi / dk
    kvec = np.array(np.meshgrid(kx, ky, kz, indexing='ij'))
    kmod = np.linalg.norm(kvec, axis=0)
    return kvec, kmod


def lpt1(x, Lbox, density_field, D1, dD1, h, counter=False, silent=False):
    r'''
    Apply first-order Lagrangian Perturbation Theory (LPT), i.e., the
    Zel'dovich approximation, to generate perturbed particle positions
    and velocities.

    In this approximation, particles are displaced from their initial
    (Lagrangian) positions :math:`\mathbf{q}` to their final (Eulerian)
    positions :math:`\mathbf{x}` using a displacement field
    :math:`\mathbf{\Psi}`:

    .. math::
        \mathbf{x}(\mathbf{q}, t) = \mathbf{q} + D_1(t) \, \mathbf{\Psi}(\mathbf{q}),

    where :math:`D_1(t)` is the linear growth factor and :math:`\mathbf{\Psi}(\mathbf{q})`
    is the displacement field computed from the initial density
    perturbations. The displacement field is related to the gravitational
    potential, and in Fourier space, it is calculated from the
    overdensity field :math:`\delta(\mathbf{k})`:

    .. math::
        \mathbf{\Psi}(\mathbf{k}) =
            -i \, \frac{\mathbf{k}}{|\mathbf{k}|^2} \, \delta(\mathbf{k})
        \quad \text{for } |\mathbf{k}| > 0,

    where :math:`\mathbf{k}` is the wavevector.

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Initial unperturbed particle positions (Lagrangian coordinates),
        in comoving Mpc/h.
    Lbox : float or list of float
        The size of the simulation box in comoving Mpc/h. If a single
        float is provided, it is assumed to be a cubic box with equal
        dimensions. If a list is provided, it should contain three
        values representing the box size in each dimension (Lx, Ly, Lz).
    density_field : ndarray of shape (Nx, Ny, Nz)
        Real-space density field on a uniform grid, from which the
        overdensity field is computed.
    D1 : float
        The linear growth factor :math:`D_1(t)` at the desired output time.
    dD1 : float
        A prefactor for the velocity calculation, typically related to the
        time derivative of the growth factor (e.g., :math:`\dot{D}_1` or
        :math:`H(a)f(a)` where f is the growth rate). The code uses this
        in a non-standard velocity formula.
    h : float
        The dimensionless Hubble parameter, :math:`h = H_0 / 100`, where
        :math:`H_0` is the Hubble constant.
    counter : bool
        If True, applies a global sign flip to the Fourier-space density
        field (equivalent to a :math:`\pi` phase shift). This is useful
        for running "counter-phased" simulations to reduce sample
        variance.
    silent : bool
        If True, suppresses output messages.

    Returns
    -------
    xpert : ndarray of shape (N, 3)
        Perturbed particle positions (Eulerian coordinates) in comoving
        Mpc/h. Positions are wrapped to lie within the periodic box.
    v : ndarray of shape (N, 3)
        Particle peculiar velocities in km/s. The velocity is computed
        according to the specific formula implemented in this function:

        .. math::
            \mathbf{v} = \frac{D_1(t) \cdot dD_1 \cdot \mathbf{\Psi}}{h}.

    Notes
    -----
    **Overview of the Implementation:**

    1.  **Fourier Grid Setup:** A 3D Fourier grid (``kvec``) is constructed
        for a mesh of size ``(Nx, Ny, Nz)``. For a real-valued field, a
        reduced FFT is used (Hermitian symmetry), so the k-space arrays
        have a shape of ``(Nx, Ny, Nz//2 + 1)``.

    2.  **Displacement Field Calculation:**
        - The overdensity field is computed from ``density_field`` and
          then Fourier-transformed to get :math:`\delta(\mathbf{k})`.
        - The Fourier-space displacement field :math:`\mathbf{\Psi}(\mathbf{k})`
          is computed for each spatial component. Division by zero at the
          DC mode (:math:`|\mathbf{k}| = 0`) is avoided.
        - An inverse FFT converts :math:`\mathbf{\Psi}(\mathbf{k})` back
          to a real-space grid.

    3.  **Interpolation and Particle Update:**
        - The gridded displacement field :math:`\mathbf{\Psi}` is
          interpolated to the Lagrangian particle positions :math:`\mathbf{q}`.
        - Particle positions are updated to their Eulerian coordinates:

          .. math::
              \mathbf{x}_{\text{pert}} = \mathbf{q} + D_1(t) \, \mathbf{\Psi}(\mathbf{q})

        - Particle velocities are computed using the interpolated
          displacement field :math:`\mathbf{\Psi}`, the growth factor
          ``D1``, and the velocity prefactor ``dD1``. The final result
          is divided by ``h`` to obtain units of km/s.
    '''
    nvox = density_field.shape
    overdensity_field = compute_overdensity(density_field)
    kvec, kmod = fourier_grid(np.min(nvox), Lbox, hermitian=True, silent=silent)
    delta_k = np.fft.rfftn(overdensity_field)
    delta_k = delta_k * np.exp(1j * np.pi) if counter else delta_k
    mask = kmod > 0.0  # Avoid division by zero at k=0
    xpert = np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x, dtype=np.float32)

    for i, xi in enumerate(('x', 'y', 'z')):
        psi1_ki = np.zeros_like(kmod, dtype=complex)
        psi1_ki[mask] = -1j * kvec[i, mask] / (kmod[mask]**2) * delta_k[mask]
        disp_field = np.fft.irfftn(psi1_ki, s=nvox, axes=(0, 1, 2))
        max_disp = np.max(np.abs(disp_field))
        if not silent:
            print(f"Maximal '{xi}' displacement: {max_disp*1e3:.3f} kpc/h; "
                  f"in units of mean particle separation: {max_disp * nvox[i] / Lbox[i]:.3f}")
        disp_field_interp = interpolate_field(x, disp_field, Lbox)
        xpert[:, i] = x[:, i] + D1 * disp_field_interp
        v[:, i] = disp_field_interp * D1 * dD1
    return np.mod(xpert, Lbox), v / h


def lpt2(x, Lbox, density_field, D1, dD1, D2, dD2, h, counter=False, silent=False):
    r'''
    Apply second-order Lagrangian Perturbation Theory (2LPT) to generate
    perturbed particle positions and velocities.

    This method provides a more accurate description of particle
    trajectories than the Zel'dovich approximation (1LPT) by including
    the second-order term in the displacement. The final (Eulerian)
    position :math:`\mathbf{x}` is computed from the initial (Lagrangian)
    position :math:`\mathbf{q}` as:

    .. math::
        \mathbf{x}(\mathbf{q}, t) = \mathbf{q} - D_1(t)\mathbf{\Psi}^{(1)}(\mathbf{q}) + D_2(t)\mathbf{\Psi}^{(2)}(\mathbf{q})

    where :math:`\mathbf{\Psi}^{(1)}` and :math:`\mathbf{\Psi}^{(2)}`
    are the first- and second-order displacement fields, and :math:`D_1`
    and :math:`D_2` are the corresponding linear and second-order growth
    factors.

    The **first-order field** is calculated from the overdensity :math:`\delta`
    as in 1LPT:
    
    .. math::
        \mathbf{\Psi}^{(1)}(\mathbf{k}) = -i \frac{\mathbf{k}}{|\mathbf{k}|^2} \delta(\mathbf{k}).

    The **second-order field** is derived from a scalar potential :math:`\phi^{(2)}`,
    where :math:`\mathbf{\Psi}^{(2)} = -\nabla\phi^{(2)}`. The potential
    itself is sourced by a quadratic source term :math:`S(\mathbf{x})`
    from the spatial derivatives of the first-order displacement.
    Following standard 2LPT theory, one may compute

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
         
    In the code we denote:
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
        
    The Poisson equation in Fourier space is solved for the second-order
    potential:

    .. math::
        \phi^{(2)}(\mathbf{k}) = -\frac{S(\mathbf{k})}{|\mathbf{k}|^2},
         
    with the $k=0$ mode appropriately masked. The second-order displacement
    in Fourier space is then given by

    .. math::
        \Psi^{(2)}_i(\mathbf{k}) = i\,k_i\,\phi^{(2)}(\mathbf{k}),
         
    and an inverse FFT yields the real-space second-order displacement
    field.
    

    Parameters
    ----------
    x : ndarray of shape (N, 3)
        Initial unperturbed particle positions (Lagrangian coordinates),
        in comoving Mpc/h.
    Lbox : float or list of float
        The size of the simulation box in comoving Mpc/h. If a single
        float is provided, it is assumed to be a cubic box with equal
        dimensions. If a list is provided, it should contain three
        values representing the box size in each dimension (Lx, Ly, Lz).
    density_field : ndarray of shape (Nx, Ny, Nz)
        Real-space density field on a uniform grid, from which the
        overdensity field is computed.
    D1 : float
        The linear growth factor :math:`D_1(t)`.
    dD1 : float
        A prefactor for the first-order velocity term.
    D2 : float
        The second-order growth factor :math:`D_2(t)`.
    dD2 : float
        A prefactor for the second-order velocity term.
    h : float
        The dimensionless Hubble parameter, :math:`h = H_0 / 100`.
    counter : bool
        If True, applies a global sign flip to the Fourier-space density
        field (equivalent to a :math:`\pi` phase shift). This is useful
        for running "counter-phased" simulations to reduce sample
        variance.
    silent : bool
        If True, suppresses output messages.

    Returns
    -------
    xpert : ndarray of shape (N, 3)
        Perturbed particle positions (Eulerian coordinates) in comoving
        Mpc/h, wrapped within the periodic box.
    v : ndarray of shape (N, 3)
        Particle peculiar velocities in km/s, computed as:

        .. math::
            \mathbf{v} = \frac{-D_1 \cdot dD_1 \cdot \mathbf{\Psi}^{(1)} + D_2 \cdot dD_2 \cdot \mathbf{\Psi}^{(2)}}{h}.

    Notes
    -----
    **Overview of the Implementation:**

    1.  **First-Order Displacement:** The first-order displacement field,
        :math:`\mathbf{\Psi}^{(1)}`, is computed from the Fourier-space
        overdensity field, identical to the Zel'dovich approximation.

    2.  **Second-Order Source:** The spatial derivatives of the first-order
        field (i.e., the deformation tensor :math:`\partial \Psi_i^{(1)} / \partial q_j`)
        are calculated using FFTs. These are then combined to form the
        source term for the second-order potential.

    3.  **Second-Order Displacement:** The Poisson equation for the
        second-order potential :math:`\phi^{(2)}` is solved in Fourier
        space. The second-order displacement field, :math:`\mathbf{\Psi}^{(2)}`,
        is then found by taking the gradient of this potential, again in
        Fourier space.

    4.  **Interpolation and Particle Update:**
        - Both the first- and second-order displacement fields are
          interpolated from the grid to the Lagrangian particle positions.
        - Particle positions and velocities are updated by combining the
          interpolated first- and second-order contributions.

    .. warning::
        The update equations in this implementation use a specific sign
        convention (:math:`-D_1\mathbf{\Psi}^{(1)} + D_2\mathbf{\Psi}^{(2)}`).
        Ensure this is consistent with the definitions of the growth
        factors and displacement fields used in your analysis.
    '''
    nvox = density_field.shape
    overdensity_field = compute_overdensity(density_field)
    kvec, kmod = fourier_grid(np.min(nvox), Lbox, hermitian=True, silent=silent)
    delta_k = np.fft.rfftn(overdensity_field)
    delta_k = delta_k * np.exp(1j * np.pi) if counter else delta_k
    mask = kmod > 0.0  # Avoid division by zero at k=0
    xpert = np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x, dtype=np.float32)
    axes = (0, 1, 2)  # Axes for the FFT operations

    # ------------------------------
    # 1. First-order displacement (Psi^(1))
    # ------------------------------
    psi1_k = []  # Store Fourier-space first-order displacement for each axis
    disp1 = []   # Real-space first-order displacement fields
    for i, xi in enumerate(('x', 'y', 'z')):
        psi1_ki = np.zeros_like(kmod, dtype=complex)
        psi1_ki[mask] = -1j * kvec[i, mask] / (kmod[mask]**2) * delta_k[mask]
        psi1_k.append(psi1_ki)
        disp_field = np.fft.irfftn(psi1_ki, s=nvox, axes=axes)
        max_disp = np.max(np.abs(disp_field))
        if not silent:
            print(f"Maximal '{xi}' 1st-order displacement: {max_disp*1e3:.3f} kpc/h; "
                f"in units of mean particle separation: {max_disp * nvox[i] / Lbox[i]:.3f}")
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
    dPxx = np.fft.irfftn(1j * kvec[0] * psi1_k[0], s=nvox, axes=axes)  # d(Psi_x)/dx
    dPxy = np.fft.irfftn(1j * kvec[1] * psi1_k[0], s=nvox, axes=axes)  # d(Psi_x)/dy
    dPxz = np.fft.irfftn(1j * kvec[2] * psi1_k[0], s=nvox, axes=axes)  # d(Psi_x)/dz
    # --
    dPyy = np.fft.irfftn(1j * kvec[1] * psi1_k[1], s=nvox, axes=axes)  # d(Psi_y)/dy
    dPyz = np.fft.irfftn(1j * kvec[2] * psi1_k[1], s=nvox, axes=axes)  # d(Psi_y)/dz
    # --
    dPzz = np.fft.irfftn(1j * kvec[2] * psi1_k[2], s=nvox, axes=axes)  # d(Psi_z)/dz

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
        disp_field = np.fft.irfftn(psi2_k, s=nvox, axes=axes)
        max_disp = np.max(np.abs(disp_field))
        if not silent:
            print(f"Maximal '{xi}' 2nd-order displacement: {max_disp*1e3:.3f} kpc/h; "
                f"in units of mean particle separation: {max_disp * nvox[i] / Lbox[i]:.3f}")
        disp2.append(disp_field)
    disp2 = np.array(disp2)  # Shape: (3, nmesh, nmesh, nmesh)

    # ------------------------------
    # 4. Interpolate and update particle positions and velocities
    # ------------------------------
    # For each spatial axis, interpolate the displacement fields (both
    # first- and second-order) from the grid to the particle positions.
    for i in range(3):
        disp1_interp = interpolate_field(x, disp1[i], Lbox)
        disp2_interp = interpolate_field(x, disp2[i], Lbox)
        xpert[:, i] = x[:, i] - D1 * disp1_interp + D2 * disp2_interp
        v[:, i] = - disp1_interp * D1 * dD1 + disp2_interp * D2 * dD2
    return np.mod(xpert, Lbox), v / h