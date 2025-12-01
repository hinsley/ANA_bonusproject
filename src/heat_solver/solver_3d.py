"""
3D Heat Equation Solver using the Douglas-Gunn ADI scheme (Delta Form).

Solves: U_t = c * (U_xx + U_yy + U_zz) + F(x, y, z, t)

============================================================================
DOUGLAS-GUNN ADI SCHEME (DELTA FORM)
============================================================================
This implementation uses the classic "Delta-form" Douglas-Gunn ADI scheme,
which splits a complex 3D problem into three simpler 1D problems solved
sequentially. The scheme solves for increments (deltas) rather than full
solution values at each fractional step.

  Step 0 (Explicit RHS):
    S = (r_x * delta_x^2 + r_y * delta_y^2 + r_z * delta_z^2) u^n
        + (1/2) * (F^n + F^{n+1})

  Step 1 (X-sweep, solve for Delta_u*):
    (1 - r_x/2 * delta_x^2) Delta_u* = S

  Step 2 (Y-sweep, solve for Delta_u**):
    (1 - r_y/2 * delta_y^2) Delta_u** = Delta_u*

  Step 3 (Z-sweep, solve for Delta_u):
    (1 - r_z/2 * delta_z^2) Delta_u = Delta_u**

  Step 4 (Final update):
    u^{n+1} = u^n + Delta_u

Where:
  r_x = c * dt / dx^2, r_y = c * dt / dy^2, r_z = c * dt / dz^2
  delta_x^2 u_i = u_{i+1} - 2*u_i + u_{i-1} (central difference operator)

This scheme is unconditionally stable and O(dt^2 + dx^2 + dy^2 + dz^2) accurate.

The key functions implementing the directional sweeps are:
  - _douglas_gunn_step_x()
  - _douglas_gunn_step_y()
  - _douglas_gunn_step_z()

Each sweep produces a tridiagonal system that is solved using the Thomas
algorithm (TDMA).
============================================================================
"""

from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .boundary import (
    BoundaryConditions3D,
    BoundaryCondition,
    BCType,
)
from .tridiagonal import solve_tridiagonal


@dataclass
class Domain3D:
    """
    3D rectangular domain specification.

    Parameters
    ----------
    x_min, x_max : float
        Domain bounds in x direction.
    y_min, y_max : float
        Domain bounds in y direction.
    z_min, z_max : float
        Domain bounds in z direction.
    nx : int
        Number of grid points in x direction (including boundaries).
    ny : int
        Number of grid points in y direction (including boundaries).
    nz : int
        Number of grid points in z direction (including boundaries).
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    nx: int
    ny: int
    nz: int

    @property
    def dx(self) -> float:
        """Grid spacing in x."""
        return (self.x_max - self.x_min) / (self.nx - 1)

    @property
    def dy(self) -> float:
        """Grid spacing in y."""
        return (self.y_max - self.y_min) / (self.ny - 1)

    @property
    def dz(self) -> float:
        """Grid spacing in z."""
        return (self.z_max - self.z_min) / (self.nz - 1)

    @property
    def x(self) -> NDArray[np.float64]:
        """1D array of x coordinates."""
        return np.linspace(self.x_min, self.x_max, self.nx)

    @property
    def y(self) -> NDArray[np.float64]:
        """1D array of y coordinates."""
        return np.linspace(self.y_min, self.y_max, self.ny)

    @property
    def z(self) -> NDArray[np.float64]:
        """1D array of z coordinates."""
        return np.linspace(self.z_min, self.z_max, self.nz)

    def meshgrid(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return X, Y, Z meshgrid arrays with indexing='ij'."""
        return np.meshgrid(self.x, self.y, self.z, indexing='ij')


class HeatSolver3D:
    """
    3D Heat equation solver using the Douglas-Gunn ADI scheme (Delta Form).

    Parameters
    ----------
    domain : Domain3D
        The spatial domain specification.
    c : float
        Diffusion coefficient (thermal diffusivity).
    bc : BoundaryConditions3D
        Boundary conditions for all six faces.
    initial_condition : callable
        Function u0(x, y, z) -> initial values. Should accept meshgrid arrays.
    forcing : callable, optional
        Function F(x, y, z, t) -> forcing values. Should accept meshgrid arrays.
        Default is zero forcing.
    """

    def __init__(
        self,
        domain: Domain3D,
        c: float,
        bc: BoundaryConditions3D,
        initial_condition: Callable[[NDArray, NDArray, NDArray], NDArray],
        forcing: Optional[Callable[[NDArray, NDArray, NDArray, float], NDArray]] = None,
    ):
        self.domain = domain
        self.c = c
        self.bc = bc
        self.initial_condition = initial_condition
        self.forcing = forcing if forcing is not None else lambda x, y, z, t: np.zeros_like(x)

        # Compute grid.
        self.X, self.Y, self.Z = domain.meshgrid()

        # Initialize solution array. Shape: (nx, ny, nz).
        self.u = self.initial_condition(self.X, self.Y, self.Z).astype(np.float64)
        self.t = 0.0

        # Apply initial boundary conditions.
        self._apply_boundary_conditions(self.u, 0.0)

    def _apply_boundary_conditions(self, u: NDArray, t: float) -> None:
        """Apply Dirichlet boundary conditions directly to the solution array."""
        y_grid, z_grid = np.meshgrid(self.domain.y, self.domain.z, indexing='ij')
        x_grid_xz, z_grid_xz = np.meshgrid(self.domain.x, self.domain.z, indexing='ij')
        x_grid_xy, y_grid_xy = np.meshgrid(self.domain.x, self.domain.y, indexing='ij')

        # x_min face.
        if self.bc.x_min.bc_type == BCType.DIRICHLET:
            u[0, :, :] = self.bc.x_min.evaluate((y_grid, z_grid), t)

        # x_max face.
        if self.bc.x_max.bc_type == BCType.DIRICHLET:
            u[-1, :, :] = self.bc.x_max.evaluate((y_grid, z_grid), t)

        # y_min face.
        if self.bc.y_min.bc_type == BCType.DIRICHLET:
            u[:, 0, :] = self.bc.y_min.evaluate((x_grid_xz, z_grid_xz), t)

        # y_max face.
        if self.bc.y_max.bc_type == BCType.DIRICHLET:
            u[:, -1, :] = self.bc.y_max.evaluate((x_grid_xz, z_grid_xz), t)

        # z_min face.
        if self.bc.z_min.bc_type == BCType.DIRICHLET:
            u[:, :, 0] = self.bc.z_min.evaluate((x_grid_xy, y_grid_xy), t)

        # z_max face.
        if self.bc.z_max.bc_type == BCType.DIRICHLET:
            u[:, :, -1] = self.bc.z_max.evaluate((x_grid_xy, y_grid_xy), t)

    def _apply_laplacian(self, u: NDArray) -> NDArray:
        """
        Compute the discrete Laplacian of u.

        Uses second-order central differences. Returns zeros at boundaries.
        """
        dx, dy, dz = self.domain.dx, self.domain.dy, self.domain.dz
        lap = np.zeros_like(u)

        # Interior points only.
        lap[1:-1, 1:-1, 1:-1] = (
            (u[2:, 1:-1, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / (dx*dx)
            + (u[1:-1, 2:, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]) / (dy*dy)
            + (u[1:-1, 1:-1, 2:] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2]) / (dz*dz)
        )

        return lap

    def _apply_delta_sq_x(self, u: NDArray) -> NDArray:
        """Apply delta^2_x operator (second difference in x)."""
        result = np.zeros_like(u)
        result[1:-1, :, :] = u[2:, :, :] - 2*u[1:-1, :, :] + u[:-2, :, :]
        return result

    def _apply_delta_sq_y(self, u: NDArray) -> NDArray:
        """Apply delta^2_y operator (second difference in y)."""
        result = np.zeros_like(u)
        result[:, 1:-1, :] = u[:, 2:, :] - 2*u[:, 1:-1, :] + u[:, :-2, :]
        return result

    def _apply_delta_sq_z(self, u: NDArray) -> NDArray:
        """Apply delta^2_z operator (second difference in z)."""
        result = np.zeros_like(u)
        result[:, :, 1:-1] = u[:, :, 2:] - 2*u[:, :, 1:-1] + u[:, :, :-2]
        return result

    def _build_tridiag_coefficients_x(
        self,
        r_x: float,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Build tridiagonal coefficients for x-direction implicit solve."""
        nx = self.domain.nx

        a = np.full(nx, -r_x / 2, dtype=np.float64)
        b = np.full(nx, 1.0 + r_x, dtype=np.float64)
        c = np.full(nx, -r_x / 2, dtype=np.float64)

        # Boundary modifications: use identity (Dirichlet-style) for all BC types.
        # The actual BC values are computed in the sweep functions.
        a[0], b[0], c[0] = 0.0, 1.0, 0.0
        a[-1], b[-1], c[-1] = 0.0, 1.0, 0.0

        return a, b, c

    def _build_tridiag_coefficients_y(
        self,
        r_y: float,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Build tridiagonal coefficients for y-direction implicit solve."""
        ny = self.domain.ny

        a = np.full(ny, -r_y / 2, dtype=np.float64)
        b = np.full(ny, 1.0 + r_y, dtype=np.float64)
        c = np.full(ny, -r_y / 2, dtype=np.float64)

        # Boundary modifications: use identity (Dirichlet-style) for all BC types.
        a[0], b[0], c[0] = 0.0, 1.0, 0.0
        a[-1], b[-1], c[-1] = 0.0, 1.0, 0.0

        return a, b, c

    def _build_tridiag_coefficients_z(
        self,
        r_z: float,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Build tridiagonal coefficients for z-direction implicit solve."""
        nz = self.domain.nz

        a = np.full(nz, -r_z / 2, dtype=np.float64)
        b = np.full(nz, 1.0 + r_z, dtype=np.float64)
        c = np.full(nz, -r_z / 2, dtype=np.float64)

        # Boundary modifications: use identity (Dirichlet-style) for all BC types.
        a[0], b[0], c[0] = 0.0, 1.0, 0.0
        a[-1], b[-1], c[-1] = 0.0, 1.0, 0.0

        return a, b, c

    # ========================================================================
    # DOUGLAS-GUNN ADI SCHEME (DELTA FORM)
    # The three step functions below implement the Douglas-Gunn scheme.
    # Each step solves for an increment (delta) rather than full solution.
    # ========================================================================

    def _douglas_gunn_step_x(
        self,
        u_n: NDArray,
        r_x: float,
        r_y: float,
        r_z: float,
        forcing_term: NDArray,
        t_new: float,
    ) -> NDArray:
        """
        Step 1: X-direction implicit solve for Delta_u*.

        First computes the explicit RHS:
            S = (r_x * delta_x^2 + r_y * delta_y^2 + r_z * delta_z^2) u^n + forcing_term

        Then solves:
            (1 - r_x/2 * delta_x^2) Delta_u* = S

        This gives the tridiagonal system:
            -r_x/2 * Delta_u*_{i-1} + (1 + r_x) * Delta_u*_i - r_x/2 * Delta_u*_{i+1} = S_i

        Parameters
        ----------
        u_n : ndarray
            Solution at time step n.
        r_x, r_y, r_z : float
            Courant-like numbers for each direction.
        forcing_term : ndarray
            Time-averaged forcing: (1/2) * (F^n + F^{n+1}).
        t_new : float
            Time at step n+1 (for boundary conditions).

        Returns
        -------
        du_star : ndarray
            Increment Delta_u* from the X-sweep.
        """
        nx, ny, nz = self.domain.nx, self.domain.ny, self.domain.nz
        dx = self.domain.dx

        # Compute explicit RHS: S = (r_x δ_x² + r_y δ_y² + r_z δ_z²) u^n + forcing.
        delta_sq_x = self._apply_delta_sq_x(u_n)
        delta_sq_y = self._apply_delta_sq_y(u_n)
        delta_sq_z = self._apply_delta_sq_z(u_n)

        rhs = (
            r_x * delta_sq_x
            + r_y * delta_sq_y
            + r_z * delta_sq_z
            + forcing_term
        )

        # Build tridiagonal system and solve along x-lines.
        a, b, c = self._build_tridiag_coefficients_x(r_x)
        du_star = np.zeros_like(u_n)

        for j in range(ny):
            for k in range(nz):
                d = rhs[:, j, k].copy()

                # Boundary conditions in x (delta form).
                # For Dirichlet: Delta_u = u_boundary(t_{n+1}) - u^n at boundary.
                y_coord, z_coord = self.domain.y[j], self.domain.z[k]

                if self.bc.x_min.bc_type == BCType.DIRICHLET:
                    u_bc = self.bc.x_min.evaluate(
                        (np.array([y_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                    d[0] = u_bc - u_n[0, j, k]
                elif self.bc.x_min.bc_type == BCType.NEUMANN:
                    # Neumann ∂u/∂n = g: extrapolate boundary using interior.
                    g_new = self.bc.x_min.evaluate(
                        (np.array([y_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                    # At x_min: ∂u/∂n = (u[0]-u[1])/dx = g => u[0] = u[1] + dx*g
                    # Target: u_bc = u^{n+1}[1] + dx*g ≈ u^n[1] + dx*g (lag interior by one step)
                    u_bc = u_n[1, j, k] + dx * g_new
                    d[0] = u_bc - u_n[0, j, k]
                elif self.bc.x_min.bc_type == BCType.ROBIN:
                    # Robin αu + β∂u/∂n = g: compute boundary value.
                    g_new = self.bc.x_min.evaluate(
                        (np.array([y_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                    alpha, beta = self.bc.x_min.alpha, self.bc.x_min.beta
                    # At x_min: ∂u/∂n = (u[0]-u[1])/dx
                    # αu[0] + β(u[0]-u[1])/dx = g => (α + β/dx)u[0] = g + (β/dx)u[1]
                    # u[0] = (g + (β/dx)*u[1]) / (α + β/dx)
                    u_bc = (g_new + (beta / dx) * u_n[1, j, k]) / (alpha + beta / dx)
                    d[0] = u_bc - u_n[0, j, k]

                if self.bc.x_max.bc_type == BCType.DIRICHLET:
                    u_bc = self.bc.x_max.evaluate(
                        (np.array([y_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                    d[-1] = u_bc - u_n[-1, j, k]
                elif self.bc.x_max.bc_type == BCType.NEUMANN:
                    g_new = self.bc.x_max.evaluate(
                        (np.array([y_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                    # At x_max: ∂u/∂n = (u[-1]-u[-2])/dx = g => u[-1] = u[-2] + dx*g
                    u_bc = u_n[-2, j, k] + dx * g_new
                    d[-1] = u_bc - u_n[-1, j, k]
                elif self.bc.x_max.bc_type == BCType.ROBIN:
                    g_new = self.bc.x_max.evaluate(
                        (np.array([y_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                    alpha, beta = self.bc.x_max.alpha, self.bc.x_max.beta
                    u_bc = (g_new + (beta / dx) * u_n[-2, j, k]) / (alpha + beta / dx)
                    d[-1] = u_bc - u_n[-1, j, k]

                du_star[:, j, k] = solve_tridiagonal(a, b, c, d)

        return du_star

    def _douglas_gunn_step_y(
        self,
        du_star: NDArray,
        u_n: NDArray,
        r_y: float,
        t_new: float,
    ) -> NDArray:
        """
        Step 2: Y-direction implicit solve for Delta_u**.

        Solves:
            (1 - r_y/2 * delta_y^2) Delta_u** = Delta_u*

        The RHS is simply the result from the X-sweep (Delta_u*).

        This gives the tridiagonal system:
            -r_y/2 * Delta_u**_{j-1} + (1 + r_y) * Delta_u**_j - r_y/2 * Delta_u**_{j+1} = Delta_u*_j

        Parameters
        ----------
        du_star : ndarray
            Increment from X-sweep (Delta_u*).
        u_n : ndarray
            Solution at time step n (for boundary conditions).
        r_y : float
            Courant-like number for y direction.
        t_new : float
            Time at step n+1 (for boundary conditions).

        Returns
        -------
        du_dstar : ndarray
            Increment Delta_u** from the Y-sweep.
        """
        nx, ny, nz = self.domain.nx, self.domain.ny, self.domain.nz
        dy = self.domain.dy

        # RHS is simply Delta_u* from the X-sweep.
        rhs = du_star

        # Build tridiagonal system and solve along y-lines.
        a, b, c = self._build_tridiag_coefficients_y(r_y)
        du_dstar = np.zeros_like(du_star)

        for i in range(nx):
            for k in range(nz):
                d = rhs[i, :, k].copy()

                # Boundary conditions in y (delta form).
                # For Dirichlet: Delta_u = u_boundary(t_{n+1}) - u^n at boundary.
                x_coord, z_coord = self.domain.x[i], self.domain.z[k]

                if self.bc.y_min.bc_type == BCType.DIRICHLET:
                    u_bc = self.bc.y_min.evaluate(
                        (np.array([x_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                    d[0] = u_bc - u_n[i, 0, k]
                elif self.bc.y_min.bc_type == BCType.NEUMANN:
                    g_new = self.bc.y_min.evaluate(
                        (np.array([x_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                    u_bc = u_n[i, 1, k] + dy * g_new
                    d[0] = u_bc - u_n[i, 0, k]
                elif self.bc.y_min.bc_type == BCType.ROBIN:
                    g_new = self.bc.y_min.evaluate(
                        (np.array([x_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                    alpha, beta = self.bc.y_min.alpha, self.bc.y_min.beta
                    u_bc = (g_new + (beta / dy) * u_n[i, 1, k]) / (alpha + beta / dy)
                    d[0] = u_bc - u_n[i, 0, k]

                if self.bc.y_max.bc_type == BCType.DIRICHLET:
                    u_bc = self.bc.y_max.evaluate(
                        (np.array([x_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                    d[-1] = u_bc - u_n[i, -1, k]
                elif self.bc.y_max.bc_type == BCType.NEUMANN:
                    g_new = self.bc.y_max.evaluate(
                        (np.array([x_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                    u_bc = u_n[i, -2, k] + dy * g_new
                    d[-1] = u_bc - u_n[i, -1, k]
                elif self.bc.y_max.bc_type == BCType.ROBIN:
                    g_new = self.bc.y_max.evaluate(
                        (np.array([x_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                    alpha, beta = self.bc.y_max.alpha, self.bc.y_max.beta
                    u_bc = (g_new + (beta / dy) * u_n[i, -2, k]) / (alpha + beta / dy)
                    d[-1] = u_bc - u_n[i, -1, k]

                du_dstar[i, :, k] = solve_tridiagonal(a, b, c, d)

        return du_dstar

    def _douglas_gunn_step_z(
        self,
        du_dstar: NDArray,
        u_n: NDArray,
        r_z: float,
        t_new: float,
    ) -> NDArray:
        """
        Step 3: Z-direction implicit solve for Delta_u (final increment).

        Solves:
            (1 - r_z/2 * delta_z^2) Delta_u = Delta_u**

        The RHS is simply the result from the Y-sweep (Delta_u**).

        This gives the tridiagonal system:
            -r_z/2 * Delta_u_{k-1} + (1 + r_z) * Delta_u_k - r_z/2 * Delta_u_{k+1} = Delta_u**_k

        Parameters
        ----------
        du_dstar : ndarray
            Increment from Y-sweep (Delta_u**).
        u_n : ndarray
            Solution at time step n (for boundary conditions).
        r_z : float
            Courant-like number for z direction.
        t_new : float
            Time at step n+1 (for boundary conditions).

        Returns
        -------
        du : ndarray
            Final increment Delta_u from the Z-sweep.
        """
        nx, ny, nz = self.domain.nx, self.domain.ny, self.domain.nz
        dz = self.domain.dz

        # RHS is simply Delta_u** from the Y-sweep.
        rhs = du_dstar

        # Build tridiagonal system and solve along z-lines.
        a, b, c = self._build_tridiag_coefficients_z(r_z)
        du = np.zeros_like(du_dstar)

        for i in range(nx):
            for j in range(ny):
                d = rhs[i, j, :].copy()

                # Boundary conditions in z (delta form).
                # For Dirichlet: Delta_u = u_boundary(t_{n+1}) - u^n at boundary.
                x_coord, y_coord = self.domain.x[i], self.domain.y[j]

                if self.bc.z_min.bc_type == BCType.DIRICHLET:
                    u_bc = self.bc.z_min.evaluate(
                        (np.array([x_coord]), np.array([y_coord])), t_new
                    ).flat[0]
                    d[0] = u_bc - u_n[i, j, 0]
                elif self.bc.z_min.bc_type == BCType.NEUMANN:
                    g_new = self.bc.z_min.evaluate(
                        (np.array([x_coord]), np.array([y_coord])), t_new
                    ).flat[0]
                    u_bc = u_n[i, j, 1] + dz * g_new
                    d[0] = u_bc - u_n[i, j, 0]
                elif self.bc.z_min.bc_type == BCType.ROBIN:
                    g_new = self.bc.z_min.evaluate(
                        (np.array([x_coord]), np.array([y_coord])), t_new
                    ).flat[0]
                    alpha, beta = self.bc.z_min.alpha, self.bc.z_min.beta
                    u_bc = (g_new + (beta / dz) * u_n[i, j, 1]) / (alpha + beta / dz)
                    d[0] = u_bc - u_n[i, j, 0]

                if self.bc.z_max.bc_type == BCType.DIRICHLET:
                    u_bc = self.bc.z_max.evaluate(
                        (np.array([x_coord]), np.array([y_coord])), t_new
                    ).flat[0]
                    d[-1] = u_bc - u_n[i, j, -1]
                elif self.bc.z_max.bc_type == BCType.NEUMANN:
                    g_new = self.bc.z_max.evaluate(
                        (np.array([x_coord]), np.array([y_coord])), t_new
                    ).flat[0]
                    u_bc = u_n[i, j, -2] + dz * g_new
                    d[-1] = u_bc - u_n[i, j, -1]
                elif self.bc.z_max.bc_type == BCType.ROBIN:
                    g_new = self.bc.z_max.evaluate(
                        (np.array([x_coord]), np.array([y_coord])), t_new
                    ).flat[0]
                    alpha, beta = self.bc.z_max.alpha, self.bc.z_max.beta
                    u_bc = (g_new + (beta / dz) * u_n[i, j, -2]) / (alpha + beta / dz)
                    d[-1] = u_bc - u_n[i, j, -1]

                du[i, j, :] = solve_tridiagonal(a, b, c, d)

        return du

    def step(self, dt: float) -> None:
        """
        Advance the solution by one time step using Douglas-Gunn ADI (Delta Form).

        The algorithm proceeds as follows:
          1. Compute explicit RHS: S = (r_x δ_x² + r_y δ_y² + r_z δ_z²) u^n + forcing
          2. X-sweep: Solve (1 - r_x/2 δ_x²) Δu* = S
          3. Y-sweep: Solve (1 - r_y/2 δ_y²) Δu** = Δu*
          4. Z-sweep: Solve (1 - r_z/2 δ_z²) Δu = Δu**
          5. Update: u^{n+1} = u^n + Δu

        Parameters
        ----------
        dt : float
            Time step size.
        """
        dx, dy, dz = self.domain.dx, self.domain.dy, self.domain.dz
        r_x = self.c * dt / (dx * dx)
        r_y = self.c * dt / (dy * dy)
        r_z = self.c * dt / (dz * dz)

        t_n = self.t
        t_np1 = self.t + dt

        # Evaluate forcing at current and next time: (1/2) * (F^n + F^{n+1}).
        F_n = self.forcing(self.X, self.Y, self.Z, t_n)
        F_np1 = self.forcing(self.X, self.Y, self.Z, t_np1)
        forcing_term = 0.5 * (F_n + F_np1)

        # Store u^n for use in all steps.
        u_n = self.u.copy()

        # Step 1: X-sweep (solve for Δu*).
        du_star = self._douglas_gunn_step_x(u_n, r_x, r_y, r_z, forcing_term, t_np1)

        # Step 2: Y-sweep (solve for Δu**).
        du_dstar = self._douglas_gunn_step_y(du_star, u_n, r_y, t_np1)

        # Step 3: Z-sweep (solve for Δu).
        du = self._douglas_gunn_step_z(du_dstar, u_n, r_z, t_np1)

        # Step 4: Final update: u^{n+1} = u^n + Δu.
        self.u = u_n + du
        self.t = t_np1

    def solve(
        self,
        t_final: float,
        dt: float,
        save_every: Optional[int] = None,
        callback: Optional[Callable[[float, NDArray], None]] = None,
    ) -> Tuple[List[float], List[NDArray]]:
        """
        Solve the heat equation from current time to t_final.

        Parameters
        ----------
        t_final : float
            Final time to solve to.
        dt : float
            Time step size.
        save_every : int, optional
            Save solution every N steps. If None, only saves initial and final.
        callback : callable, optional
            Function called at each saved time step: callback(t, u).

        Returns
        -------
        times : list of float
            Times at which solution was saved.
        solutions : list of ndarray
            Solution arrays at saved times.
        """
        times = [self.t]
        solutions = [self.u.copy()]

        n_steps = int(np.ceil((t_final - self.t) / dt))
        save_every = save_every or n_steps

        for step_num in range(1, n_steps + 1):
            actual_dt = min(dt, t_final - self.t)
            if actual_dt <= 0:
                break

            self.step(actual_dt)

            if step_num % save_every == 0 or step_num == n_steps:
                times.append(self.t)
                solutions.append(self.u.copy())

                if callback is not None:
                    callback(self.t, self.u)

        return times, solutions

    def get_solution(self) -> Tuple[NDArray, NDArray, NDArray, NDArray, float]:
        """
        Get the current solution state.

        Returns
        -------
        X : ndarray
            X coordinates meshgrid.
        Y : ndarray
            Y coordinates meshgrid.
        Z : ndarray
            Z coordinates meshgrid.
        u : ndarray
            Current solution values.
        t : float
            Current time.
        """
        return self.X, self.Y, self.Z, self.u, self.t

    def reset(self) -> None:
        """Reset solution to initial condition."""
        self.u = self.initial_condition(self.X, self.Y, self.Z).astype(np.float64)
        self.t = 0.0
        self._apply_boundary_conditions(self.u, 0.0)

    def get_stability_parameters(self, dt: float) -> dict:
        """
        Compute and return stability-related parameters.

        Parameters
        ----------
        dt : float
            Time step size.

        Returns
        -------
        params : dict
            Dictionary with r_x, r_y, r_z, and stability notes.
        """
        dx, dy, dz = self.domain.dx, self.domain.dy, self.domain.dz
        r_x = self.c * dt / (dx * dx)
        r_y = self.c * dt / (dy * dy)
        r_z = self.c * dt / (dz * dz)

        return {
            'r_x': r_x,
            'r_y': r_y,
            'r_z': r_z,
            'dt': dt,
            'dx': dx,
            'dy': dy,
            'dz': dz,
            'c': self.c,
            'note': "Douglas-Gunn ADI scheme is unconditionally stable and "
                    "O(dt^2 + dx^2 + dy^2 + dz^2) accurate for the heat equation.",
        }
