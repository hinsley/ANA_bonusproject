"""
3D Heat Equation Solver using an adapted Peaceman-Rachford ADI scheme.

Solves: U_t = c * (U_xx + U_yy + U_zz) + F(x, y, z, t)

============================================================================
TODO: VERIFY - 3D PEACEMAN-RACHFORD ADAPTATION
============================================================================
This implementation adapts the 2D Peaceman-Rachford scheme to 3D using a
three-step fractional splitting approach (Douglas-Gunn style). The specific
form used here is:

  Step 1 (x-implicit):
    (1 - r_x/2 * delta_x^2) u* = u^n + r/2 * (delta_x^2 + delta_y^2 + delta_z^2) u^n
                           + dt/2 * (F^n + F^{n+1})

  Step 2 (y-implicit):
    (1 - r_y/2 * delta_y^2) u** = u* - r_y/2 * delta_y^2 u^n

  Step 3 (z-implicit):
    (1 - r_z/2 * delta_z^2) u^{n+1} = u** - r_z/2 * delta_z^2 u^n

Where:
  r_x = c * dt / dx^2, r_y = c * dt / dy^2, r_z = c * dt / dz^2

This should be unconditionally stable and O(dt^2 + dx^2 + dy^2 + dz^2) accurate.

If this does not match your derivation, the key functions to modify are:
  - _peaceman_rachford_step_x()
  - _peaceman_rachford_step_y()
  - _peaceman_rachford_step_z()

Each step is isolated and clearly documented for easy modification.
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
    get_bc_coefficients,
    get_bc_rhs_contribution,
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
    3D Heat equation solver using adapted Peaceman-Rachford ADI scheme.

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

        # Boundary modifications.
        a_coef, b_coef, c_coef = get_bc_coefficients(
            self.bc.x_min, self.domain.dx, is_min_boundary=True
        )
        a[0], b[0], c[0] = a_coef, b_coef, c_coef

        a_coef, b_coef, c_coef = get_bc_coefficients(
            self.bc.x_max, self.domain.dx, is_min_boundary=False
        )
        a[-1], b[-1], c[-1] = a_coef, b_coef, c_coef

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

        a_coef, b_coef, c_coef = get_bc_coefficients(
            self.bc.y_min, self.domain.dy, is_min_boundary=True
        )
        a[0], b[0], c[0] = a_coef, b_coef, c_coef

        a_coef, b_coef, c_coef = get_bc_coefficients(
            self.bc.y_max, self.domain.dy, is_min_boundary=False
        )
        a[-1], b[-1], c[-1] = a_coef, b_coef, c_coef

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

        a_coef, b_coef, c_coef = get_bc_coefficients(
            self.bc.z_min, self.domain.dz, is_min_boundary=True
        )
        a[0], b[0], c[0] = a_coef, b_coef, c_coef

        a_coef, b_coef, c_coef = get_bc_coefficients(
            self.bc.z_max, self.domain.dz, is_min_boundary=False
        )
        a[-1], b[-1], c[-1] = a_coef, b_coef, c_coef

        return a, b, c

    # ========================================================================
    # TODO: VERIFY - The three step functions below implement the adapted
    # Peaceman-Rachford scheme. Modify these if your derivation differs.
    # ========================================================================

    def _peaceman_rachford_step_x(
        self,
        u_n: NDArray,
        r_x: float,
        r_y: float,
        r_z: float,
        forcing_term: NDArray,
        t_new: float,
    ) -> NDArray:
        """
        Step 1: X-direction implicit solve.

        TODO: VERIFY this step matches your derivation.

        Solves:
            (1 - r_x/2 * delta_x^2) u* = u^n + r/2 * (delta_x^2 + delta_y^2 + delta_z^2) u^n + forcing_term

        The RHS can be rewritten as:
            u^n + r_x/2 * delta_x^2 u^n + r_y/2 * delta_y^2 u^n + r_z/2 * delta_z^2 u^n + forcing_term
            = (1 + r_x/2 * delta_x^2) u^n + r_y/2 * delta_y^2 u^n + r_z/2 * delta_z^2 u^n + forcing_term
        """
        nx, ny, nz = self.domain.nx, self.domain.ny, self.domain.nz
        dx = self.domain.dx

        # Compute RHS.
        delta_sq_x = self._apply_delta_sq_x(u_n)
        delta_sq_y = self._apply_delta_sq_y(u_n)
        delta_sq_z = self._apply_delta_sq_z(u_n)

        rhs = (
            u_n
            + (r_x / 2) * delta_sq_x
            + (r_y / 2) * delta_sq_y
            + (r_z / 2) * delta_sq_z
            + forcing_term
        )

        # Build tridiagonal system and solve along x-lines.
        a, b, c = self._build_tridiag_coefficients_x(r_x)
        u_star = np.zeros_like(u_n)

        y_grid, z_grid = np.meshgrid(self.domain.y, self.domain.z, indexing='ij')

        for j in range(ny):
            for k in range(nz):
                d = rhs[:, j, k].copy()

                # Boundary conditions in x.
                y_coord, z_coord = self.domain.y[j], self.domain.z[k]

                if self.bc.x_min.bc_type == BCType.DIRICHLET:
                    d[0] = self.bc.x_min.evaluate(
                        (np.array([y_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                elif self.bc.x_min.bc_type in (BCType.NEUMANN, BCType.ROBIN):
                    d[0] = get_bc_rhs_contribution(
                        self.bc.x_min,
                        (np.array([y_coord]), np.array([z_coord])),
                        t_new, dx, True
                    ).flat[0]

                if self.bc.x_max.bc_type == BCType.DIRICHLET:
                    d[-1] = self.bc.x_max.evaluate(
                        (np.array([y_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                elif self.bc.x_max.bc_type in (BCType.NEUMANN, BCType.ROBIN):
                    d[-1] = get_bc_rhs_contribution(
                        self.bc.x_max,
                        (np.array([y_coord]), np.array([z_coord])),
                        t_new, dx, False
                    ).flat[0]

                u_star[:, j, k] = solve_tridiagonal(a, b, c, d)

        return u_star

    def _peaceman_rachford_step_y(
        self,
        u_star: NDArray,
        u_n: NDArray,
        r_y: float,
        t_new: float,
    ) -> NDArray:
        """
        Step 2: Y-direction implicit solve.

        TODO: VERIFY this step matches your derivation.

        Solves:
            (1 - r_y/2 * delta_y^2) u** = u* - r_y/2 * delta_y^2 u^n

        This subtracts out the explicit y-contribution from step 1 and replaces
        it with an implicit treatment.
        """
        nx, ny, nz = self.domain.nx, self.domain.ny, self.domain.nz
        dy = self.domain.dy

        # Compute RHS.
        delta_sq_y_un = self._apply_delta_sq_y(u_n)
        rhs = u_star - (r_y / 2) * delta_sq_y_un

        # Build tridiagonal system and solve along y-lines.
        a, b, c = self._build_tridiag_coefficients_y(r_y)
        u_dstar = np.zeros_like(u_star)

        for i in range(nx):
            for k in range(nz):
                d = rhs[i, :, k].copy()

                # Boundary conditions in y.
                x_coord, z_coord = self.domain.x[i], self.domain.z[k]

                if self.bc.y_min.bc_type == BCType.DIRICHLET:
                    d[0] = self.bc.y_min.evaluate(
                        (np.array([x_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                elif self.bc.y_min.bc_type in (BCType.NEUMANN, BCType.ROBIN):
                    d[0] = get_bc_rhs_contribution(
                        self.bc.y_min,
                        (np.array([x_coord]), np.array([z_coord])),
                        t_new, dy, True
                    ).flat[0]

                if self.bc.y_max.bc_type == BCType.DIRICHLET:
                    d[-1] = self.bc.y_max.evaluate(
                        (np.array([x_coord]), np.array([z_coord])), t_new
                    ).flat[0]
                elif self.bc.y_max.bc_type in (BCType.NEUMANN, BCType.ROBIN):
                    d[-1] = get_bc_rhs_contribution(
                        self.bc.y_max,
                        (np.array([x_coord]), np.array([z_coord])),
                        t_new, dy, False
                    ).flat[0]

                u_dstar[i, :, k] = solve_tridiagonal(a, b, c, d)

        return u_dstar

    def _peaceman_rachford_step_z(
        self,
        u_dstar: NDArray,
        u_n: NDArray,
        r_z: float,
        t_new: float,
    ) -> NDArray:
        """
        Step 3: Z-direction implicit solve.

        TODO: VERIFY this step matches your derivation.

        Solves:
            (1 - r_z/2 * delta_z^2) u^{n+1} = u** - r_z/2 * delta_z^2 u^n

        This subtracts out the explicit z-contribution from step 1 and replaces
        it with an implicit treatment.
        """
        nx, ny, nz = self.domain.nx, self.domain.ny, self.domain.nz
        dz = self.domain.dz

        # Compute RHS.
        delta_sq_z_un = self._apply_delta_sq_z(u_n)
        rhs = u_dstar - (r_z / 2) * delta_sq_z_un

        # Build tridiagonal system and solve along z-lines.
        a, b, c = self._build_tridiag_coefficients_z(r_z)
        u_new = np.zeros_like(u_dstar)

        for i in range(nx):
            for j in range(ny):
                d = rhs[i, j, :].copy()

                # Boundary conditions in z.
                x_coord, y_coord = self.domain.x[i], self.domain.y[j]

                if self.bc.z_min.bc_type == BCType.DIRICHLET:
                    d[0] = self.bc.z_min.evaluate(
                        (np.array([x_coord]), np.array([y_coord])), t_new
                    ).flat[0]
                elif self.bc.z_min.bc_type in (BCType.NEUMANN, BCType.ROBIN):
                    d[0] = get_bc_rhs_contribution(
                        self.bc.z_min,
                        (np.array([x_coord]), np.array([y_coord])),
                        t_new, dz, True
                    ).flat[0]

                if self.bc.z_max.bc_type == BCType.DIRICHLET:
                    d[-1] = self.bc.z_max.evaluate(
                        (np.array([x_coord]), np.array([y_coord])), t_new
                    ).flat[0]
                elif self.bc.z_max.bc_type in (BCType.NEUMANN, BCType.ROBIN):
                    d[-1] = get_bc_rhs_contribution(
                        self.bc.z_max,
                        (np.array([x_coord]), np.array([y_coord])),
                        t_new, dz, False
                    ).flat[0]

                u_new[i, j, :] = solve_tridiagonal(a, b, c, d)

        return u_new

    def step(self, dt: float) -> None:
        """
        Advance the solution by one time step using adapted Peaceman-Rachford ADI.

        TODO: VERIFY - If your 3D derivation differs, modify the three
        _peaceman_rachford_step_* methods above.

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

        # Evaluate forcing at current and next time.
        F_n = self.forcing(self.X, self.Y, self.Z, t_n)
        F_np1 = self.forcing(self.X, self.Y, self.Z, t_np1)
        forcing_term = (dt / 2) * (F_n + F_np1)

        # Store u^n for use in all steps.
        u_n = self.u.copy()

        # Step 1: X-implicit.
        u_star = self._peaceman_rachford_step_x(u_n, r_x, r_y, r_z, forcing_term, t_np1)

        # Step 2: Y-implicit.
        u_dstar = self._peaceman_rachford_step_y(u_star, u_n, r_y, t_np1)

        # Step 3: Z-implicit.
        u_new = self._peaceman_rachford_step_z(u_dstar, u_n, r_z, t_np1)

        self.u = u_new
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
            'note': "Adapted Peaceman-Rachford scheme should be unconditionally stable. "
                    "TODO: VERIFY this claim for the specific 3D adaptation used.",
        }
