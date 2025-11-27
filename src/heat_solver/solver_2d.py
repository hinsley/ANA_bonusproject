"""
2D Heat Equation Solver using the D'Yakonov ADI scheme.

Solves: U_t = c * (U_xx + U_yy) + F(x, y, t)

The D'Yakonov scheme (equations 4.4.77-78) is unconditionally stable and
second-order accurate in both space and time.

Scheme:
    Step 1: (1 - r_x/2 * δ²_x) u* = (1 + r_x/2 * δ²_x)(1 + r_y/2 * δ²_y) u^n
                                   + Δt/2 * (F^n + F^{n+1})
    Step 2: (1 - r_y/2 * δ²_y) u^{n+1} = u*

Where:
    r_x = c * Δt / Δx²
    r_y = c * Δt / Δy²
    δ²_x u_{j,k} = u_{j-1,k} - 2*u_{j,k} + u_{j+1,k}
    δ²_y u_{j,k} = u_{j,k-1} - 2*u_{j,k} + u_{j,k+1}
"""

from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .boundary import (
    BoundaryConditions2D,
    BoundaryCondition,
    BCType,
    get_bc_coefficients,
    get_bc_rhs_contribution,
)
from .tridiagonal import solve_tridiagonal


@dataclass
class Domain2D:
    """
    2D rectangular domain specification.

    Parameters
    ----------
    x_min, x_max : float
        Domain bounds in x direction.
    y_min, y_max : float
        Domain bounds in y direction.
    nx : int
        Number of grid points in x direction (including boundaries).
    ny : int
        Number of grid points in y direction (including boundaries).
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int
    
    @property
    def dx(self) -> float:
        """Grid spacing in x."""
        return (self.x_max - self.x_min) / (self.nx - 1)
    
    @property
    def dy(self) -> float:
        """Grid spacing in y."""
        return (self.y_max - self.y_min) / (self.ny - 1)
    
    @property
    def x(self) -> NDArray[np.float64]:
        """1D array of x coordinates."""
        return np.linspace(self.x_min, self.x_max, self.nx)
    
    @property
    def y(self) -> NDArray[np.float64]:
        """1D array of y coordinates."""
        return np.linspace(self.y_min, self.y_max, self.ny)
    
    def meshgrid(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return X, Y meshgrid arrays."""
        return np.meshgrid(self.x, self.y, indexing='ij')


class HeatSolver2D:
    """
    2D Heat equation solver using D'Yakonov ADI scheme.

    Parameters
    ----------
    domain : Domain2D
        The spatial domain specification.
    c : float
        Diffusion coefficient (thermal diffusivity).
    bc : BoundaryConditions2D
        Boundary conditions for all four edges.
    initial_condition : callable
        Function u0(x, y) -> initial values. Should accept meshgrid arrays.
    forcing : callable, optional
        Function F(x, y, t) -> forcing values. Should accept meshgrid arrays.
        Default is zero forcing.
    """
    
    def __init__(
        self,
        domain: Domain2D,
        c: float,
        bc: BoundaryConditions2D,
        initial_condition: Callable[[NDArray, NDArray], NDArray],
        forcing: Optional[Callable[[NDArray, NDArray, float], NDArray]] = None,
    ):
        self.domain = domain
        self.c = c
        self.bc = bc
        self.initial_condition = initial_condition
        self.forcing = forcing if forcing is not None else lambda x, y, t: np.zeros_like(x)
        
        # Compute grid.
        self.X, self.Y = domain.meshgrid()
        
        # Initialize solution array.
        self.u = self.initial_condition(self.X, self.Y).astype(np.float64)
        self.t = 0.0
        
        # Apply initial boundary conditions.
        self._apply_boundary_conditions(self.u, 0.0)
    
    def _apply_boundary_conditions(self, u: NDArray, t: float) -> None:
        """Apply Dirichlet boundary conditions directly to the solution array."""
        # x_min boundary (left edge).
        if self.bc.x_min.bc_type == BCType.DIRICHLET:
            u[0, :] = self.bc.x_min.evaluate(self.domain.y, t)
        
        # x_max boundary (right edge).
        if self.bc.x_max.bc_type == BCType.DIRICHLET:
            u[-1, :] = self.bc.x_max.evaluate(self.domain.y, t)
        
        # y_min boundary (bottom edge).
        if self.bc.y_min.bc_type == BCType.DIRICHLET:
            u[:, 0] = self.bc.y_min.evaluate(self.domain.x, t)
        
        # y_max boundary (top edge).
        if self.bc.y_max.bc_type == BCType.DIRICHLET:
            u[:, -1] = self.bc.y_max.evaluate(self.domain.x, t)
    
    def _build_tridiag_coefficients_x(
        self,
        r_x: float,
        j: int,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Build tridiagonal coefficients for x-direction implicit solve.

        The system is: (1 - r_x/2 * δ²_x) u = RHS.
        This gives: -r_x/2 * u_{i-1} + (1 + r_x) * u_i - r_x/2 * u_{i+1} = RHS.
        """
        nx = self.domain.nx
        
        a = np.full(nx, -r_x / 2, dtype=np.float64)
        b = np.full(nx, 1.0 + r_x, dtype=np.float64)
        c = np.full(nx, -r_x / 2, dtype=np.float64)
        
        # Apply boundary conditions.
        # Left boundary (x_min).
        a_coef, b_coef, c_coef = get_bc_coefficients(
            self.bc.x_min, self.domain.dx, is_min_boundary=True
        )
        a[0] = a_coef
        b[0] = b_coef
        c[0] = c_coef
        
        # Right boundary (x_max).
        a_coef, b_coef, c_coef = get_bc_coefficients(
            self.bc.x_max, self.domain.dx, is_min_boundary=False
        )
        a[-1] = a_coef
        b[-1] = b_coef
        c[-1] = c_coef
        
        return a, b, c
    
    def _build_tridiag_coefficients_y(
        self,
        r_y: float,
        i: int,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Build tridiagonal coefficients for y-direction implicit solve.

        The system is: (1 - r_y/2 * δ²_y) u = RHS.
        This gives: -r_y/2 * u_{j-1} + (1 + r_y) * u_j - r_y/2 * u_{j+1} = RHS.
        """
        ny = self.domain.ny
        
        a = np.full(ny, -r_y / 2, dtype=np.float64)
        b = np.full(ny, 1.0 + r_y, dtype=np.float64)
        c = np.full(ny, -r_y / 2, dtype=np.float64)
        
        # Apply boundary conditions.
        # Bottom boundary (y_min).
        a_coef, b_coef, c_coef = get_bc_coefficients(
            self.bc.y_min, self.domain.dy, is_min_boundary=True
        )
        a[0] = a_coef
        b[0] = b_coef
        c[0] = c_coef
        
        # Top boundary (y_max).
        a_coef, b_coef, c_coef = get_bc_coefficients(
            self.bc.y_max, self.domain.dy, is_min_boundary=False
        )
        a[-1] = a_coef
        b[-1] = b_coef
        c[-1] = c_coef
        
        return a, b, c
    
    def _apply_explicit_operator_y(
        self,
        u: NDArray,
        r_y: float,
    ) -> NDArray:
        """
        Apply (1 + r_y/2 * δ²_y) to u.

        Result: r_y/2 * u_{j,k-1} + (1 - r_y) * u_{j,k} + r_y/2 * u_{j,k+1}.
        """
        result = np.zeros_like(u)
        
        # Interior points.
        result[:, 1:-1] = (
            (r_y / 2) * u[:, :-2]
            + (1.0 - r_y) * u[:, 1:-1]
            + (r_y / 2) * u[:, 2:]
        )
        
        # Boundary values will be handled separately.
        result[:, 0] = u[:, 0]
        result[:, -1] = u[:, -1]
        
        return result
    
    def _apply_explicit_operator_x(
        self,
        u: NDArray,
        r_x: float,
    ) -> NDArray:
        """
        Apply (1 + r_x/2 * δ²_x) to u.

        Result: r_x/2 * u_{j-1,k} + (1 - r_x) * u_{j,k} + r_x/2 * u_{j+1,k}.
        """
        result = np.zeros_like(u)
        
        # Interior points.
        result[1:-1, :] = (
            (r_x / 2) * u[:-2, :]
            + (1.0 - r_x) * u[1:-1, :]
            + (r_x / 2) * u[2:, :]
        )
        
        # Boundary values will be handled separately.
        result[0, :] = u[0, :]
        result[-1, :] = u[-1, :]
        
        return result
    
    def step(self, dt: float) -> None:
        """
        Advance the solution by one time step using D'Yakonov ADI.

        Parameters
        ----------
        dt : float
            Time step size.
        """
        dx, dy = self.domain.dx, self.domain.dy
        r_x = self.c * dt / (dx * dx)
        r_y = self.c * dt / (dy * dy)
        
        t_n = self.t
        t_np1 = self.t + dt
        t_mid = self.t + dt / 2
        
        # Evaluate forcing at current and next time.
        F_n = self.forcing(self.X, self.Y, t_n)
        F_np1 = self.forcing(self.X, self.Y, t_np1)
        
        # ===== Step 1: Implicit in x, explicit in y =====
        # RHS = (1 + r_x/2 * δ²_x)(1 + r_y/2 * δ²_y) u^n + dt/2 * (F^n + F^{n+1}).
        # First apply y operator, then x operator.
        temp = self._apply_explicit_operator_y(self.u, r_y)
        rhs_step1 = self._apply_explicit_operator_x(temp, r_x)
        rhs_step1 += (dt / 2) * (F_n + F_np1)
        
        # Solve (1 - r_x/2 * δ²_x) u* = RHS along each y-line.
        u_star = np.zeros_like(self.u)
        
        for k in range(self.domain.ny):
            a, b, c = self._build_tridiag_coefficients_x(r_x, k)
            d = rhs_step1[:, k].copy()
            
            # Modify RHS for boundary conditions.
            y_coord = self.domain.y[k]
            
            # Left boundary.
            if self.bc.x_min.bc_type == BCType.DIRICHLET:
                d[0] = self.bc.x_min.evaluate(np.array([y_coord]), t_np1)[0]
            elif self.bc.x_min.bc_type == BCType.NEUMANN:
                d[0] = get_bc_rhs_contribution(
                    self.bc.x_min, np.array([y_coord]), t_np1, dx, True
                )[0]
            elif self.bc.x_min.bc_type == BCType.ROBIN:
                d[0] = get_bc_rhs_contribution(
                    self.bc.x_min, np.array([y_coord]), t_np1, dx, True
                )[0]
            
            # Right boundary.
            if self.bc.x_max.bc_type == BCType.DIRICHLET:
                d[-1] = self.bc.x_max.evaluate(np.array([y_coord]), t_np1)[0]
            elif self.bc.x_max.bc_type == BCType.NEUMANN:
                d[-1] = get_bc_rhs_contribution(
                    self.bc.x_max, np.array([y_coord]), t_np1, dx, False
                )[0]
            elif self.bc.x_max.bc_type == BCType.ROBIN:
                d[-1] = get_bc_rhs_contribution(
                    self.bc.x_max, np.array([y_coord]), t_np1, dx, False
                )[0]
            
            u_star[:, k] = solve_tridiagonal(a, b, c, d)
        
        # ===== Step 2: Implicit in y =====
        # Solve (1 - r_y/2 * δ²_y) u^{n+1} = u*.
        u_new = np.zeros_like(self.u)
        
        for i in range(self.domain.nx):
            a, b, c = self._build_tridiag_coefficients_y(r_y, i)
            d = u_star[i, :].copy()
            
            # Modify RHS for boundary conditions.
            x_coord = self.domain.x[i]
            
            # Bottom boundary.
            if self.bc.y_min.bc_type == BCType.DIRICHLET:
                d[0] = self.bc.y_min.evaluate(np.array([x_coord]), t_np1)[0]
            elif self.bc.y_min.bc_type == BCType.NEUMANN:
                d[0] = get_bc_rhs_contribution(
                    self.bc.y_min, np.array([x_coord]), t_np1, dy, True
                )[0]
            elif self.bc.y_min.bc_type == BCType.ROBIN:
                d[0] = get_bc_rhs_contribution(
                    self.bc.y_min, np.array([x_coord]), t_np1, dy, True
                )[0]
            
            # Top boundary.
            if self.bc.y_max.bc_type == BCType.DIRICHLET:
                d[-1] = self.bc.y_max.evaluate(np.array([x_coord]), t_np1)[0]
            elif self.bc.y_max.bc_type == BCType.NEUMANN:
                d[-1] = get_bc_rhs_contribution(
                    self.bc.y_max, np.array([x_coord]), t_np1, dy, False
                )[0]
            elif self.bc.y_max.bc_type == BCType.ROBIN:
                d[-1] = get_bc_rhs_contribution(
                    self.bc.y_max, np.array([x_coord]), t_np1, dy, False
                )[0]
            
            u_new[i, :] = solve_tridiagonal(a, b, c, d)
        
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
        save_every = save_every or n_steps  # Default: only save final.
        
        for step_num in range(1, n_steps + 1):
            # Adjust final step if needed.
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
    
    def get_solution(self) -> Tuple[NDArray, NDArray, NDArray, float]:
        """
        Get the current solution state.

        Returns
        -------
        X : ndarray
            X coordinates meshgrid.
        Y : ndarray
            Y coordinates meshgrid.
        u : ndarray
            Current solution values.
        t : float
            Current time.
        """
        return self.X, self.Y, self.u, self.t
    
    def reset(self) -> None:
        """Reset solution to initial condition."""
        self.u = self.initial_condition(self.X, self.Y).astype(np.float64)
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
            Dictionary with r_x, r_y, and stability notes.
        """
        dx, dy = self.domain.dx, self.domain.dy
        r_x = self.c * dt / (dx * dx)
        r_y = self.c * dt / (dy * dy)
        
        return {
            'r_x': r_x,
            'r_y': r_y,
            'dt': dt,
            'dx': dx,
            'dy': dy,
            'c': self.c,
            'note': "D'Yakonov scheme is unconditionally stable for the heat equation.",
        }

