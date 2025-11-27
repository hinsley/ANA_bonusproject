"""
Library of exact solutions for verifying the heat equation solver.

This module provides analytical solutions to the heat equation for
testing convergence and accuracy of the numerical solver.

Each solution is defined as a class with:
- initial_condition(X, Y) or initial_condition(X, Y, Z): Initial values.
- forcing(X, Y, t) or forcing(X, Y, Z, t): Forcing function F.
- exact(X, Y, t) or exact(X, Y, Z, t): Exact solution at time t.
- boundary_value(coords, t): Boundary values for Dirichlet BCs.
- Metadata: domain bounds, diffusion coefficient, etc.
"""

from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray


# ===========================================================================
# 2D Exact Solutions
# ===========================================================================

@dataclass
class ManufacturedSolution2D:
    """
    Manufactured solution for 2D heat equation testing.

    The method of manufactured solutions: choose u(x,y,t), then compute
    the required forcing F = u_t - c*Laplacian(u) to make it an exact solution.

    Solution: u(x,y,t) = sin(pi*x) * sin(pi*y) * exp(-decay*t)

    This satisfies homogeneous Dirichlet BCs on [0,1]^2.
    """
    c: float = 1.0
    decay: float = 2.0  # Controls time decay rate.
    
    @property
    def domain_bounds(self) -> Tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max)."""
        return (0.0, 1.0, 0.0, 1.0)
    
    def initial_condition(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """u(x, y, 0) = sin(pi*x) * sin(pi*y)."""
        return np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    def exact(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Exact solution u(x, y, t)."""
        return np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-self.decay * t)
    
    def forcing(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """
        Forcing function F(x, y, t) required to make the manufactured solution exact.

        F = u_t - c * (u_xx + u_yy)
          = -decay * u - c * (-2*pi^2) * u
          = (-decay + 2*c*pi^2) * u
        """
        u = self.exact(X, Y, t)
        return (-self.decay + 2 * self.c * np.pi**2) * u
    
    def boundary_value(
        self,
        coords: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Boundary values (zero for this solution)."""
        return np.zeros_like(coords)


@dataclass
class SeparableSolution2D:
    """
    Separable exact solution for the homogeneous heat equation.

    Solution: u(x,y,t) = sin(n*pi*x/Lx) * sin(m*pi*y/Ly) * exp(-lambda*t)

    where lambda = c * pi^2 * (n^2/Lx^2 + m^2/Ly^2).

    This satisfies homogeneous Dirichlet BCs with zero forcing.
    """
    c: float = 1.0
    Lx: float = 1.0
    Ly: float = 1.0
    n: int = 1
    m: int = 1
    
    @property
    def domain_bounds(self) -> Tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max)."""
        return (0.0, self.Lx, 0.0, self.Ly)
    
    @property
    def decay_rate(self) -> float:
        """Eigenvalue lambda determining decay rate."""
        return self.c * np.pi**2 * (self.n**2 / self.Lx**2 + self.m**2 / self.Ly**2)
    
    def initial_condition(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Initial condition u(x, y, 0)."""
        return (np.sin(self.n * np.pi * X / self.Lx) *
                np.sin(self.m * np.pi * Y / self.Ly))
    
    def exact(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Exact solution u(x, y, t)."""
        spatial = (np.sin(self.n * np.pi * X / self.Lx) *
                   np.sin(self.m * np.pi * Y / self.Ly))
        return spatial * np.exp(-self.decay_rate * t)
    
    def forcing(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Zero forcing for this exact solution."""
        return np.zeros_like(X)
    
    def boundary_value(
        self,
        coords: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Boundary values (zero for this solution)."""
        return np.zeros_like(coords)


@dataclass
class GaussianDiffusion2D:
    """
    Gaussian diffusion solution (fundamental solution).

    Solution: u(x,y,t) = (1/(4*pi*c*(t+t0))) * exp(-(|r-r0|^2)/(4*c*(t+t0)))

    where r = (x, y), r0 = (x0, y0).

    Note: This solution is on an infinite domain. For finite domains,
    it's only approximate and requires compatible boundary conditions.
    """
    c: float = 1.0
    x0: float = 0.5  # Center x.
    y0: float = 0.5  # Center y.
    t0: float = 0.01  # Initial time offset (prevents singularity).
    amplitude: float = 1.0
    
    @property
    def domain_bounds(self) -> Tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max)."""
        return (0.0, 1.0, 0.0, 1.0)
    
    def initial_condition(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Initial condition u(x, y, 0)."""
        return self.exact(X, Y, 0.0)
    
    def exact(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Exact solution u(x, y, t)."""
        tau = t + self.t0
        r_sq = (X - self.x0)**2 + (Y - self.y0)**2
        return (self.amplitude / (4 * np.pi * self.c * tau)) * np.exp(-r_sq / (4 * self.c * tau))
    
    def forcing(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Zero forcing for the fundamental solution."""
        return np.zeros_like(X)
    
    def boundary_value_x_min(
        self,
        y: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Boundary value at x = x_min."""
        x_min = self.domain_bounds[0]
        X = np.full_like(y, x_min)
        return self.exact(X, y, t)
    
    def boundary_value_x_max(
        self,
        y: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Boundary value at x = x_max."""
        x_max = self.domain_bounds[1]
        X = np.full_like(y, x_max)
        return self.exact(X, y, t)
    
    def boundary_value_y_min(
        self,
        x: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Boundary value at y = y_min."""
        y_min = self.domain_bounds[2]
        Y = np.full_like(x, y_min)
        return self.exact(x, Y, t)
    
    def boundary_value_y_max(
        self,
        x: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Boundary value at y = y_max."""
        y_max = self.domain_bounds[3]
        Y = np.full_like(x, y_max)
        return self.exact(x, Y, t)


# ===========================================================================
# 3D Exact Solutions
# ===========================================================================

@dataclass
class ManufacturedSolution3D:
    """
    Manufactured solution for 3D heat equation testing.

    Solution: u(x,y,z,t) = sin(pi*x) * sin(pi*y) * sin(pi*z) * exp(-decay*t)

    This satisfies homogeneous Dirichlet BCs on [0,1]^3.
    """
    c: float = 1.0
    decay: float = 3.0  # Controls time decay rate.
    
    @property
    def domain_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max, z_min, z_max)."""
        return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    
    def initial_condition(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        Z: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """u(x, y, z, 0) = sin(pi*x) * sin(pi*y) * sin(pi*z)."""
        return np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    
    def exact(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        Z: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Exact solution u(x, y, z, t)."""
        spatial = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
        return spatial * np.exp(-self.decay * t)
    
    def forcing(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        Z: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """
        Forcing function F(x, y, z, t).

        F = u_t - c * (u_xx + u_yy + u_zz)
          = -decay * u - c * (-3*pi^2) * u
          = (-decay + 3*c*pi^2) * u
        """
        u = self.exact(X, Y, Z, t)
        return (-self.decay + 3 * self.c * np.pi**2) * u
    
    def boundary_value(
        self,
        coords: Tuple[NDArray[np.float64], NDArray[np.float64]],
        t: float,
    ) -> NDArray[np.float64]:
        """Boundary values (zero for this solution)."""
        return np.zeros_like(coords[0])


@dataclass
class SeparableSolution3D:
    """
    Separable exact solution for the 3D homogeneous heat equation.

    Solution: u = sin(n*pi*x/Lx) * sin(m*pi*y/Ly) * sin(p*pi*z/Lz) * exp(-lambda*t)

    where lambda = c * pi^2 * (n^2/Lx^2 + m^2/Ly^2 + p^2/Lz^2).
    """
    c: float = 1.0
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    n: int = 1
    m: int = 1
    p: int = 1
    
    @property
    def domain_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max, z_min, z_max)."""
        return (0.0, self.Lx, 0.0, self.Ly, 0.0, self.Lz)
    
    @property
    def decay_rate(self) -> float:
        """Eigenvalue lambda determining decay rate."""
        return self.c * np.pi**2 * (
            self.n**2 / self.Lx**2 +
            self.m**2 / self.Ly**2 +
            self.p**2 / self.Lz**2
        )
    
    def initial_condition(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        Z: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Initial condition u(x, y, z, 0)."""
        return (np.sin(self.n * np.pi * X / self.Lx) *
                np.sin(self.m * np.pi * Y / self.Ly) *
                np.sin(self.p * np.pi * Z / self.Lz))
    
    def exact(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        Z: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Exact solution u(x, y, z, t)."""
        spatial = (np.sin(self.n * np.pi * X / self.Lx) *
                   np.sin(self.m * np.pi * Y / self.Ly) *
                   np.sin(self.p * np.pi * Z / self.Lz))
        return spatial * np.exp(-self.decay_rate * t)
    
    def forcing(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        Z: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Zero forcing for this exact solution."""
        return np.zeros_like(X)
    
    def boundary_value(
        self,
        coords: Tuple[NDArray[np.float64], NDArray[np.float64]],
        t: float,
    ) -> NDArray[np.float64]:
        """Boundary values (zero for this solution)."""
        return np.zeros_like(coords[0])


# ===========================================================================
# Utility Functions
# ===========================================================================

def list_available_solutions() -> None:
    """Print available exact solutions."""
    print("Available 2D Solutions:")
    print("  - ManufacturedSolution2D: sin(pi*x)*sin(pi*y)*exp(-t) with forcing.")
    print("  - SeparableSolution2D: Eigenmode solutions, no forcing.")
    print("  - GaussianDiffusion2D: Gaussian blob diffusion.")
    print()
    print("Available 3D Solutions:")
    print("  - ManufacturedSolution3D: sin(pi*x)*sin(pi*y)*sin(pi*z)*exp(-t) with forcing.")
    print("  - SeparableSolution3D: Eigenmode solutions, no forcing.")

