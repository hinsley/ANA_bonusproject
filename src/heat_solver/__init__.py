"""
Heat Solver: 2D/3D forced heat equation PDE solver using ADI methods.

This package provides numerical solvers for the forced heat equation:
    U_t = c * nabla^2(U) + F(x, t)

Supported schemes:
    - 2D: D'Yakonov ADI scheme
    - 3D: Peaceman-Rachford ADI scheme (adapted for 3D)

Boundary condition types:
    - Dirichlet: u = g(x, t)
    - Neumann: du/dn = g(x, t)
    - Robin: alpha*u + beta*du/dn = g(x, t)
"""

from .boundary import (
    DirichletBC,
    NeumannBC,
    RobinBC,
    BoundaryConditions,
    BoundaryConditions2D,
    BoundaryConditions3D,
)
from .solver_2d import HeatSolver2D
from .solver_3d import HeatSolver3D
from .tridiagonal import solve_tridiagonal
from .visualization import (
    plot_solution_2d,
    plot_solution_3d,
    animate_solution_2d,
    animate_solution_3d,
    save_solution_series,
)
from .analysis import compute_errors, convergence_study

__all__ = [
    "DirichletBC",
    "NeumannBC",
    "RobinBC",
    "BoundaryConditions",
    "BoundaryConditions2D",
    "BoundaryConditions3D",
    "HeatSolver2D",
    "HeatSolver3D",
    "solve_tridiagonal",
    "plot_solution_2d",
    "plot_solution_3d",
    "animate_solution_2d",
    "animate_solution_3d",
    "save_solution_series",
    "compute_errors",
    "convergence_study",
]

