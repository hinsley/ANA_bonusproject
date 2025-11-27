"""
Example 2D Heat Equation Problem Definition.

This script demonstrates how to set up and solve a 2D heat equation
problem. Modify the functions and parameters below to define your own
problems.

Usage:
    python problems/problem_2d_example.py
"""

import sys
from pathlib import Path

# Add src to path for imports.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.typing import NDArray

from heat_solver import (
    HeatSolver2D,
    DirichletBC,
    NeumannBC,
    RobinBC,
    BoundaryConditions2D,
    plot_solution_2d,
    animate_solution_2d,
    save_solution_series,
)
from heat_solver.solver_2d import Domain2D


# =============================================================================
# PROBLEM DEFINITION - Modify these to define your own problem.
# =============================================================================

# Domain bounds.
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0

# Grid resolution.
NX = 51  # Number of grid points in x.
NY = 51  # Number of grid points in y.

# Physical parameters.
DIFFUSIVITY = 0.1  # Thermal diffusivity c.

# Time parameters.
T_FINAL = 1.0      # Final time.
DT = 0.001         # Time step.
SAVE_EVERY = 50    # Save solution every N steps.


def initial_condition(X: NDArray, Y: NDArray) -> NDArray:
    """
    Define the initial condition u(x, y, 0).

    Parameters
    ----------
    X, Y : ndarray
        Meshgrid coordinate arrays.

    Returns
    -------
    u0 : ndarray
        Initial temperature distribution.
    """
    # Example: Gaussian hot spot in the center.
    x0, y0 = 0.5, 0.5  # Center of the Gaussian.
    sigma = 0.1        # Width.
    return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))


def forcing_function(X: NDArray, Y: NDArray, t: float) -> NDArray:
    """
    Define the forcing (source) term F(x, y, t).

    Parameters
    ----------
    X, Y : ndarray
        Meshgrid coordinate arrays.
    t : float
        Current time.

    Returns
    -------
    F : ndarray
        Forcing values.
    """
    # Example: No external forcing.
    return np.zeros_like(X)
    
    # Alternative: Oscillating heat source.
    # x0, y0 = 0.25, 0.25
    # return 10.0 * np.sin(2 * np.pi * t) * np.exp(-((X - x0)**2 + (Y - y0)**2) / 0.01)


# =============================================================================
# BOUNDARY CONDITIONS - Modify these to set your boundary conditions.
# =============================================================================

def bc_x_min(y: NDArray, t: float) -> NDArray:
    """Boundary condition at x = X_MIN (left edge)."""
    # Dirichlet: fixed temperature of 0.
    return np.zeros_like(y)


def bc_x_max(y: NDArray, t: float) -> NDArray:
    """Boundary condition at x = X_MAX (right edge)."""
    # Dirichlet: fixed temperature of 0.
    return np.zeros_like(y)


def bc_y_min(x: NDArray, t: float) -> NDArray:
    """Boundary condition at y = Y_MIN (bottom edge)."""
    # Dirichlet: fixed temperature of 0.
    return np.zeros_like(x)


def bc_y_max(x: NDArray, t: float) -> NDArray:
    """Boundary condition at y = Y_MAX (top edge)."""
    # Dirichlet: fixed temperature of 0.
    return np.zeros_like(x)


# Choose boundary condition types.
# Options: DirichletBC, NeumannBC, RobinBC.
BC_X_MIN = DirichletBC(bc_x_min)
BC_X_MAX = DirichletBC(bc_x_max)
BC_Y_MIN = DirichletBC(bc_y_min)
BC_Y_MAX = DirichletBC(bc_y_max)


# =============================================================================
# MAIN SOLVER SCRIPT - Usually no need to modify below this line.
# =============================================================================

def main():
    """Run the 2D heat equation solver."""
    print("=" * 60)
    print("2D Heat Equation Solver")
    print("=" * 60)
    
    # Create domain.
    domain = Domain2D(X_MIN, X_MAX, Y_MIN, Y_MAX, NX, NY)
    print(f"\nDomain: [{X_MIN}, {X_MAX}] x [{Y_MIN}, {Y_MAX}]")
    print(f"Grid: {NX} x {NY} points")
    print(f"Spacing: dx = {domain.dx:.6f}, dy = {domain.dy:.6f}")
    
    # Create boundary conditions.
    bc = BoundaryConditions2D(
        x_min=BC_X_MIN,
        x_max=BC_X_MAX,
        y_min=BC_Y_MIN,
        y_max=BC_Y_MAX,
    )
    
    # Create solver.
    solver = HeatSolver2D(
        domain=domain,
        c=DIFFUSIVITY,
        bc=bc,
        initial_condition=initial_condition,
        forcing=forcing_function,
    )
    
    # Check stability parameters.
    stability = solver.get_stability_parameters(DT)
    print(f"\nStability parameters:")
    print(f"  r_x = {stability['r_x']:.4f}")
    print(f"  r_y = {stability['r_y']:.4f}")
    print(f"  {stability['note']}")
    
    # Solve.
    print(f"\nSolving from t=0 to t={T_FINAL} with dt={DT}...")
    print(f"Saving every {SAVE_EVERY} steps.")
    
    def progress_callback(t, u):
        print(f"  t = {t:.4f}, max(u) = {np.max(u):.6f}, min(u) = {np.min(u):.6f}")
    
    times, solutions = solver.solve(
        t_final=T_FINAL,
        dt=DT,
        save_every=SAVE_EVERY,
        callback=progress_callback,
    )
    
    print(f"\nSolution computed at {len(times)} time points.")
    
    # Plot final solution.
    X, Y, u_final, t_final = solver.get_solution()
    fig, ax = plot_solution_2d(X, Y, u_final, t_final, plot_type="contourf")
    fig.savefig("problems/solution_2d_final.png", dpi=150, bbox_inches='tight')
    print("\nFinal solution saved to: problems/solution_2d_final.png")
    
    # Create animation.
    print("\nCreating animation...")
    anim = animate_solution_2d(
        X, Y, times, solutions,
        title_prefix="2D Heat Equation",
        save_path="problems/solution_2d_animation.gif",
    )
    print("Animation saved to: problems/solution_2d_animation.gif")
    
    # Save solution data.
    save_solution_series(
        "problems/solution_2d_data.npz",
        X, Y, times, solutions,
        metadata={
            'c': DIFFUSIVITY,
            'dt': DT,
            'nx': NX,
            'ny': NY,
        }
    )
    print("Solution data saved to: problems/solution_2d_data.npz")
    
    # Show plots.
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()

