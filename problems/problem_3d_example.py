"""
Example 3D Heat Equation Problem Definition.

This script demonstrates how to set up and solve a 3D heat equation
problem. Modify the functions and parameters below to define your own
problems.

Usage:
    python problems/problem_3d_example.py

Note: 3D simulations can be memory and compute intensive. Start with
small grid sizes to verify your setup.
"""

import sys
from pathlib import Path

# Add src to path for imports.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.typing import NDArray

from heat_solver import (
    HeatSolver3D,
    DirichletBC,
    NeumannBC,
    RobinBC,
    BoundaryConditions3D,
    plot_solution_3d,
    animate_solution_3d,
    save_solution_series,
)
from heat_solver.solver_3d import Domain3D


# =============================================================================
# PROBLEM DEFINITION - Modify these to define your own problem.
# =============================================================================

# Domain bounds.
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.0

# Grid resolution (keep small for initial tests).
NX = 21  # Number of grid points in x.
NY = 21  # Number of grid points in y.
NZ = 21  # Number of grid points in z.

# Physical parameters.
DIFFUSIVITY = 0.1  # Thermal diffusivity c.

# Time parameters.
T_FINAL = 0.5      # Final time.
DT = 0.001         # Time step.
SAVE_EVERY = 25    # Save solution every N steps.


def initial_condition(X: NDArray, Y: NDArray, Z: NDArray) -> NDArray:
    """
    Define the initial condition u(x, y, z, 0).

    Parameters
    ----------
    X, Y, Z : ndarray
        Meshgrid coordinate arrays.

    Returns
    -------
    u0 : ndarray
        Initial temperature distribution.
    """
    # Example: Gaussian hot spot in the center.
    x0, y0, z0 = 0.5, 0.5, 0.5  # Center.
    sigma = 0.15                 # Width.
    r_sq = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
    return np.exp(-r_sq / (2 * sigma**2))


def forcing_function(X: NDArray, Y: NDArray, Z: NDArray, t: float) -> NDArray:
    """
    Define the forcing (source) term F(x, y, z, t).

    Parameters
    ----------
    X, Y, Z : ndarray
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


# =============================================================================
# BOUNDARY CONDITIONS - Modify these to set your boundary conditions.
# =============================================================================

# For 3D, boundary functions receive (coord1, coord2) as a tuple.
# x_min/x_max faces: coords = (Y, Z)
# y_min/y_max faces: coords = (X, Z)
# z_min/z_max faces: coords = (X, Y)

def bc_x_min(coords: tuple, t: float) -> NDArray:
    """Boundary condition at x = X_MIN face."""
    Y, Z = coords
    return np.zeros_like(Y)


def bc_x_max(coords: tuple, t: float) -> NDArray:
    """Boundary condition at x = X_MAX face."""
    Y, Z = coords
    return np.zeros_like(Y)


def bc_y_min(coords: tuple, t: float) -> NDArray:
    """Boundary condition at y = Y_MIN face."""
    X, Z = coords
    return np.zeros_like(X)


def bc_y_max(coords: tuple, t: float) -> NDArray:
    """Boundary condition at y = Y_MAX face."""
    X, Z = coords
    return np.zeros_like(X)


def bc_z_min(coords: tuple, t: float) -> NDArray:
    """Boundary condition at z = Z_MIN face."""
    X, Y = coords
    return np.zeros_like(X)


def bc_z_max(coords: tuple, t: float) -> NDArray:
    """Boundary condition at z = Z_MAX face."""
    X, Y = coords
    return np.zeros_like(X)


# Choose boundary condition types.
BC_X_MIN = DirichletBC(bc_x_min)
BC_X_MAX = DirichletBC(bc_x_max)
BC_Y_MIN = DirichletBC(bc_y_min)
BC_Y_MAX = DirichletBC(bc_y_max)
BC_Z_MIN = DirichletBC(bc_z_min)
BC_Z_MAX = DirichletBC(bc_z_max)


# =============================================================================
# MAIN SOLVER SCRIPT - Usually no need to modify below this line.
# =============================================================================

def main():
    """Run the 3D heat equation solver."""
    print("=" * 60)
    print("3D Heat Equation Solver")
    print("=" * 60)
    
    # Create domain.
    domain = Domain3D(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, NX, NY, NZ)
    print(f"\nDomain: [{X_MIN}, {X_MAX}] x [{Y_MIN}, {Y_MAX}] x [{Z_MIN}, {Z_MAX}]")
    print(f"Grid: {NX} x {NY} x {NZ} points = {NX * NY * NZ} total")
    print(f"Spacing: dx = {domain.dx:.6f}, dy = {domain.dy:.6f}, dz = {domain.dz:.6f}")
    
    # Estimate memory usage.
    mem_estimate = NX * NY * NZ * 8 / (1024**2)  # MB for one array.
    print(f"Estimated memory per solution array: {mem_estimate:.1f} MB")
    
    # Create boundary conditions.
    bc = BoundaryConditions3D(
        x_min=BC_X_MIN,
        x_max=BC_X_MAX,
        y_min=BC_Y_MIN,
        y_max=BC_Y_MAX,
        z_min=BC_Z_MIN,
        z_max=BC_Z_MAX,
    )
    
    # Create solver.
    solver = HeatSolver3D(
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
    print(f"  r_z = {stability['r_z']:.4f}")
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
    
    # Plot slices of final solution (orthogonal view for better context).
    X, Y, Z, u_final, t_final = solver.get_solution()
    
    fig, axes = plot_solution_3d(
        X,
        Y,
        Z,
        u_final,
        t_final,
        view_mode="orthogonal",
        orthogonal_indices=(NX // 4, NY // 2, 3 * NZ // 4),
        figsize=(14, 4),
    )
    fig.savefig("problems/solution_3d_slices.png", dpi=150, bbox_inches='tight')
    print("\nSlice plots saved to: problems/solution_3d_slices.png")
    
    # Create animation showing all three orthogonal slices.
    print("\nCreating animation (orthogonal slices)...")
    anim = animate_solution_3d(
        X,
        Y,
        Z,
        times,
        solutions,
        title_prefix="3D Heat Equation",
        view_mode="orthogonal",
        orthogonal_indices=(NX // 4, NY // 2, 3 * NZ // 4),
        figsize=(14, 4),
        save_path="problems/solution_3d_animation.gif",
    )
    print("Animation saved to: problems/solution_3d_animation.gif")
    
    # Save solution data.
    save_solution_series(
        "problems/solution_3d_data.npz",
        X[:, :, 0], Y[:, :, 0], times, 
        [sol[:, :, NZ // 2] for sol in solutions],  # Save middle slice only.
        metadata={
            'c': DIFFUSIVITY,
            'dt': DT,
            'nx': NX,
            'ny': NY,
            'nz': NZ,
            'saved_slice': 'z_middle',
        }
    )
    print("Solution data (middle slice) saved to: problems/solution_3d_data.npz")
    
    # Optional: volumetric snapshots at each saved time (requires PyVista).
    try:
        from datetime import datetime
        from heat_solver.visualization import save_volume_snapshots

        # Create timestamped output folder.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        volume_dir = f"problems/volume_snapshots_{timestamp}"

        print(f"\nSaving 3D volumetric snapshots to: {volume_dir}/")
        saved_files = save_volume_snapshots(
            X,
            Y,
            Z,
            times,
            solutions,
            output_dir=volume_dir,
            cmap="viridis",
            opacity=0.5,
            title_prefix="3D Heat Equation",
        )
        print(f"Saved {len(saved_files)} volumetric images.")
    except ImportError:
        print("\nPyVista not available. Skipping 3D volume rendering.")
        print("Install with: uv pip install pyvista")
    
    # Show plots.
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()

