"""
3D Convergence Study Script.

Runs grid refinement studies for the 3D heat equation solver and
verifies the expected order of accuracy.

Usage:
    python analysis/convergence_3d.py

Note: 3D convergence studies can be computationally expensive.
Consider using smaller grid sizes or running on a machine with
sufficient memory.
"""

import sys
from pathlib import Path

# Add src to path for imports.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from heat_solver.solver_3d import HeatSolver3D, Domain3D
from heat_solver.boundary import DirichletBC, BoundaryConditions3D
from heat_solver.analysis import (
    convergence_study,
    plot_convergence,
    print_convergence_table,
    compute_errors,
    estimate_convergence_order,
)
from exact_solutions import ManufacturedSolution3D, SeparableSolution3D


def run_manufactured_solution_study():
    """
    Convergence study using the 3D manufactured solution.

    This tests the solver with non-zero forcing.
    """
    print("\n" + "=" * 60)
    print("3D Convergence Study: Manufactured Solution")
    print("=" * 60)
    
    # Create exact solution object.
    exact_sol = ManufacturedSolution3D(c=1.0, decay=3.0)
    x_min, x_max, y_min, y_max, z_min, z_max = exact_sol.domain_bounds
    
    def solver_factory(n: int) -> HeatSolver3D:
        """Create a solver with grid size n x n x n."""
        domain = Domain3D(x_min, x_max, y_min, y_max, z_min, z_max, n, n, n)
        
        # Homogeneous Dirichlet BCs on all six faces.
        bc = BoundaryConditions3D(
            x_min=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
            x_max=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
            y_min=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
            y_max=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
            z_min=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
            z_max=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
        )
        
        return HeatSolver3D(
            domain=domain,
            c=exact_sol.c,
            bc=bc,
            initial_condition=exact_sol.initial_condition,
            forcing=exact_sol.forcing,
        )
    
    # Run convergence study.
    # Note: 3D is expensive, use smaller grids.
    grid_sizes = [8, 16, 32]
    t_final = 0.05
    
    # Manual convergence study for 3D (to handle different API).
    h_values = []
    l2_errors = []
    linf_errors = []
    times = []
    
    import time
    
    for n in grid_sizes:
        print(f"Running with grid size {n}x{n}x{n}...")
        
        solver = solver_factory(n)
        h = solver.domain.dx
        h_values.append(h)
        
        dt = 0.25 * h * h / solver.c
        
        start = time.time()
        solver.solve(t_final, dt)
        elapsed = time.time() - start
        times.append(elapsed)
        
        X, Y, Z, u_num, t = solver.get_solution()
        u_exact = exact_sol.exact(X, Y, Z, t)
        
        metrics = compute_errors(u_num, u_exact, h)
        l2_errors.append(metrics.l2_error)
        linf_errors.append(metrics.linf_error)
        
        print(f"  h = {h:.6f}, L2 error = {metrics.l2_error:.6e}, "
              f"Linf error = {metrics.linf_error:.6e}, time = {elapsed:.2f}s")
    
    # Estimate orders.
    l2_order, _ = estimate_convergence_order(h_values, l2_errors)
    linf_order, _ = estimate_convergence_order(h_values, linf_errors)
    
    print(f"\nEstimated convergence orders:")
    print(f"  L2: {l2_order:.2f}")
    print(f"  Linf: {linf_order:.2f}")
    print("  (Expected: ~2 for second-order scheme)")
    
    # Plot results.
    fig, ax = plt.subplots(figsize=(10, 6))
    h = np.array(h_values)
    
    ax.loglog(h, l2_errors, 'o-', label=f'L2 error (order ≈ {l2_order:.2f})')
    ax.loglog(h, linf_errors, 's-', label=f'L∞ error (order ≈ {linf_order:.2f})')
    
    # Reference line.
    ref = l2_errors[0] * (h / h[0]) ** 2
    ax.loglog(h, ref, '--', color='gray', alpha=0.7, label='O(h²) reference')
    
    ax.set_xlabel('Grid spacing h')
    ax.set_ylabel('Error')
    ax.set_title('3D Manufactured Solution Convergence')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    # Secondary axis with grid sizes.
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xscale('log')
    ax2.set_xticks(h)
    ax2.set_xticklabels([f'n={n}' for n in grid_sizes])
    ax2.set_xlabel('Grid size')
    
    plt.tight_layout()
    plt.savefig("analysis/convergence_3d_manufactured.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to: analysis/convergence_3d_manufactured.png")


def run_separable_solution_study():
    """
    Convergence study using the 3D separable (eigenmode) solution.

    This tests the solver without forcing (homogeneous equation).
    """
    print("\n" + "=" * 60)
    print("3D Convergence Study: Separable Solution (no forcing)")
    print("=" * 60)
    
    # Create exact solution object.
    exact_sol = SeparableSolution3D(c=1.0, Lx=1.0, Ly=1.0, Lz=1.0, n=1, m=1, p=1)
    x_min, x_max, y_min, y_max, z_min, z_max = exact_sol.domain_bounds
    
    def solver_factory(n: int) -> HeatSolver3D:
        """Create a solver with grid size n x n x n."""
        domain = Domain3D(x_min, x_max, y_min, y_max, z_min, z_max, n, n, n)
        
        bc = BoundaryConditions3D(
            x_min=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
            x_max=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
            y_min=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
            y_max=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
            z_min=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
            z_max=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
        )
        
        return HeatSolver3D(
            domain=domain,
            c=exact_sol.c,
            bc=bc,
            initial_condition=exact_sol.initial_condition,
            forcing=exact_sol.forcing,  # Zero forcing.
        )
    
    # Run study.
    grid_sizes = [8, 16, 32]
    t_final = 0.05
    
    h_values = []
    l2_errors = []
    linf_errors = []
    
    import time
    
    for n in grid_sizes:
        print(f"Running with grid size {n}x{n}x{n}...")
        
        solver = solver_factory(n)
        h = solver.domain.dx
        h_values.append(h)
        
        dt = 0.25 * h * h / solver.c
        
        start = time.time()
        solver.solve(t_final, dt)
        elapsed = time.time() - start
        
        X, Y, Z, u_num, t = solver.get_solution()
        u_exact = exact_sol.exact(X, Y, Z, t)
        
        metrics = compute_errors(u_num, u_exact, h)
        l2_errors.append(metrics.l2_error)
        linf_errors.append(metrics.linf_error)
        
        print(f"  h = {h:.6f}, L2 error = {metrics.l2_error:.6e}, "
              f"Linf error = {metrics.linf_error:.6e}, time = {elapsed:.2f}s")
    
    # Estimate orders.
    l2_order, _ = estimate_convergence_order(h_values, l2_errors)
    linf_order, _ = estimate_convergence_order(h_values, linf_errors)
    
    print(f"\nEstimated convergence orders:")
    print(f"  L2: {l2_order:.2f}")
    print(f"  Linf: {linf_order:.2f}")
    
    # Plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    h = np.array(h_values)
    
    ax.loglog(h, l2_errors, 'o-', label=f'L2 error (order ≈ {l2_order:.2f})')
    ax.loglog(h, linf_errors, 's-', label=f'L∞ error (order ≈ {linf_order:.2f})')
    
    ref = l2_errors[0] * (h / h[0]) ** 2
    ax.loglog(h, ref, '--', color='gray', alpha=0.7, label='O(h²) reference')
    
    ax.set_xlabel('Grid spacing h')
    ax.set_ylabel('Error')
    ax.set_title('3D Separable Solution Convergence')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("analysis/convergence_3d_separable.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to: analysis/convergence_3d_separable.png")


if __name__ == "__main__":
    print("=" * 60)
    print("3D Heat Equation Convergence Studies")
    print("=" * 60)
    print("\nNote: 3D studies can be computationally expensive.")
    print("Using smaller grid sizes than 2D studies.\n")
    
    run_manufactured_solution_study()
    run_separable_solution_study()
    
    plt.show()

