"""
2D Convergence Study Script.

Runs grid refinement studies for the 2D heat equation solver and
verifies the expected order of accuracy.

Usage:
    python analysis/convergence_2d.py
"""

import sys
from pathlib import Path

# Add src to path for imports.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from heat_solver.solver_2d import HeatSolver2D, Domain2D
from heat_solver.boundary import DirichletBC, BoundaryConditions2D
from heat_solver.analysis import (
    convergence_study,
    plot_convergence,
    print_convergence_table,
    compute_errors,
)
from exact_solutions import ManufacturedSolution2D, SeparableSolution2D


def run_manufactured_solution_study():
    """
    Convergence study using the manufactured solution.

    This tests the solver with non-zero forcing.
    """
    print("\n" + "=" * 60)
    print("2D Convergence Study: Manufactured Solution")
    print("=" * 60)
    
    # Create exact solution object.
    exact_sol = ManufacturedSolution2D(c=1.0, decay=2.0)
    x_min, x_max, y_min, y_max = exact_sol.domain_bounds
    
    def solver_factory(n: int) -> HeatSolver2D:
        """Create a solver with grid size n x n."""
        domain = Domain2D(x_min, x_max, y_min, y_max, n, n)
        
        # Homogeneous Dirichlet BCs.
        bc = BoundaryConditions2D(
            x_min=DirichletBC(lambda y, t: np.zeros_like(y)),
            x_max=DirichletBC(lambda y, t: np.zeros_like(y)),
            y_min=DirichletBC(lambda x, t: np.zeros_like(x)),
            y_max=DirichletBC(lambda x, t: np.zeros_like(x)),
        )
        
        return HeatSolver2D(
            domain=domain,
            c=exact_sol.c,
            bc=bc,
            initial_condition=exact_sol.initial_condition,
            forcing=exact_sol.forcing,
        )
    
    # Run convergence study.
    grid_sizes = [16, 32, 64, 128]
    t_final = 0.1
    
    result = convergence_study(
        solver_factory=solver_factory,
        exact_solution=exact_sol.exact,
        grid_sizes=grid_sizes,
        t_final=t_final,
        dt_ratio=0.25,  # dt = 0.25 * h^2.
    )
    
    # Print results.
    print_convergence_table(result)
    
    # Plot results.
    fig = plot_convergence(result, expected_order=2.0)
    fig.suptitle("2D Manufactured Solution Convergence", fontsize=14)
    plt.savefig("analysis/convergence_2d_manufactured.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to: analysis/convergence_2d_manufactured.png")
    
    return result


def run_separable_solution_study():
    """
    Convergence study using the separable (eigenmode) solution.

    This tests the solver without forcing (homogeneous equation).
    """
    print("\n" + "=" * 60)
    print("2D Convergence Study: Separable Solution (no forcing)")
    print("=" * 60)
    
    # Create exact solution object.
    exact_sol = SeparableSolution2D(c=1.0, Lx=1.0, Ly=1.0, n=1, m=1)
    x_min, x_max, y_min, y_max = exact_sol.domain_bounds
    
    def solver_factory(n: int) -> HeatSolver2D:
        """Create a solver with grid size n x n."""
        domain = Domain2D(x_min, x_max, y_min, y_max, n, n)
        
        # Homogeneous Dirichlet BCs.
        bc = BoundaryConditions2D(
            x_min=DirichletBC(lambda y, t: np.zeros_like(y)),
            x_max=DirichletBC(lambda y, t: np.zeros_like(y)),
            y_min=DirichletBC(lambda x, t: np.zeros_like(x)),
            y_max=DirichletBC(lambda x, t: np.zeros_like(x)),
        )
        
        return HeatSolver2D(
            domain=domain,
            c=exact_sol.c,
            bc=bc,
            initial_condition=exact_sol.initial_condition,
            forcing=exact_sol.forcing,  # Zero forcing.
        )
    
    # Run convergence study.
    grid_sizes = [16, 32, 64, 128]
    t_final = 0.1
    
    result = convergence_study(
        solver_factory=solver_factory,
        exact_solution=exact_sol.exact,
        grid_sizes=grid_sizes,
        t_final=t_final,
        dt_ratio=0.25,
    )
    
    # Print results.
    print_convergence_table(result)
    
    # Plot results.
    fig = plot_convergence(result, expected_order=2.0)
    fig.suptitle("2D Separable Solution Convergence", fontsize=14)
    plt.savefig("analysis/convergence_2d_separable.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to: analysis/convergence_2d_separable.png")
    
    return result


def run_temporal_convergence_study():
    """
    Temporal convergence study (fixed spatial grid, varying dt).
    """
    print("\n" + "=" * 60)
    print("2D Temporal Convergence Study")
    print("=" * 60)
    
    # Use a fine spatial grid.
    n = 64
    exact_sol = ManufacturedSolution2D(c=1.0, decay=2.0)
    x_min, x_max, y_min, y_max = exact_sol.domain_bounds
    
    domain = Domain2D(x_min, x_max, y_min, y_max, n, n)
    
    bc = BoundaryConditions2D(
        x_min=DirichletBC(lambda y, t: np.zeros_like(y)),
        x_max=DirichletBC(lambda y, t: np.zeros_like(y)),
        y_min=DirichletBC(lambda x, t: np.zeros_like(x)),
        y_max=DirichletBC(lambda x, t: np.zeros_like(x)),
    )
    
    t_final = 0.1
    dt_values = [0.01, 0.005, 0.0025, 0.00125]
    l2_errors = []
    
    for dt in dt_values:
        solver = HeatSolver2D(
            domain=domain,
            c=exact_sol.c,
            bc=bc,
            initial_condition=exact_sol.initial_condition,
            forcing=exact_sol.forcing,
        )
        
        solver.solve(t_final, dt)
        X, Y, u_num, t = solver.get_solution()
        u_exact = exact_sol.exact(X, Y, t)
        
        metrics = compute_errors(u_num, u_exact)
        l2_errors.append(metrics.l2_error)
        print(f"  dt = {dt:.6f}, L2 error = {metrics.l2_error:.6e}")
    
    # Estimate temporal order.
    from heat_solver.analysis import estimate_convergence_order
    order, _ = estimate_convergence_order(dt_values, l2_errors)
    print(f"\nEstimated temporal order: {order:.2f}")
    print("(Expected: ~2 for D'Yakonov scheme)")
    
    # Plot.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(dt_values, l2_errors, 'o-', label=f'L2 error (order ≈ {order:.2f})')
    
    # Reference line.
    ref = l2_errors[0] * (np.array(dt_values) / dt_values[0]) ** 2
    ax.loglog(dt_values, ref, '--', color='gray', label='O(dt²) reference')
    
    ax.set_xlabel('Time step dt')
    ax.set_ylabel('L2 Error')
    ax.set_title('2D Temporal Convergence')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    plt.savefig("analysis/convergence_2d_temporal.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to: analysis/convergence_2d_temporal.png")


if __name__ == "__main__":
    # Run all studies.
    run_manufactured_solution_study()
    run_separable_solution_study()
    run_temporal_convergence_study()
    
    plt.show()



