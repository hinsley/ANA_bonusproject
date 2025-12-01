"""
Test Case 1 (2D): The "Decaying Bubble"

Isotropic diffusion with homogeneous Dirichlet boundary conditions.
This tests the 2D D'Yakonov solver against an exact analytical solution.

Domain: Unit Square [0, 1]^2
Exact Solution: u(x,y,t) = exp(-2*pi^2*t) * sin(pi*x) * sin(pi*y)
Boundary Conditions: u = 0 on all boundaries (Dirichlet)
Forcing: F = 0

Features:
- Error comparison against exact solution
- Spatial convergence analysis (grid refinement study)
- 2D animated visualization
- Results saved to report file

Usage:
    python analysis/test_case_1_decaying_bubble_2d.py

Output:
    analysis/test_case_1_decaying_bubble_2d_report.txt
    analysis/test_case_1_decaying_bubble_2d.gif
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.typing import NDArray

from heat_solver import (
    HeatSolver2D,
    DirichletBC,
    BoundaryConditions2D,
    animate_solution_2d,
)
from heat_solver.solver_2d import Domain2D


# =============================================================================
# OUTPUT HELPER
# =============================================================================

class Reporter:
    """Writes output to both console and file."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = open(filepath, 'w', encoding='utf-8')
        self.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def write(self, text: str = ""):
        """Write text to console and file."""
        print(text)
        self.file.write(text + "\n")
        self.file.flush()
    
    def close(self):
        self.file.close()


# =============================================================================
# EXACT SOLUTION
# =============================================================================

def exact_solution(X: NDArray, Y: NDArray, t: float) -> NDArray:
    """
    Exact solution: u(x,y,t) = exp(-2*pi^2*t) * sin(pi*x) * sin(pi*y).
    """
    return (
        np.exp(-2 * np.pi**2 * t)
        * np.sin(np.pi * X)
        * np.sin(np.pi * Y)
    )


def initial_condition(X: NDArray, Y: NDArray) -> NDArray:
    """Initial condition: u_0 = sin(pi*x) * sin(pi*y)."""
    return exact_solution(X, Y, 0.0)


def forcing_function(X: NDArray, Y: NDArray, t: float) -> NDArray:
    """Forcing function: F = 0."""
    return np.zeros_like(X)


def bc_zero(coords: NDArray, t: float) -> NDArray:
    """Zero Dirichlet boundary condition."""
    return np.zeros_like(coords)


# =============================================================================
# SOLVER RUNNER
# =============================================================================

def run_solver(nx: int, dt: float, t_final: float = 0.5) -> tuple:
    """
    Run the solver with given grid resolution and time step.

    Returns (times, solutions, X, Y, error_linf, error_l2, dx).
    """
    domain = Domain2D(0.0, 1.0, 0.0, 1.0, nx, nx)

    bc = BoundaryConditions2D(
        x_min=DirichletBC(bc_zero), x_max=DirichletBC(bc_zero),
        y_min=DirichletBC(bc_zero), y_max=DirichletBC(bc_zero),
    )

    solver = HeatSolver2D(
        domain=domain,
        c=1.0,
        bc=bc,
        initial_condition=initial_condition,
        forcing=forcing_function,
    )

    times, solutions = solver.solve(t_final=t_final, dt=dt, save_every=1)
    X, Y, _, _ = solver.get_solution()

    # Compute errors at final time.
    u_exact = exact_solution(X, Y, t_final)
    u_num = solutions[-1]
    error = np.abs(u_num - u_exact)

    error_linf = np.max(error)
    error_l2 = np.sqrt(np.mean(error**2))

    return times, solutions, X, Y, error_linf, error_l2, domain.dx


# =============================================================================
# CONVERGENCE ANALYSIS
# =============================================================================

def run_convergence_study(report: Reporter) -> list:
    """
    Perform spatial convergence study with grid refinement.

    Uses successively finer grids and measures error against exact solution.
    """
    report.write("\n" + "=" * 70)
    report.write("SPATIAL CONVERGENCE STUDY")
    report.write("=" * 70)

    # Grid resolutions to test.
    grid_sizes = [11, 21, 41, 81, 161]
    t_final = 0.05  # Short time to reduce temporal error contribution.

    results = []

    for nx in grid_sizes:
        # Use moderate dt (scheme is unconditionally stable).
        dt = 0.001
        _, _, _, _, err_linf, err_l2, dx = run_solver(nx, dt, t_final)
        results.append((nx, dx, err_linf, err_l2))
        report.write(f"  Grid {nx:3d}^2: dx = {dx:.6f}, L-inf error = {err_linf:.6e}, L2 error = {err_l2:.6e}")

    # Compute convergence rates.
    report.write("\n" + "-" * 70)
    report.write("Convergence Rates (comparing successive refinements)")
    report.write("-" * 70)
    report.write(f"{'Grid 1':>10} {'Grid 2':>10} {'dx ratio':>10} {'L-inf rate':>12} {'L2 rate':>10}")
    report.write("-" * 70)

    for i in range(1, len(results)):
        nx1, dx1, linf1, l2_1 = results[i - 1]
        nx2, dx2, linf2, l2_2 = results[i]

        dx_ratio = dx1 / dx2
        rate_linf = np.log(linf1 / linf2) / np.log(dx_ratio) if linf2 > 0 else 0
        rate_l2 = np.log(l2_1 / l2_2) / np.log(dx_ratio) if l2_2 > 0 else 0

        report.write(f"{nx1:>10}^2 {nx2:>10}^2 {dx_ratio:>10.2f} {rate_linf:>12.2f} {rate_l2:>10.2f}")

    report.write("-" * 70)
    report.write("Expected rate for 2nd-order scheme: ~2.0")

    return results


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    """Run Test Case 1 (2D): Decaying Bubble."""
    report = Reporter("analysis/test_case_1_decaying_bubble_2d_report.txt")
    
    report.write("=" * 70)
    report.write("Test Case 1 (2D): The 'Decaying Bubble'")
    report.write("=" * 70)
    report.write("\nIsotropic diffusion with homogeneous Dirichlet BCs.")
    report.write("Exact solution: u = exp(-2*pi^2*t) * sin(pi*x)*sin(pi*y)")
    report.write("Time range: t in [0, 1]")

    # Run convergence study.
    run_convergence_study(report)

    # Run detailed simulation for visualization.
    report.write("\n" + "=" * 70)
    report.write("DETAILED SIMULATION FOR VISUALIZATION")
    report.write("=" * 70)

    nx = 51
    dt = 0.001
    t_final = 1.0
    save_every = 10  # More frames for smoother animation.

    domain = Domain2D(0.0, 1.0, 0.0, 1.0, nx, nx)
    report.write(f"\nGrid: {nx}^2 = {nx**2} points, dx = {domain.dx:.6f}")
    report.write(f"Time stepping: dt = {dt}, steps = {int(t_final/dt)}")

    bc = BoundaryConditions2D(
        x_min=DirichletBC(bc_zero), x_max=DirichletBC(bc_zero),
        y_min=DirichletBC(bc_zero), y_max=DirichletBC(bc_zero),
    )

    solver = HeatSolver2D(
        domain=domain,
        c=1.0,
        bc=bc,
        initial_condition=initial_condition,
        forcing=forcing_function,
    )

    times, solutions = solver.solve(t_final=t_final, dt=dt, save_every=save_every)
    X, Y, _, _ = solver.get_solution()

    # Error analysis.
    report.write("\n" + "-" * 70)
    report.write("Error Analysis vs Exact Solution")
    report.write("-" * 70)
    report.write(f"{'Time':>10} {'Max |u|':>12} {'L-inf Err':>12} {'L2 Error':>12} {'Rel Error':>12}")
    report.write("-" * 70)

    for t, u_num in zip(times, solutions):
        u_exact = exact_solution(X, Y, t)
        error = np.abs(u_num - u_exact)
        max_u = np.max(np.abs(u_exact))
        err_linf = np.max(error)
        err_l2 = np.sqrt(np.mean(error**2))
        rel_err = err_linf / max_u if max_u > 1e-15 else 0.0
        report.write(f"{t:10.4f} {max_u:12.6e} {err_linf:12.6e} {err_l2:12.6e} {rel_err:12.6e}")

    # Create visualization.
    report.write("\n" + "-" * 70)
    report.write("Creating 2D animation...")
    anim = animate_solution_2d(
        X, Y, times, solutions,
        cmap="inferno",
        title_prefix="Decaying Bubble (2D)",
        save_path="analysis/test_case_1_decaying_bubble_2d.gif",
        interval=100,
    )
    report.write("Animation saved to: analysis/test_case_1_decaying_bubble_2d.gif")

    report.write("\n" + "=" * 70)
    report.write("Test Case 1 (2D) Complete")
    report.write("=" * 70)
    report.write(f"\nReport saved to: {report.filepath}")
    report.close()


if __name__ == "__main__":
    main()
