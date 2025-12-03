"""
Test Case 2 (3D): The "Standing Wave"

Isotropic diffusion with inhomogeneous forcing and homogeneous Neumann BCs.
This tests the 3D Douglas-Gunn solver against an exact analytical solution.

Domain: Unit Cube [0, 1]^3
Exact Solution: u(x,y,z,t) = exp(-t) * cos(pi*x) * cos(pi*y) * cos(pi*z)
Boundary Conditions: du/dn = 0 on all boundaries (Neumann)
Forcing: F = (3*pi^2 - 1) * exp(-t) * cos(pi*x) * cos(pi*y) * cos(pi*z)

Features:
- Error comparison against exact solution
- Spatial convergence analysis (grid refinement study)
- 3D volumetric visualization
- Results saved to report file

Usage:
    python analysis/test_case_2_standing_wave_3d.py

Output:
    analysis/test_case_2_standing_wave_3d_report.txt
    analysis/test_case_2_standing_wave_3d.gif
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from heat_solver import (
    HeatSolver3D,
    NeumannBC,
    BoundaryConditions3D,
    create_volume_animation,
)
from heat_solver.solver_3d import Domain3D


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

def exact_solution(X: NDArray, Y: NDArray, Z: NDArray, t: float) -> NDArray:
    """
    Exact solution: u(x,y,z,t) = exp(-t) * cos(pi*x) * cos(pi*y) * cos(pi*z).
    """
    return (
        np.exp(-t)
        * np.cos(np.pi * X)
        * np.cos(np.pi * Y)
        * np.cos(np.pi * Z)
    )


def initial_condition(X: NDArray, Y: NDArray, Z: NDArray) -> NDArray:
    """Initial condition: u_0 = cos(pi*x) * cos(pi*y) * cos(pi*z)."""
    return exact_solution(X, Y, Z, 0.0)


def forcing_function(X: NDArray, Y: NDArray, Z: NDArray, t: float) -> NDArray:
    """
    Forcing function: F = (3*pi^2 - 1) * exp(-t) * cos(pi*x) * cos(pi*y) * cos(pi*z).
    """
    coeff = 3 * np.pi**2 - 1
    return (
        coeff * np.exp(-t)
        * np.cos(np.pi * X)
        * np.cos(np.pi * Y)
        * np.cos(np.pi * Z)
    )


def bc_zero_flux(coords: tuple, t: float) -> NDArray:
    """Zero Neumann (no-flux) boundary condition."""
    return np.zeros_like(coords[0])


# =============================================================================
# SOLVER RUNNER
# =============================================================================

def run_solver(nx: int, dt: float, t_final: float = 0.5) -> tuple:
    """
    Run the solver with given grid resolution and time step.

    Returns (times, solutions, X, Y, Z, error_linf, error_l2, dx).
    """
    domain = Domain3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, nx, nx)

    bc = BoundaryConditions3D(
        x_min=NeumannBC(bc_zero_flux), x_max=NeumannBC(bc_zero_flux),
        y_min=NeumannBC(bc_zero_flux), y_max=NeumannBC(bc_zero_flux),
        z_min=NeumannBC(bc_zero_flux), z_max=NeumannBC(bc_zero_flux),
    )

    solver = HeatSolver3D(
        domain=domain,
        c=1.0,
        bc=bc,
        initial_condition=initial_condition,
        forcing=forcing_function,
    )

    times, solutions = solver.solve(t_final=t_final, dt=dt, save_every=1)
    X, Y, Z, _, _ = solver.get_solution()

    # Compute errors at final time.
    u_exact = exact_solution(X, Y, Z, t_final)
    u_num = solutions[-1]
    error = np.abs(u_num - u_exact)

    error_linf = np.max(error)
    error_l2 = np.sqrt(np.mean(error**2))

    return times, solutions, X, Y, Z, error_linf, error_l2, domain.dx


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
    grid_sizes = [11, 16, 21, 31, 41]
    t_final = 0.1  # Moderate time to balance errors.

    results = []

    for nx in grid_sizes:
        # Use moderate dt (scheme is unconditionally stable).
        dt = 0.005
        _, _, _, _, _, err_linf, err_l2, dx = run_solver(nx, dt, t_final)
        results.append((nx, dx, err_linf, err_l2))
        report.write(f"  Grid {nx:3d}^3: dx = {dx:.6f}, L-inf error = {err_linf:.6e}, L2 error = {err_l2:.6e}")

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

        report.write(f"{nx1:>10}^3 {nx2:>10}^3 {dx_ratio:>10.2f} {rate_linf:>12.2f} {rate_l2:>10.2f}")

    report.write("-" * 70)
    report.write("Expected rate for 2nd-order scheme: ~2.0")

    return results


# =============================================================================
# ERROR PLOTTING
# =============================================================================

def create_error_plots(
    times: list,
    errors_linf: list,
    errors_l2: list,
    rel_errors: list,
    max_u_values: list,
    save_path: str,
    title_prefix: str = "",
) -> None:
    """
    Create a 3-panel figure showing error metrics over time.

    Panel 1: L-infinity and L2 errors (dual y-axes).
    Panel 2: Relative error.
    Panel 3: Max |u| values.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"{title_prefix} Error Analysis", fontsize=14, fontweight="bold")

    # Panel 1: L-infinity and L2 errors with dual y-axes.
    ax1 = axes[0]
    color_linf = "tab:blue"
    color_l2 = "tab:red"

    ax1.set_ylabel(r"$L_\infty$ Error", color=color_linf)
    line1, = ax1.plot(times, errors_linf, color=color_linf, label=r"$L_\infty$ Error", linewidth=1.5)
    ax1.tick_params(axis="y", labelcolor=color_linf)
    ax1.set_yscale("log")

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel(r"$L_2$ Error", color=color_l2)
    line2, = ax1_twin.plot(times, errors_l2, color=color_l2, label=r"$L_2$ Error", linewidth=1.5, linestyle="--")
    ax1_twin.tick_params(axis="y", labelcolor=color_l2)
    ax1_twin.set_yscale("log")

    ax1.set_title("Error Norms")
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Relative error.
    ax2 = axes[1]
    ax2.plot(times, rel_errors, color="tab:green", linewidth=1.5)
    ax2.set_ylabel("Relative Error")
    ax2.set_title("Relative Error (L-inf / Max |u|)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Max |u| values.
    ax3 = axes[2]
    ax3.plot(times, max_u_values, color="tab:purple", linewidth=1.5)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Max |u|")
    ax3.set_title("Maximum Absolute Solution Value")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    """Run Test Case 2 (3D): Standing Wave."""
    report = Reporter("analysis/test_case_2_standing_wave_3d_report.txt")
    
    report.write("=" * 70)
    report.write("Test Case 2 (3D): The 'Standing Wave'")
    report.write("=" * 70)
    report.write("\nIsotropic diffusion with forcing and homogeneous Neumann BCs.")
    report.write("Exact solution: u = exp(-t) * cos(pi*x)*cos(pi*y)*cos(pi*z)")
    report.write("Forcing: F = (3*pi^2 - 1) * exp(-t) * cos(pi*x)*cos(pi*y)*cos(pi*z)")
    report.write("Time range: t in [0, 2]")

    # Run convergence study.
    run_convergence_study(report)

    # Run detailed simulation for visualization.
    report.write("\n" + "=" * 70)
    report.write("DETAILED SIMULATION FOR VISUALIZATION")
    report.write("=" * 70)

    nx = 31
    dt = 0.005
    t_final = 2.0
    report_every = 4  # Interval for text output (matching original save_every).

    domain = Domain3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, nx, nx)
    report.write(f"\nGrid: {nx}^3 = {nx**3} points, dx = {domain.dx:.6f}")
    report.write(f"Time stepping: dt = {dt}, steps = {int(t_final/dt)}")

    bc = BoundaryConditions3D(
        x_min=NeumannBC(bc_zero_flux), x_max=NeumannBC(bc_zero_flux),
        y_min=NeumannBC(bc_zero_flux), y_max=NeumannBC(bc_zero_flux),
        z_min=NeumannBC(bc_zero_flux), z_max=NeumannBC(bc_zero_flux),
    )

    solver = HeatSolver3D(
        domain=domain,
        c=1.0,
        bc=bc,
        initial_condition=initial_condition,
        forcing=forcing_function,
    )

    # Run solver saving every step for complete error tracking.
    times_all, solutions_all = solver.solve(t_final=t_final, dt=dt, save_every=1)
    X, Y, Z, _, _ = solver.get_solution()

    # Compute errors at ALL timesteps for plotting.
    all_times = []
    all_errors_linf = []
    all_errors_l2 = []
    all_rel_errors = []
    all_max_u = []

    for t, u_num in zip(times_all, solutions_all):
        u_exact = exact_solution(X, Y, Z, t)
        error = np.abs(u_num - u_exact)
        max_u = np.max(np.abs(u_exact))
        err_linf = np.max(error)
        err_l2 = np.sqrt(np.mean(error**2))
        rel_err = err_linf / max_u if max_u > 1e-15 else 0.0

        all_times.append(t)
        all_errors_linf.append(err_linf)
        all_errors_l2.append(err_l2)
        all_rel_errors.append(rel_err)
        all_max_u.append(max_u)

    # Error analysis (text output at original intervals).
    report.write("\n" + "-" * 70)
    report.write("Error Analysis vs Exact Solution")
    report.write("-" * 70)
    report.write(f"{'Time':>10} {'Max |u|':>12} {'L-inf Err':>12} {'L2 Error':>12} {'Rel Error':>12}")
    report.write("-" * 70)

    for i, (t, u_num) in enumerate(zip(times_all, solutions_all)):
        if i % report_every == 0 or i == len(times_all) - 1:
            u_exact = exact_solution(X, Y, Z, t)
            error = np.abs(u_num - u_exact)
            max_u = np.max(np.abs(u_exact))
            err_linf = np.max(error)
            err_l2 = np.sqrt(np.mean(error**2))
            rel_err = err_linf / max_u if max_u > 1e-15 else 0.0
            report.write(f"{t:10.4f} {max_u:12.6e} {err_linf:12.6e} {err_l2:12.6e} {rel_err:12.6e}")

    # Create error plots.
    report.write("\n" + "-" * 70)
    report.write("Creating error plots...")
    create_error_plots(
        all_times, all_errors_linf, all_errors_l2, all_rel_errors, all_max_u,
        save_path="analysis/test_case_2_standing_wave_3d_errors.png",
        title_prefix="Standing Wave (3D)",
    )
    report.write("Error plots saved to: analysis/test_case_2_standing_wave_3d_errors.png")

    # Create visualization (subsample for animation).
    times_anim = times_all[::report_every]
    solutions_anim = solutions_all[::report_every]

    report.write("\n" + "-" * 70)
    report.write("Creating 3D volumetric animation...")
    try:
        output_path = create_volume_animation(
            X, Y, Z, times_anim, solutions_anim,
            output_path="analysis/test_case_2_standing_wave_3d.gif",
            cmap="coolwarm",
            opacity=0.95,
            title_prefix="Standing Wave (3D)",
            fps=15,
            window_size=(1024, 768),
            camera_position=[(3.0, 1.8, 2.0), (0.5, 0.5, 0.5), (0, 0, 1)],
        )
        report.write(f"Animation saved to: {output_path}")
    except ImportError as e:
        report.write(f"Could not create animation: {e}")

    report.write("\n" + "=" * 70)
    report.write("Test Case 2 (3D) Complete")
    report.write("=" * 70)
    report.write(f"\nReport saved to: {report.filepath}")
    report.close()


if __name__ == "__main__":
    main()
