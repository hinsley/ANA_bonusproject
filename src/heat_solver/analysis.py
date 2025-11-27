"""
Analysis utilities for convergence and error analysis.

Provides functions for:
- Computing error norms (L2, L∞, relative errors).
- Convergence studies with grid refinement.
- Richardson extrapolation for order estimation.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt


@dataclass
class ErrorMetrics:
    """Container for error metrics."""
    l2_error: float          # L2 norm of error.
    linf_error: float        # L-infinity (max) norm of error.
    l2_relative: float       # Relative L2 error.
    linf_relative: float     # Relative L-infinity error.
    rms_error: float         # Root mean square error.


def compute_errors(
    numerical: NDArray[np.float64],
    exact: NDArray[np.float64],
    h: Optional[Union[float, Tuple[float, ...]]] = None,
) -> ErrorMetrics:
    """
    Compute error metrics between numerical and exact solutions.

    Parameters
    ----------
    numerical : ndarray
        Numerical solution.
    exact : ndarray
        Exact (analytical) solution.
    h : float or tuple, optional
        Grid spacing(s) for proper L2 norm scaling. If None, uses
        discrete norms without volume weighting.

    Returns
    -------
    metrics : ErrorMetrics
        Container with all error metrics.
    """
    error = numerical - exact
    
    # Compute norms.
    linf_error = np.max(np.abs(error))
    linf_exact = np.max(np.abs(exact))
    
    if h is None:
        # Discrete norms.
        l2_error = np.sqrt(np.mean(error ** 2))
        l2_exact = np.sqrt(np.mean(exact ** 2))
    else:
        # Volume-weighted norms.
        if isinstance(h, (float, int)):
            dV = h ** numerical.ndim
        else:
            dV = np.prod(h)
        
        l2_error = np.sqrt(np.sum(error ** 2) * dV)
        l2_exact = np.sqrt(np.sum(exact ** 2) * dV)
    
    # Compute relative errors (avoid division by zero).
    l2_relative = l2_error / l2_exact if l2_exact > 1e-15 else np.inf
    linf_relative = linf_error / linf_exact if linf_exact > 1e-15 else np.inf
    
    # RMS error.
    rms_error = np.sqrt(np.mean(error ** 2))
    
    return ErrorMetrics(
        l2_error=l2_error,
        linf_error=linf_error,
        l2_relative=l2_relative,
        linf_relative=linf_relative,
        rms_error=rms_error,
    )


def estimate_convergence_order(
    h_values: List[float],
    errors: List[float],
) -> Tuple[float, float]:
    """
    Estimate convergence order from error vs grid spacing data.

    Assumes errors ~ C * h^p, so log(error) = log(C) + p * log(h).
    Uses least squares fit.

    Parameters
    ----------
    h_values : list of float
        Grid spacings.
    errors : list of float
        Corresponding errors.

    Returns
    -------
    order : float
        Estimated convergence order.
    constant : float
        Estimated error constant C.
    """
    log_h = np.log(np.array(h_values))
    log_e = np.log(np.array(errors))
    
    # Linear fit: log(e) = log(C) + p * log(h).
    coeffs = np.polyfit(log_h, log_e, 1)
    order = coeffs[0]
    constant = np.exp(coeffs[1])
    
    return order, constant


def richardson_extrapolation(
    u_coarse: NDArray[np.float64],
    u_fine: NDArray[np.float64],
    r: float = 2.0,
    p: float = 2.0,
) -> NDArray[np.float64]:
    """
    Apply Richardson extrapolation to estimate a more accurate solution.

    Given solutions on coarse (h) and fine (h/r) grids, estimates the
    exact solution using:
        u_exact ≈ (r^p * u_fine - u_coarse) / (r^p - 1)

    Parameters
    ----------
    u_coarse : ndarray
        Solution on coarser grid (sampled to match fine grid).
    u_fine : ndarray
        Solution on finer grid.
    r : float
        Refinement ratio (h_coarse / h_fine).
    p : float
        Expected order of accuracy.

    Returns
    -------
    u_extrap : ndarray
        Richardson-extrapolated solution.
    """
    factor = r ** p
    return (factor * u_fine - u_coarse) / (factor - 1)


@dataclass
class ConvergenceResult:
    """Results from a convergence study."""
    grid_sizes: List[int]
    h_values: List[float]
    l2_errors: List[float]
    linf_errors: List[float]
    l2_order: float
    linf_order: float
    times: List[float]


def convergence_study(
    solver_factory: Callable[[int], "BaseSolver"],
    exact_solution: Callable[[NDArray, NDArray, float], NDArray],
    grid_sizes: List[int],
    t_final: float,
    dt_ratio: Optional[float] = None,
) -> ConvergenceResult:
    """
    Perform a grid convergence study.

    Parameters
    ----------
    solver_factory : callable
        Function that takes grid size n and returns a configured solver.
        The solver should have solve() method and domain with dx, dy properties.
    exact_solution : callable
        Function exact(X, Y, t) -> u_exact for 2D, or exact(X, Y, Z, t) for 3D.
    grid_sizes : list of int
        Grid sizes to test (e.g., [16, 32, 64, 128]).
    t_final : float
        Final time to solve to.
    dt_ratio : float, optional
        If provided, dt = dt_ratio * h^2 to maintain stability ratio.
        Otherwise uses solver's default or a reasonable dt.

    Returns
    -------
    result : ConvergenceResult
        Container with convergence study results.
    """
    h_values = []
    l2_errors = []
    linf_errors = []
    computation_times = []
    
    import time
    
    for n in grid_sizes:
        print(f"Running with grid size {n}...")
        
        # Create solver.
        solver = solver_factory(n)
        
        # Determine grid spacing.
        h = solver.domain.dx  # Assume uniform spacing.
        h_values.append(h)
        
        # Determine time step.
        if dt_ratio is not None:
            dt = dt_ratio * h * h
        else:
            # Default: use dt that gives reasonable r values.
            dt = 0.1 * h * h / solver.c
        
        # Solve.
        start_time = time.time()
        solver.solve(t_final, dt)
        elapsed = time.time() - start_time
        computation_times.append(elapsed)
        
        # Get numerical solution.
        if hasattr(solver, 'Z'):
            # 3D solver.
            X, Y, Z, u_num, t = solver.get_solution()
            u_exact = exact_solution(X, Y, Z, t)
        else:
            # 2D solver.
            X, Y, u_num, t = solver.get_solution()
            u_exact = exact_solution(X, Y, t)
        
        # Compute errors.
        metrics = compute_errors(u_num, u_exact, h)
        l2_errors.append(metrics.l2_error)
        linf_errors.append(metrics.linf_error)
        
        print(f"  h = {h:.6f}, L2 error = {metrics.l2_error:.6e}, "
              f"Linf error = {metrics.linf_error:.6e}, time = {elapsed:.2f}s")
    
    # Estimate convergence orders.
    l2_order, _ = estimate_convergence_order(h_values, l2_errors)
    linf_order, _ = estimate_convergence_order(h_values, linf_errors)
    
    print(f"\nEstimated convergence orders:")
    print(f"  L2: {l2_order:.2f}")
    print(f"  Linf: {linf_order:.2f}")
    
    return ConvergenceResult(
        grid_sizes=grid_sizes,
        h_values=h_values,
        l2_errors=l2_errors,
        linf_errors=linf_errors,
        l2_order=l2_order,
        linf_order=linf_order,
        times=computation_times,
    )


def plot_convergence(
    result: ConvergenceResult,
    expected_order: float = 2.0,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot convergence study results.

    Parameters
    ----------
    result : ConvergenceResult
        Results from convergence_study().
    expected_order : float
        Expected theoretical order for reference line.
    figsize : tuple
        Figure size.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    h = np.array(result.h_values)
    
    # Plot errors.
    ax.loglog(h, result.l2_errors, 'o-', label=f'L2 error (order ≈ {result.l2_order:.2f})')
    ax.loglog(h, result.linf_errors, 's-', label=f'L∞ error (order ≈ {result.linf_order:.2f})')
    
    # Reference line for expected order.
    ref_errors = result.l2_errors[0] * (h / h[0]) ** expected_order
    ax.loglog(h, ref_errors, '--', color='gray', alpha=0.7,
              label=f'O(h^{expected_order:.0f}) reference')
    
    ax.set_xlabel('Grid spacing h')
    ax.set_ylabel('Error')
    ax.set_title('Convergence Study')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    # Add grid sizes as secondary x-axis labels.
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xscale('log')
    ax2.set_xticks(h)
    ax2.set_xticklabels([f'n={n}' for n in result.grid_sizes])
    ax2.set_xlabel('Grid size')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def print_convergence_table(result: ConvergenceResult) -> None:
    """
    Print a formatted table of convergence study results.

    Parameters
    ----------
    result : ConvergenceResult
        Results from convergence_study().
    """
    print("\n" + "=" * 75)
    print("Convergence Study Results")
    print("=" * 75)
    print(f"{'Grid Size':>10} {'h':>12} {'L2 Error':>14} {'L2 Ratio':>10} "
          f"{'L∞ Error':>14} {'L∞ Ratio':>10}")
    print("-" * 75)
    
    for i, n in enumerate(result.grid_sizes):
        h = result.h_values[i]
        l2_e = result.l2_errors[i]
        linf_e = result.linf_errors[i]
        
        if i > 0:
            l2_ratio = result.l2_errors[i - 1] / l2_e
            linf_ratio = result.linf_errors[i - 1] / linf_e
            print(f"{n:>10} {h:>12.6f} {l2_e:>14.6e} {l2_ratio:>10.2f} "
                  f"{linf_e:>14.6e} {linf_ratio:>10.2f}")
        else:
            print(f"{n:>10} {h:>12.6f} {l2_e:>14.6e} {'--':>10} "
                  f"{linf_e:>14.6e} {'--':>10}")
    
    print("-" * 75)
    print(f"Estimated L2 order:  {result.l2_order:.3f}")
    print(f"Estimated L∞ order: {result.linf_order:.3f}")
    print("=" * 75)


def check_stability(
    c: float,
    dt: float,
    dx: float,
    dy: float,
    dz: Optional[float] = None,
    scheme: str = "dyakonov",
) -> dict:
    """
    Check stability parameters for the numerical scheme.

    Parameters
    ----------
    c : float
        Diffusion coefficient.
    dt : float
        Time step.
    dx, dy : float
        Grid spacings.
    dz : float, optional
        Z grid spacing for 3D.
    scheme : str
        Numerical scheme: "dyakonov" or "peaceman-rachford".

    Returns
    -------
    info : dict
        Dictionary with stability parameters and assessment.
    """
    r_x = c * dt / (dx * dx)
    r_y = c * dt / (dy * dy)
    
    info = {
        'r_x': r_x,
        'r_y': r_y,
        'dt': dt,
        'dx': dx,
        'dy': dy,
        'c': c,
    }
    
    if dz is not None:
        r_z = c * dt / (dz * dz)
        info['r_z'] = r_z
        info['dz'] = dz
    
    if scheme.lower() == "dyakonov":
        info['scheme'] = "D'Yakonov ADI"
        info['stable'] = True  # Unconditionally stable.
        info['note'] = ("D'Yakonov scheme is unconditionally stable. "
                        "However, accuracy may degrade for very large r values. "
                        f"Current r_x={r_x:.3f}, r_y={r_y:.3f}.")
        if r_x > 10 or r_y > 10:
            info['warning'] = "Large r values may affect accuracy."
    
    elif scheme.lower() in ("peaceman-rachford", "peaceman_rachford"):
        info['scheme'] = "Peaceman-Rachford ADI (3D adapted)"
        info['stable'] = True  # Should be unconditionally stable.
        info['note'] = ("Adapted Peaceman-Rachford scheme should be unconditionally stable. "
                        "TODO: VERIFY for 3D adaptation.")
        if dz is not None:
            r_z = info['r_z']
            if r_x > 10 or r_y > 10 or r_z > 10:
                info['warning'] = "Large r values may affect accuracy."
    
    else:
        info['note'] = f"Unknown scheme: {scheme}. Cannot assess stability."
        info['stable'] = None
    
    return info

