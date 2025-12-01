"""
Thomas algorithm (TDMA) for solving tridiagonal linear systems.

The Thomas algorithm solves systems of the form:
    a_i * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i

This is O(n) and is used extensively in ADI methods where we solve
many independent tridiagonal systems per time step.
"""

import numpy as np
from numpy.typing import NDArray


def solve_tridiagonal(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Solve a tridiagonal system using the Thomas algorithm.

    Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]

    Parameters
    ----------
    a : ndarray
        Sub-diagonal coefficients (a[0] is not used). Shape: (n,).
    b : ndarray
        Main diagonal coefficients. Shape: (n,).
    c : ndarray
        Super-diagonal coefficients (c[n-1] is not used). Shape: (n,).
    d : ndarray
        Right-hand side vector. Shape: (n,).

    Returns
    -------
    x : ndarray
        Solution vector. Shape: (n,).

    Notes
    -----
    The algorithm modifies copies of the input arrays. The original
    arrays are not modified.
    """
    n = len(d)
    
    # Make copies to avoid modifying inputs.
    c_prime = np.zeros(n, dtype=np.float64)
    d_prime = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    
    # Forward sweep.
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom
    
    # Back substitution.
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    
    return x


def solve_tridiagonal_batch(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Solve multiple independent tridiagonal systems in batch.

    This is useful for ADI methods where we solve many systems with the
    same structure along grid lines.

    Parameters
    ----------
    a : ndarray
        Sub-diagonal coefficients. Shape: (n_systems, n).
    b : ndarray
        Main diagonal coefficients. Shape: (n_systems, n).
    c : ndarray
        Super-diagonal coefficients. Shape: (n_systems, n).
    d : ndarray
        Right-hand side vectors. Shape: (n_systems, n).

    Returns
    -------
    x : ndarray
        Solution vectors. Shape: (n_systems, n).
    """
    n_systems, n = d.shape
    
    # Allocate working arrays.
    c_prime = np.zeros_like(c)
    d_prime = np.zeros_like(d)
    x = np.zeros_like(d)
    
    # Forward sweep (vectorized over systems).
    c_prime[:, 0] = c[:, 0] / b[:, 0]
    d_prime[:, 0] = d[:, 0] / b[:, 0]
    
    for i in range(1, n):
        denom = b[:, i] - a[:, i] * c_prime[:, i - 1]
        c_prime[:, i] = c[:, i] / denom
        d_prime[:, i] = (d[:, i] - a[:, i] * d_prime[:, i - 1]) / denom
    
    # Back substitution (vectorized over systems).
    x[:, n - 1] = d_prime[:, n - 1]
    for i in range(n - 2, -1, -1):
        x[:, i] = d_prime[:, i] - c_prime[:, i] * x[:, i + 1]
    
    return x


def solve_tridiagonal_periodic(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Solve a periodic tridiagonal system using Sherman-Morrison.

    This handles the case where the matrix has corner elements connecting
    the first and last unknowns, as in periodic boundary conditions.

    Parameters
    ----------
    a : ndarray
        Sub-diagonal coefficients with a[0] being the corner element. Shape: (n,).
    b : ndarray
        Main diagonal coefficients. Shape: (n,).
    c : ndarray
        Super-diagonal coefficients with c[n-1] being the corner element. Shape: (n,).
    d : ndarray
        Right-hand side vector. Shape: (n,).

    Returns
    -------
    x : ndarray
        Solution vector. Shape: (n,).
    """
    n = len(d)
    
    # Use Sherman-Morrison formula.
    # The periodic system can be written as (A + u*v^T)*x = d,
    # where A is tridiagonal and u*v^T captures the corner elements.
    gamma = -b[0]
    
    # Modified diagonal for the base system.
    b_mod = b.copy()
    b_mod[0] = b[0] - gamma
    b_mod[n - 1] = b[n - 1] - a[0] * c[n - 1] / gamma
    
    # Solve A*y = d.
    y = solve_tridiagonal(a, b_mod, c, d)
    
    # Construct u vector.
    u = np.zeros(n, dtype=np.float64)
    u[0] = gamma
    u[n - 1] = c[n - 1]
    
    # Solve A*z = u.
    z = solve_tridiagonal(a, b_mod, c, u)
    
    # Apply Sherman-Morrison correction.
    # v = [1, 0, ..., 0, a[0]/gamma]
    v_dot_y = y[0] + (a[0] / gamma) * y[n - 1]
    v_dot_z = z[0] + (a[0] / gamma) * z[n - 1]
    
    x = y - (v_dot_y / (1.0 + v_dot_z)) * z
    
    return x



