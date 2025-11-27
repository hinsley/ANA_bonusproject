# Heat Equation PDE Solver

Advanced Numerical Analysis bonus project: 2D/3D forced heat equation solver using ADI methods.

## Overview

This package solves the forced heat equation:

$$U_t = c \nabla^2 U + F(\vec{x}, t)$$

on rectangular domains in 2D and 3D with Dirichlet, Neumann, or Robin boundary conditions.

### Numerical Schemes

- **2D**: D'Yakonov ADI scheme (unconditionally stable, second-order accurate)
- **3D**: Adapted Peaceman-Rachford ADI scheme (see note below about verification)

## Installation

Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Quick Start

### 2D Example

```python
import numpy as np
from heat_solver import HeatSolver2D, DirichletBC, BoundaryConditions2D
from heat_solver.solver_2d import Domain2D

# Define domain.
domain = Domain2D(x_min=0, x_max=1, y_min=0, y_max=1, nx=51, ny=51)

# Define boundary conditions (homogeneous Dirichlet).
bc = BoundaryConditions2D(
    x_min=DirichletBC(lambda y, t: np.zeros_like(y)),
    x_max=DirichletBC(lambda y, t: np.zeros_like(y)),
    y_min=DirichletBC(lambda x, t: np.zeros_like(x)),
    y_max=DirichletBC(lambda x, t: np.zeros_like(x)),
)

# Define initial condition.
def u0(X, Y):
    return np.sin(np.pi * X) * np.sin(np.pi * Y)

# Create and run solver.
solver = HeatSolver2D(domain=domain, c=1.0, bc=bc, initial_condition=u0)
times, solutions = solver.solve(t_final=0.5, dt=0.001, save_every=100)
```

### 3D Example

```python
import numpy as np
from heat_solver import HeatSolver3D, DirichletBC, BoundaryConditions3D
from heat_solver.solver_3d import Domain3D

# Define domain.
domain = Domain3D(0, 1, 0, 1, 0, 1, nx=21, ny=21, nz=21)

# Define boundary conditions.
bc = BoundaryConditions3D(
    x_min=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
    x_max=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
    y_min=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
    y_max=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
    z_min=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
    z_max=DirichletBC(lambda coords, t: np.zeros_like(coords[0])),
)

# Define initial condition.
def u0(X, Y, Z):
    return np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)

# Create and run solver.
solver = HeatSolver3D(domain=domain, c=1.0, bc=bc, initial_condition=u0)
times, solutions = solver.solve(t_final=0.1, dt=0.001)
```

## Project Structure

```
ANA_bonusproject/
├── pyproject.toml              # Package configuration
├── src/heat_solver/
│   ├── __init__.py             # Package exports
│   ├── solver_2d.py            # D'Yakonov ADI scheme
│   ├── solver_3d.py            # Peaceman-Rachford 3D adaptation
│   ├── boundary.py             # Boundary condition classes
│   ├── tridiagonal.py          # Thomas algorithm
│   ├── visualization.py        # Plotting and animation
│   └── analysis.py             # Error and convergence analysis
├── problems/
│   ├── problem_2d_example.py   # Customizable 2D problem script
│   └── problem_3d_example.py   # Customizable 3D problem script
├── analysis/
│   ├── exact_solutions.py      # Analytical solutions for testing
│   ├── convergence_2d.py       # 2D convergence study
│   └── convergence_3d.py       # 3D convergence study
└── README.md
```

## Boundary Conditions

Three types of boundary conditions are supported:

### Dirichlet

Specifies the value of u on the boundary:
$$u = g(\vec{x}, t)$$

```python
DirichletBC(lambda y, t: np.sin(np.pi * y) * np.exp(-t))
```

### Neumann

Specifies the normal derivative of u on the boundary:
$$\frac{\partial u}{\partial n} = g(\vec{x}, t)$$

```python
NeumannBC(lambda y, t: np.zeros_like(y))  # Zero-flux BC.
```

### Robin

Mixed boundary condition:
$$\alpha u + \beta \frac{\partial u}{\partial n} = g(\vec{x}, t)$$

```python
RobinBC(alpha=1.0, beta=0.5, g=lambda y, t: np.zeros_like(y))
```

## Customizing Problems

The easiest way to define your own problems is to copy and modify the example scripts in `problems/`:

1. Edit the domain bounds and grid resolution
2. Define your initial condition function
3. Define your forcing function (or set to zero)
4. Define boundary condition functions for each face
5. Run the script

## Convergence Analysis

Run convergence studies to verify the solver accuracy:

```bash
# 2D convergence study
python analysis/convergence_2d.py

# 3D convergence study  
python analysis/convergence_3d.py
```

These scripts test the solver against known analytical solutions and verify the expected second-order convergence.

## Visualization

The package includes built-in visualization:

```python
from heat_solver import plot_solution_2d, animate_solution_2d

# Static plot.
plot_solution_2d(X, Y, u, t, plot_type="contourf")

# Animation.
anim = animate_solution_2d(X, Y, times, solutions, save_path="heat.gif")
```

For 3D, the default view now shows three orthogonal slices (x/y/z planes) for
better spatial awareness. You can still fall back to a stacked single-axis view
by passing `view_mode="stacked"`.

```python
from heat_solver import plot_solution_3d, animate_solution_3d

# Orthogonal slices with custom indices.
plot_solution_3d(
    X, Y, Z, u, t,
    view_mode="orthogonal",
    orthogonal_indices=(X.shape[0] // 4, Y.shape[1] // 2, Z.shape[2] * 3 // 4),
)

# Slower animation (300 ms per frame) of the orthogonal view.
animate_solution_3d(
    X, Y, Z, times, solutions,
    view_mode="orthogonal",
    save_path="heat3d.gif",
)

# Volumetric rendering & animation (requires pip install -e ".[viz3d]")
from heat_solver import plot_solution_3d_volume, animate_solution_3d_volume

plot_solution_3d_volume(X, Y, Z, solutions[-1], times[-1])
animate_solution_3d_volume(
    X, Y, Z, times, solutions,
    save_path="heat3d_volume.gif",
    fps=3,  # slow playback
)
```

## 3D Scheme Verification Note

The 3D Peaceman-Rachford adaptation uses a Douglas-Gunn style three-step splitting. The implementation is clearly marked with `TODO: VERIFY` comments in `src/heat_solver/solver_3d.py`.

If you need to modify the scheme based on your own derivation, the key functions to edit are:
- `_peaceman_rachford_step_x()`
- `_peaceman_rachford_step_y()`
- `_peaceman_rachford_step_z()`

Each step is isolated and documented for easy modification.

## Dependencies

- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- Optional extras:
  - `pip install -e ".[viz3d]"` (or `uv pip install .[viz3d]`) installs `pyvista`
    for true 3D volume rendering.

## License

See LICENSE file.
