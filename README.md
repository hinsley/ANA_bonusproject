# Heat Equation PDE Solver

Advanced Numerical Analysis bonus project: 2D/3D forced heat equation solver using ADI methods.

## Overview

This package solves the forced heat equation:

$$\frac{\partial u}{\partial t} = c \nabla^2 u + F(\vec{x}, t)$$

on rectangular domains in 2D and 3D with Dirichlet, Neumann, or Robin boundary conditions.

### Numerical Schemes

- **2D**: D'Yakonov ADI scheme — unconditionally stable, second-order accurate in space and time
- **3D**: Douglas-Gunn ADI scheme (delta form) — unconditionally stable, second-order accurate in space and time

See the detailed documentation in `docs/` for mathematical derivations.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Install uv (if not already installed)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

```bash
# Install base dependencies
uv sync

# Install with 3D visualization support (PyVista + Pillow)
uv sync --extra viz3d

# Install with development tools
uv sync --extra dev
```

## Running Scripts

Use `uv run` to execute scripts with the correct environment:

```bash
# Run a test case
uv run python analysis/test_case_1_decaying_bubble_2d.py

# Run all 2D test cases
uv run python analysis/test_case_1_decaying_bubble_2d.py
uv run python analysis/test_case_2_standing_wave_2d.py
uv run python analysis/test_case_3_quadratic_decay_2d.py

# Run all 3D test cases (requires viz3d extra for animations)
uv run python analysis/test_case_1_decaying_bubble_3d.py
uv run python analysis/test_case_2_standing_wave_3d.py
uv run python analysis/test_case_3_quadratic_decay_3d.py
```

## Test Cases

Six test cases with known exact solutions are provided for solver verification:

| Test Case | BC Type | Exact Solution | Description |
|-----------|---------|----------------|-------------|
| 1. Decaying Bubble | Dirichlet | $e^{-k\pi^2 t} \sin(\pi x)\sin(\pi y)...$ | Heat diffusing to cold boundaries |
| 2. Standing Wave | Neumann | $e^{-t} \cos(\pi x)\cos(\pi y)...$ | Forced solution with no-flux BCs |
| 3. Quadratic Decay | Robin | $e^{-t}(1 + x^2 + y^2 + ...)$ | Mixed BCs with polynomial solution |

Each test case:
- Runs a **spatial convergence study** to verify second-order accuracy
- Compares numerical solution to **exact analytical solution**
- Generates an **animated gif** visualization
- Saves a **report file** with all results

### Output Files

After running a test case, you'll find:
- `analysis/test_case_*_report.txt` — Full text report with convergence rates and errors
- `analysis/test_case_*.gif` — Animated visualization

## Project Structure

```
ANA_bonusproject/
├── pyproject.toml                  # Package configuration & dependencies
├── uv.lock                         # Locked dependency versions
├── README.md
├── LICENSE
│
├── src/heat_solver/                # Main package
│   ├── __init__.py                 # Public API exports
│   ├── solver_2d.py                # 2D D'Yakonov ADI scheme
│   ├── solver_3d.py                # 3D Douglas-Gunn ADI scheme
│   ├── boundary.py                 # Dirichlet, Neumann, Robin BCs
│   ├── tridiagonal.py              # Thomas algorithm (TDMA)
│   ├── visualization.py            # 2D/3D plotting and animation
│   └── analysis.py                 # Error metrics and utilities
│
├── analysis/                       # Test cases with exact solutions
│   ├── test_case_1_decaying_bubble_2d.py
│   ├── test_case_1_decaying_bubble_3d.py
│   ├── test_case_2_standing_wave_2d.py
│   ├── test_case_2_standing_wave_3d.py
│   ├── test_case_3_quadratic_decay_2d.py
│   └── test_case_3_quadratic_decay_3d.py
│
└── docs/                           # Mathematical documentation
    ├── solver_2d_dyakonov.md       # 2D scheme derivation
    ├── solver_3d_douglas_gunn.md   # 3D scheme derivation
    └── test_case_*.md              # Test case specifications
```

## Quick Start

### 2D Example

```python
import numpy as np
from heat_solver import HeatSolver2D, DirichletBC, BoundaryConditions2D
from heat_solver.solver_2d import Domain2D

# Define domain: unit square with 51x51 grid
domain = Domain2D(x_min=0, x_max=1, y_min=0, y_max=1, nx=51, ny=51)

# Homogeneous Dirichlet BCs (u=0 on boundary)
bc = BoundaryConditions2D(
    x_min=DirichletBC(lambda y, t: np.zeros_like(y)),
    x_max=DirichletBC(lambda y, t: np.zeros_like(y)),
    y_min=DirichletBC(lambda x, t: np.zeros_like(x)),
    y_max=DirichletBC(lambda x, t: np.zeros_like(x)),
)

# Initial condition: Gaussian hot spot
def u0(X, Y):
    return np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.02)

# Create solver and run
solver = HeatSolver2D(domain=domain, c=1.0, bc=bc, initial_condition=u0)
times, solutions = solver.solve(t_final=0.5, dt=0.001, save_every=50)

# Create animation
from heat_solver import animate_solution_2d
X, Y, _, _ = solver.get_solution()
animate_solution_2d(X, Y, times, solutions, save_path="heat_2d.gif")
```

### 3D Example

```python
import numpy as np
from heat_solver import HeatSolver3D, DirichletBC, BoundaryConditions3D
from heat_solver.solver_3d import Domain3D

# Define domain: unit cube with 31x31x31 grid
domain = Domain3D(0, 1, 0, 1, 0, 1, nx=31, ny=31, nz=31)

# Homogeneous Dirichlet BCs
zero_bc = DirichletBC(lambda coords, t: np.zeros_like(coords[0]))
bc = BoundaryConditions3D(
    x_min=zero_bc, x_max=zero_bc,
    y_min=zero_bc, y_max=zero_bc,
    z_min=zero_bc, z_max=zero_bc,
)

# Initial condition
def u0(X, Y, Z):
    return np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)

# Create solver and run
solver = HeatSolver3D(domain=domain, c=1.0, bc=bc, initial_condition=u0)
times, solutions = solver.solve(t_final=0.1, dt=0.001, save_every=10)

# Create 3D volumetric animation (requires viz3d extra)
from heat_solver import create_volume_animation
X, Y, Z, _, _ = solver.get_solution()
create_volume_animation(X, Y, Z, times, solutions, output_path="heat_3d.gif")
```

## Boundary Conditions

Three types of boundary conditions are supported:

### Dirichlet

Specifies the value of $u$ on the boundary:
$$u = g(\vec{x}, t)$$

```python
DirichletBC(lambda y, t: np.sin(np.pi * y) * np.exp(-t))
```

### Neumann

Specifies the normal derivative on the boundary:
$$\frac{\partial u}{\partial n} = g(\vec{x}, t)$$

```python
NeumannBC(lambda y, t: np.zeros_like(y))  # Zero-flux (insulated)
```

### Robin

Mixed boundary condition:
$$\alpha u + \beta \frac{\partial u}{\partial n} = g(\vec{x}, t)$$

```python
RobinBC(alpha=1.0, beta=1.0, g=lambda y, t: some_function(y, t))
```

## Visualization

### 2D Animation

```python
from heat_solver import animate_solution_2d

animate_solution_2d(
    X, Y, times, solutions,
    cmap="inferno",
    save_path="animation.gif",
    interval=100,  # ms between frames
)
```

### 3D Volumetric Animation

Requires the `viz3d` extra (`uv sync --extra viz3d`).

```python
from heat_solver import create_volume_animation

create_volume_animation(
    X, Y, Z, times, solutions,
    output_path="volume.gif",
    cmap="inferno",
    opacity=0.7,
    fps=15,
    camera_position=[(3.0, 1.8, 2.0), (0.5, 0.5, 0.5), (0, 0, 1)],
)
```

## Dependencies

### Required
- numpy ≥ 1.24.0
- scipy ≥ 1.10.0
- matplotlib ≥ 3.7.0

### Optional (viz3d)
- pyvista ≥ 0.43.0 — 3D volumetric rendering
- pillow ≥ 10.0.0 — GIF creation

### Development (dev)
- pytest, black, ruff

## Documentation

Detailed mathematical documentation is available in `docs/`:

- [`solver_2d_dyakonov.md`](docs/solver_2d_dyakonov.md) — 2D D'Yakonov ADI scheme
- [`solver_3d_douglas_gunn.md`](docs/solver_3d_douglas_gunn.md) — 3D Douglas-Gunn ADI scheme
- `test_case_*.md` — Test case specifications with exact solutions

## License

See LICENSE file.
