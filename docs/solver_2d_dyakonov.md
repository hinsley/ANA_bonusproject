# 2D Heat Equation Solver: D'Yakonov ADI Scheme

## Overview

The 2D heat equation solver uses the **D'Yakonov ADI (Alternating Direction Implicit)** scheme to solve the heat equation:

$$
\frac{\partial u}{\partial t} = c \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) + F(x, y, t)
$$

This scheme is **unconditionally stable** and has **second-order accuracy** in both space and time: $O(\Delta t^2 + \Delta x^2 + \Delta y^2)$.

## Mathematical Formulation

### Discretization

Define the spatial grid and time steps:

- **Grid points:** $x_i$ for $i = 0, \ldots, N_x$ and $y_j$ for $j = 0, \ldots, N_y$
- **Courant-like numbers:**

$$
r_x = \frac{c \cdot \Delta t}{\Delta x^2}, \quad r_y = \frac{c \cdot \Delta t}{\Delta y^2}
$$

- **Central difference operator:**

$$
\delta^2_x u_{i,j} = u_{i+1,j} - 2u_{i,j} + u_{i-1,j}
$$

### The D'Yakonov Scheme

The scheme consists of two fractional steps per time step:

#### Step 1: X-direction implicit, Y-direction explicit

$$
\left(1 - \frac{r_x}{2} \delta^2_x \right) u^* = \left(1 + \frac{r_x}{2} \delta^2_x \right) \left(1 + \frac{r_y}{2} \delta^2_y \right) u^n + \frac{\Delta t}{2} \left( F^n + F^{n+1} \right)
$$

This equation couples nodes in the x-direction only. For each fixed $j$, we solve a tridiagonal system:

$$
-\frac{r_x}{2} u^{\*}_{i-1,j} + (1 + r_x) u^{\*}_{i,j} - \frac{r_x}{2} u^{\*}_{i+1,j} = \text{RHS}_{i,j}
$$

Where the RHS is computed by applying explicit operators to $u^n$.

#### Step 2: Y-direction implicit

$$
\left(1 - \frac{r_y}{2} \delta^2_y \right) u^{n+1} = u^*
$$

For each fixed $i$, we solve a tridiagonal system:

$$
-\frac{r_y}{2} u^{n+1}_{i,j-1} + (1 + r_y) u^{n+1}_{i,j} - \frac{r_y}{2} u^{n+1}_{i,j+1} = u^*_{i,j}
$$

### Explicit Operator Expansion

The explicit operator $(1 + \frac{r}{2} \delta^2)$ applied to $u$ produces:

$$
\left(1 + \frac{r}{2} \delta^2 \right) u_i = \frac{r}{2} u_{i-1} + (1 - r) u_i + \frac{r}{2} u_{i+1}
$$

## Algorithm Flow

```
For each time step n → n+1:
    
    1. Compute Courant numbers r_x, r_y
    
    2. Compute explicit RHS:
       temp = (1 + r_y/2 * δ²_y) u^n       [apply y-operator first]
       RHS  = (1 + r_x/2 * δ²_x) temp      [then x-operator]
       RHS += Δt/2 * (F^n + F^{n+1})       [add forcing]
    
    3. X-SWEEP (Step 1):
       For each j = 0 to N_y:
           Construct tridiagonal system with coefficients:
               a[i] = -r_x/2  (sub-diagonal)
               b[i] = 1 + r_x (main diagonal)
               c[i] = -r_x/2  (super-diagonal)
               d[i] = RHS[i,j]
           Apply boundary conditions at i=0 and i=N_x
           Solve for u*[:,j] using Thomas algorithm
    
    4. Y-SWEEP (Step 2):
       For each i = 0 to N_x:
           Construct tridiagonal system with coefficients:
               a[j] = -r_y/2  (sub-diagonal)
               b[j] = 1 + r_y (main diagonal)
               c[j] = -r_y/2  (super-diagonal)
               d[j] = u*[i,j]
           Apply boundary conditions at j=0 and j=N_y
           Solve for u^{n+1}[i,:] using Thomas algorithm
    
    5. Update time: t = t + Δt
```

## Implementation Details

### Key Methods

| Method | Description |
|--------|-------------|
| `__init__` | Initialize solver with domain, diffusivity, BCs, initial condition |
| `step(dt)` | Advance solution by one time step |
| `solve(t_final, dt)` | Solve to final time, optionally saving intermediate solutions |
| `_apply_explicit_operator_x/y` | Apply $(1 + r/2 \cdot \delta^2)$ operator |
| `_build_tridiag_coefficients_x/y` | Construct tridiagonal matrix coefficients |

### Boundary Condition Handling

The solver supports three types of boundary conditions:

1. **Dirichlet:** $u = g(x,t)$ — Solution value is prescribed at the boundary.
2. **Neumann:** $\frac{\partial u}{\partial n} = g(x,t)$ — Normal derivative is prescribed.
3. **Robin:** $\alpha u + \beta \frac{\partial u}{\partial n} = g(x,t)$ — Mixed condition.

Boundary conditions modify the tridiagonal matrix coefficients at the first and last rows.

### Tridiagonal Solver

Each directional sweep produces a tridiagonal system of the form:

$$
a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i
$$

This is solved efficiently using the **Thomas Algorithm (TDMA)** in $O(n)$ time.

## Usage Example

```python
from heat_solver import HeatSolver2D, DirichletBC, BoundaryConditions2D
from heat_solver.solver_2d import Domain2D
import numpy as np

# Define domain [0,1] x [0,1] with 51x51 grid.
domain = Domain2D(0.0, 1.0, 0.0, 1.0, nx=51, ny=51)

# Define boundary conditions (zero Dirichlet on all edges).
bc = BoundaryConditions2D(
    x_min=DirichletBC(lambda y, t: np.zeros_like(y)),
    x_max=DirichletBC(lambda y, t: np.zeros_like(y)),
    y_min=DirichletBC(lambda x, t: np.zeros_like(x)),
    y_max=DirichletBC(lambda x, t: np.zeros_like(x)),
)

# Define initial condition (Gaussian bump).
def initial_condition(X, Y):
    return np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.02)

# Create solver.
solver = HeatSolver2D(
    domain=domain,
    c=0.1,  # Thermal diffusivity.
    bc=bc,
    initial_condition=initial_condition,
)

# Solve to t=0.5 with dt=0.001.
times, solutions = solver.solve(t_final=0.5, dt=0.001, save_every=50)
```

## Note: Full Form vs Delta Form

Unlike the 3D Douglas-Gunn solver which uses **delta form** (solving for increments $\Delta u$), the 2D D'Yakonov scheme solves for **full solution values** $u^*$ and $u^{n+1}$ directly. This means:

- Boundary conditions are applied directly: $u = g(t^{n+1})$ at Dirichlet boundaries
- No subtraction of current values is needed for Robin/Neumann BCs
- The intermediate variable $u^*$ represents an approximate solution, not an increment

## Stability

The D'Yakonov ADI scheme is **unconditionally stable** for the heat equation. This means:

- There is no restriction on the time step $\Delta t$ for stability.
- Accuracy considerations (not stability) determine the appropriate time step.
- Large time steps are possible when high temporal accuracy is not required.

## References

- D'Yakonov, E. G. (1963). *Difference schemes with a "disintegrating operator" for multidimensional unsteady problems.*
- Morton, K. W., & Mayers, D. F. (2005). *Numerical Solution of Partial Differential Equations*. Cambridge University Press.

