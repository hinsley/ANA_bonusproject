# 3D Heat Equation Solver: Douglas-Gunn ADI Scheme (Delta Form)

## Overview

The 3D heat equation solver uses the **Douglas-Gunn ADI (Alternating Direction Implicit)** scheme in **delta form** to solve the heat equation:

$$
\frac{\partial u}{\partial t} = c \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right) + F(x, y, z, t)
$$

This scheme is **unconditionally stable** and has **second-order accuracy** in both space and time: $O(\Delta t^2 + \Delta x^2 + \Delta y^2 + \Delta z^2)$.

## Key Feature: Delta Form

Unlike schemes that solve for full solution values at intermediate stages, the Douglas-Gunn **delta form** solves for **increments** (changes) at each fractional step:

- $\Delta u^* =$ increment from X-sweep
- $\Delta u^{**} =$ increment from Y-sweep  
- $\Delta u =$ final increment from Z-sweep
- $u^{n+1} = u^n + \Delta u$

This formulation is efficient and numerically stable.

## Mathematical Formulation

### Discretization

Define the spatial grid and time steps:

- **Grid points:** 
  - $x_i$ for $i = 0, \ldots, N_x$
  - $y_j$ for $j = 0, \ldots, N_y$
  - $z_k$ for $k = 0, \ldots, N_z$

- **Courant-like numbers:**

$$
r_x = \frac{c \cdot \Delta t}{\Delta x^2}, \quad r_y = \frac{c \cdot \Delta t}{\Delta y^2}, \quad r_z = \frac{c \cdot \Delta t}{\Delta z^2}
$$

- **Central difference operator:**

$$
\delta^2_x u_{i,j,k} = u_{i+1,j,k} - 2u_{i,j,k} + u_{i-1,j,k}
$$

### The Douglas-Gunn Scheme (Delta Form)

The scheme consists of four steps per time step:

#### Step 0: Compute Explicit RHS

Calculate the right-hand side using known values at time step $n$:

$$
S_{i,j,k} = \left( r_x \delta^2_x + r_y \delta^2_y + r_z \delta^2_z \right) u^n_{i,j,k} + \frac{1}{2} \left( F^n_{i,j,k} + F^{n+1}_{i,j,k} \right)
$$

#### Step 1: X-Sweep (Solve for $\Delta u^*$)

$$
\left(1 - \frac{r_x}{2} \delta^2_x \right) \Delta u^* = S
$$

For each fixed $(j, k)$ pair, solve the tridiagonal system:

$$
-\frac{r_x}{2} \Delta u^{*}_{i-1} + (1 + r_x) \Delta u^{*}_{i} - \frac{r_x}{2} \Delta u^{*}_{i+1} = S_{i,j,k}
$$

#### Step 2: Y-Sweep (Solve for $\Delta u^{**}$)

$$
\left(1 - \frac{r_y}{2} \delta^2_y \right) \Delta u^{**} = \Delta u^*
$$

For each fixed $(i, k)$ pair, solve the tridiagonal system:

$$
-\frac{r_y}{2} \Delta u^{**}_{j-1} + (1 + r_y) \Delta u^{**}_{j} - \frac{r_y}{2} \Delta u^{**}_{j+1} = \Delta u^{*}_{i,j,k}
$$

#### Step 3: Z-Sweep (Solve for $\Delta u$)

$$
\left(1 - \frac{r_z}{2} \delta^2_z \right) \Delta u = \Delta u^{**}
$$

For each fixed $(i, j)$ pair, solve the tridiagonal system:

$$
-\frac{r_z}{2} \Delta u_{k-1} + (1 + r_z) \Delta u_{k} - \frac{r_z}{2} \Delta u_{k+1} = \Delta u^{**}_{i,j,k}
$$

#### Step 4: Final Update

$$
u^{n+1}_{i,j,k} = u^n_{i,j,k} + \Delta u_{i,j,k}
$$

## Algorithm Flow

```
Initialize u[N_x, N_y, N_z]
Compute coefficients r_x, r_y, r_z

For each time step n → n+1:

    1. Compute Explicit RHS S[i,j,k]:
       δ²_x = u[i+1,j,k] - 2*u[i,j,k] + u[i-1,j,k]
       δ²_y = u[i,j+1,k] - 2*u[i,j,k] + u[i,j-1,k]
       δ²_z = u[i,j,k+1] - 2*u[i,j,k] + u[i,j,k-1]
       S = r_x*δ²_x + r_y*δ²_y + r_z*δ²_z + 0.5*(F^n + F^{n+1})

    2. X-SWEEP:
       For each (j, k):
           Construct tridiagonal system:
               a[i] = -r_x/2, b[i] = 1+r_x, c[i] = -r_x/2
               d[i] = S[i,j,k]
           Apply boundary conditions (Δu = u_bc - u^n at Dirichlet boundaries)
           Solve for Δu*[:,j,k] using Thomas algorithm

    3. Y-SWEEP:
       For each (i, k):
           Construct tridiagonal system:
               a[j] = -r_y/2, b[j] = 1+r_y, c[j] = -r_y/2
               d[j] = Δu*[i,j,k]
           Apply boundary conditions
           Solve for Δu**[i,:,k] using Thomas algorithm

    4. Z-SWEEP:
       For each (i, j):
           Construct tridiagonal system:
               a[k] = -r_z/2, b[k] = 1+r_z, c[k] = -r_z/2
               d[k] = Δu**[i,j,k]
           Apply boundary conditions
           Solve for Δu[i,j,:] using Thomas algorithm

    5. Update: u = u + Δu
    6. Update time: t = t + Δt
```

## Implementation Details

### Key Methods

| Method | Description |
|--------|-------------|
| `__init__` | Initialize solver with domain, diffusivity, BCs, initial condition |
| `step(dt)` | Advance solution by one time step |
| `solve(t_final, dt)` | Solve to final time, optionally saving intermediate solutions |
| `_douglas_gunn_step_x` | X-sweep: compute RHS S and solve for Δu* |
| `_douglas_gunn_step_y` | Y-sweep: solve for Δu** using Δu* as RHS |
| `_douglas_gunn_step_z` | Z-sweep: solve for Δu using Δu** as RHS |
| `_apply_delta_sq_x/y/z` | Apply $\delta^2$ second difference operators |
| `_build_tridiag_coefficients_x/y/z` | Construct tridiagonal matrix coefficients |

### Boundary Condition Handling in Delta Form

Because we solve for increments $\Delta u$ rather than full values, boundary conditions require special handling.

#### Dirichlet Boundary Conditions

For $u = g$ at the boundary:

$$
\Delta u_{\text{boundary}} = g(t^{n+1}) - u^n_{\text{boundary}}
$$

- If the boundary value is **time-independent**: $\Delta u = 0$
- If the boundary value is **time-dependent**: $\Delta u = g(t^{n+1}) - u^n$

#### Neumann and Robin Boundary Conditions (Pseudo-Dirichlet Approach)

For Neumann ($\frac{\partial u}{\partial n} = g$) and Robin ($\alpha u + \beta \frac{\partial u}{\partial n} = g$) conditions, we use a **pseudo-Dirichlet** approach:

1. **Compute the target boundary value** from the BC equation using interior values:
   - **Neumann**: $u_{\text{bc}} = u_{\text{interior}} + h \cdot g(t^{n+1})$
   - **Robin**: $u_{\text{bc}} = \frac{g(t^{n+1}) + (\beta/h) \cdot u_{\text{interior}}}{\alpha + \beta/h}$

2. **Treat as Dirichlet**: $\Delta u_{\text{boundary}} = u_{\text{bc}} - u^n_{\text{boundary}}$

3. **Use identity row** in the tridiagonal system at boundaries.

This approach is stable and provides good convergence, though the boundary accuracy is first-order due to lagging interior values.

### Tridiagonal Solver

Each directional sweep produces a tridiagonal system of the form:

$$
a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i
$$

This is solved efficiently using the **Thomas Algorithm (TDMA)** in $O(n)$ time.

## Usage Example

```python
from heat_solver import HeatSolver3D, DirichletBC, BoundaryConditions3D
from heat_solver.solver_3d import Domain3D
import numpy as np

# Define domain [0,1]³ with 21x21x21 grid.
domain = Domain3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx=21, ny=21, nz=21)

# Define zero Dirichlet boundary conditions on all faces.
zero_bc = lambda coords, t: np.zeros_like(coords[0])
bc = BoundaryConditions3D(
    x_min=DirichletBC(zero_bc),
    x_max=DirichletBC(zero_bc),
    y_min=DirichletBC(zero_bc),
    y_max=DirichletBC(zero_bc),
    z_min=DirichletBC(zero_bc),
    z_max=DirichletBC(zero_bc),
)

# Define initial condition (Gaussian hot spot).
def initial_condition(X, Y, Z):
    r_sq = (X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2
    return np.exp(-r_sq / 0.02)

# Create solver.
solver = HeatSolver3D(
    domain=domain,
    c=0.1,  # Thermal diffusivity.
    bc=bc,
    initial_condition=initial_condition,
)

# Check stability parameters.
params = solver.get_stability_parameters(dt=0.001)
print(f"r_x={params['r_x']:.4f}, r_y={params['r_y']:.4f}, r_z={params['r_z']:.4f}")

# Solve to t=0.5 with dt=0.001, saving every 25 steps.
times, solutions = solver.solve(t_final=0.5, dt=0.001, save_every=25)
```

## Computational Complexity

For a grid of size $N_x \times N_y \times N_z$:

| Operation | Complexity per Time Step |
|-----------|-------------------------|
| Compute RHS S | $O(N_x N_y N_z)$ |
| X-sweep | $O(N_x N_y N_z)$ — solves $N_y \times N_z$ systems of size $N_x$ |
| Y-sweep | $O(N_x N_y N_z)$ — solves $N_x \times N_z$ systems of size $N_y$ |
| Z-sweep | $O(N_x N_y N_z)$ — solves $N_x \times N_y$ systems of size $N_z$ |
| **Total** | $O(N_x N_y N_z)$ |

The scheme is **linear in the total number of grid points**, making it highly efficient for 3D problems.

## Stability

The Douglas-Gunn ADI scheme is **unconditionally stable** for the heat equation. This means:

- There is no restriction on the time step $\Delta t$ for stability.
- Accuracy considerations (not stability) determine the appropriate time step.
- Large time steps are possible when high temporal accuracy is not required.

## Comparison: Douglas-Gunn vs. Other 3D ADI Schemes

| Feature | Douglas-Gunn (Delta Form) | Peaceman-Rachford Extension |
|---------|---------------------------|------------------------------|
| Unknowns | Increments $\Delta u$ | Full values $u$ |
| RHS structure | Simple: previous delta | Complex: correction terms |
| Boundary handling | $\Delta u = u_{bc} - u^n$ | Direct value assignment |
| Implementation | Simpler Y/Z sweeps | More complex RHS computations |

## References

- Douglas, J., & Gunn, J. E. (1964). "A general formulation of alternating direction methods." *Numerische Mathematik*, 6(1), 428-453.
- Morton, K. W., & Mayers, D. F. (2005). *Numerical Solution of Partial Differential Equations*. Cambridge University Press.
- Strikwerda, J. C. (2004). *Finite Difference Schemes and Partial Differential Equations*. SIAM.

