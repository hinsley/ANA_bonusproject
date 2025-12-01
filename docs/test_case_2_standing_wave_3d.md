# Test Case 2 (3D): The "Standing Wave"

## Overview

This test case validates the 3D heat equation solver against an exact analytical solution using **homogeneous Neumann boundary conditions** and **inhomogeneous forcing**. It tests the solver's ability to handle no-flux boundaries and time-dependent source terms.

## Mathematical Formulation

### PDE

$$
\frac{\partial u}{\partial t} = c \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right) + F(x,y,z,t)
$$

### Domain

$$
\Omega = [0, 1] \times [0, 1] \times [0, 1] \quad \text{(Unit Cube)}
$$

### Parameters

| Parameter | Value |
|-----------|-------|
| Diffusion coefficient | $c = 1$ |
| Time range | $t \in [0, 1]$ |

### Exact Solution

$$
u(x,y,z,t) = e^{-t} \cos(\pi x) \cos(\pi y) \cos(\pi z)
$$

### Initial Condition

$$
u_0(x,y,z) = \cos(\pi x) \cos(\pi y) \cos(\pi z)
$$

### Forcing Function

$$
F(x,y,z,t) = (3\pi^2 - 1) e^{-t} \cos(\pi x) \cos(\pi y) \cos(\pi z)
$$

#### Derivation of Forcing

Given $u = e^{-t} \cos(\pi x) \cos(\pi y) \cos(\pi z)$:

1. Time derivative: $u_t = -e^{-t} \cos(\pi x) \cos(\pi y) \cos(\pi z)$
2. Laplacian: $\nabla^2 u = -3\pi^2 e^{-t} \cos(\pi x) \cos(\pi y) \cos(\pi z)$
3. Substituting into PDE: $u_t = \nabla^2 u + F$
4. Solving for $F$: $F = u_t - \nabla^2 u = (-1 + 3\pi^2) e^{-t} \cos(\pi x) \cos(\pi y) \cos(\pi z)$

### Boundary Conditions

Homogeneous Neumann (no-flux) on all six faces:

$$
\frac{\partial u}{\partial n} = 0 \quad \text{on } \partial\Omega
$$

This means:
- $\frac{\partial u}{\partial x}(0, y, z, t) = 0$
- $\frac{\partial u}{\partial x}(1, y, z, t) = 0$
- $\frac{\partial u}{\partial y}(x, 0, z, t) = 0$
- $\frac{\partial u}{\partial y}(x, 1, z, t) = 0$
- $\frac{\partial u}{\partial z}(x, y, 0, t) = 0$
- $\frac{\partial u}{\partial z}(x, y, 1, t) = 0$

**Verification**: Since $\frac{\partial}{\partial x}[\cos(\pi x)] = -\pi\sin(\pi x)$, at $x = 0$ and $x = 1$, we have $\sin(0) = \sin(\pi) = 0$. The same applies to $y$ and $z$ directions.

## Physical Interpretation

The solution represents a **standing wave pattern** that decays with characteristic time $\tau = 1$:

- At $t = 0$: Full amplitude cosine pattern
- At $t = 1$: Amplitude reduced to $1/e \approx 0.37$
- At $t = 2$: Amplitude reduced to $1/e^2 \approx 0.14$

The forcing term $(3\pi^2 - 1) \approx 28.6$ is positive, which means it injects energy to counteract what would otherwise be faster decay due to diffusion.

## Why This Test Case?

1. **Neumann BCs**: Tests the one-sided difference approximation for flux conditions.
2. **Non-zero forcing**: Verifies correct implementation of the source term.
3. **Cosine modes**: Natural eigenfunctions for Neumann problems.
4. **Moderate decay**: $e^{-t}$ decay allows testing over meaningful time scales.
5. **Sign changes**: The cosine pattern has regions of positive and negative values.

## Verification

The Python script computes:

1. **Numerical solution** using the Douglas-Gunn ADI scheme
2. **Exact solution** at each saved time step
3. **Error metrics**:
   - $L^\infty$ error: $\max |u_{\text{num}} - u_{\text{exact}}|$
   - $L^2$ error: $\sqrt{\frac{1}{N}\sum |u_{\text{num}} - u_{\text{exact}}|^2}$
   - Relative error: $\frac{L^\infty \text{ error}}{\max |u_{\text{exact}}|}$

## Expected Results

With grid resolution $31 \times 31 \times 31$ and $\Delta t = 0.005$:

- Relative error should be $O(10^{-3})$ to $O(10^{-2})$
- Error should remain bounded over the simulation
- Neumann BC should preserve total "mass" when forcing is zero

## Usage

```bash
python analysis/test_case_2_standing_wave.py
```

## Output

- Console: Error analysis at each saved time step
- Animation: `analysis/test_case_2_standing_wave.gif`

## Related Files

- Python script: [`analysis/test_case_2_standing_wave.py`](../analysis/test_case_2_standing_wave.py)
- Solver: [`src/heat_solver/solver_3d.py`](../src/heat_solver/solver_3d.py)

