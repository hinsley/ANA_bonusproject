# Test Case 1 (3D): The "Decaying Bubble"

## Overview

This test case validates the 3D heat equation solver against an exact analytical solution using **homogeneous Dirichlet boundary conditions** and **zero forcing**. It represents the classic problem of heat diffusing from an initial sinusoidal "bubble" in a cube with all boundaries held at zero temperature.

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
| Time range | $t \in [0, 0.1]$ |
| Forcing | $F = 0$ |

### Exact Solution

$$
u(x,y,z,t) = e^{-3\pi^2 t} \sin(\pi x) \sin(\pi y) \sin(\pi z)
$$

### Initial Condition

$$
u_0(x,y,z) = \sin(\pi x) \sin(\pi y) \sin(\pi z)
$$

### Boundary Conditions

Homogeneous Dirichlet on all six faces:

$$
u = 0 \quad \text{on } \partial\Omega
$$

This includes:
- $u(0, y, z, t) = 0$
- $u(1, y, z, t) = 0$
- $u(x, 0, z, t) = 0$
- $u(x, 1, z, t) = 0$
- $u(x, y, 0, t) = 0$
- $u(x, y, 1, t) = 0$

## Physical Interpretation

The solution represents a **sinusoidal temperature distribution** that decays exponentially in time. The decay rate is $3\pi^2 \approx 29.6$, which means:

- At $t = 0$: Maximum amplitude = 1.0
- At $t = 0.05$: Amplitude $\approx 0.23$ (77% decay)
- At $t = 0.1$: Amplitude $\approx 0.05$ (95% decay)

The fast decay is due to the eigenvalue of the Laplacian for the $\sin(\pi x)\sin(\pi y)\sin(\pi z)$ mode being $-3\pi^2$.

## Why This Test Case?

1. **Smooth initial data**: The sinusoidal initial condition is infinitely differentiable.
2. **Zero forcing**: Tests the pure diffusion dynamics without external sources.
3. **Homogeneous Dirichlet BCs**: The simplest boundary condition type.
4. **Fast decay**: Tests solver accuracy as the solution approaches machine precision.
5. **Symmetry preservation**: The solver should maintain the 8-fold symmetry of the solution.

## Verification

The Python script computes:

1. **Numerical solution** using the Douglas-Gunn ADI scheme
2. **Exact solution** at each saved time step
3. **Error metrics**:
   - $L^\infty$ error: $\max |u_{\text{num}} - u_{\text{exact}}|$
   - $L^2$ error: $\sqrt{\frac{1}{N}\sum |u_{\text{num}} - u_{\text{exact}}|^2}$
   - Relative error: $\frac{L^\infty \text{ error}}{\max |u_{\text{exact}}|}$

## Expected Results

With grid resolution $31 \times 31 \times 31$ and $\Delta t = 0.0005$:

- Relative error should be $O(10^{-4})$ to $O(10^{-3})$
- Error should remain stable as solution decays
- No spurious oscillations or negative values

## Usage

```bash
python analysis/test_case_1_decaying_bubble.py
```

## Output

- Console: Error analysis at each saved time step
- Animation: `analysis/test_case_1_decaying_bubble.gif`

## Related Files

- Python script: [`analysis/test_case_1_decaying_bubble.py`](../analysis/test_case_1_decaying_bubble.py)
- Solver: [`src/heat_solver/solver_3d.py`](../src/heat_solver/solver_3d.py)

