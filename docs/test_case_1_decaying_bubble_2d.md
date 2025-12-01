# Test Case 1 (2D): The "Decaying Bubble"

## Overview

This test case validates the 2D D'Yakonov ADI solver using an exact analytical solution for isotropic diffusion with homogeneous Dirichlet boundary conditions.

## Mathematical Formulation

### PDE

$$\frac{\partial u}{\partial t} = \nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$$

### Domain

Unit square: $\Omega = [0, 1] \times [0, 1]$

### Diffusion Coefficient

$$c = 1$$

### Exact Solution

$$u(x, y, t) = e^{-2\pi^2 t} \sin(\pi x) \sin(\pi y)$$

### Initial Condition

$$u_0(x, y) = \sin(\pi x) \sin(\pi y)$$

### Forcing Function

$$F(x, y, t) = 0$$

### Boundary Conditions

Homogeneous Dirichlet on all boundaries:

$$u = 0 \quad \text{on } \partial\Omega$$

## Verification

The exact solution satisfies:

$$\frac{\partial u}{\partial t} = -2\pi^2 e^{-2\pi^2 t} \sin(\pi x) \sin(\pi y)$$

$$\nabla^2 u = -2\pi^2 e^{-2\pi^2 t} \sin(\pi x) \sin(\pi y)$$

Thus $\partial u / \partial t = \nabla^2 u$ ✓

## Physical Interpretation

The solution represents a "bubble" of heat centered in the domain that decays exponentially as heat diffuses out to the cold (zero temperature) boundaries. The sinusoidal pattern is preserved over time, only decreasing in amplitude.

## Expected Behavior

- Initial maximum at center $(0.5, 0.5)$ with value 1
- Exponential decay: $u_{\max}(t) = e^{-2\pi^2 t} \approx e^{-19.74 t}$
- At $t = 0.1$: $u_{\max} \approx 0.138$
- At $t = 0.5$: $u_{\max} \approx 5.2 \times 10^{-5}$
- At $t = 1.0$: $u_{\max} \approx 2.7 \times 10^{-9}$

## Convergence Properties

For the D'Yakonov ADI scheme:

- **Spatial accuracy**: Second-order ( $O(\Delta x^2 + \Delta y^2)$ )
- **Temporal accuracy**: Second-order ( $O(\Delta t^2)$ )
- **Stability**: Unconditionally stable

## Usage

```bash
python analysis/test_case_1_decaying_bubble_2d.py
```

## Output

1. **Convergence study**: Error vs grid size with computed convergence rates
2. **Error analysis**: Time-series of L∞, L², and relative errors
3. **Visualization**: `analysis/test_case_1_decaying_bubble_2d.gif`

