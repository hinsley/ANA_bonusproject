# Test Case 2 (2D): The "Standing Wave"

## Overview

This test case validates the 2D D'Yakonov ADI solver using an exact analytical solution for isotropic diffusion with inhomogeneous forcing and homogeneous Neumann boundary conditions.

## Mathematical Formulation

### PDE

$$\frac{\partial u}{\partial t} = \nabla^2 u + F(x, y, t)$$

### Domain

Unit square: $\Omega = [0, 1] \times [0, 1]$

### Diffusion Coefficient

$$c = 1$$

### Exact Solution

$$u(x, y, t) = e^{-t} \cos(\pi x) \cos(\pi y)$$

### Initial Condition

$$u_0(x, y) = \cos(\pi x) \cos(\pi y)$$

### Forcing Function

$$F(x, y, t) = (2\pi^2 - 1) e^{-t} \cos(\pi x) \cos(\pi y)$$

### Boundary Conditions

Homogeneous Neumann on all boundaries:

$$\frac{\partial u}{\partial n} = 0 \quad \text{on } \partial\Omega$$

## Verification

Computing derivatives:

$$\frac{\partial u}{\partial t} = -e^{-t} \cos(\pi x) \cos(\pi y)$$

$$\nabla^2 u = -2\pi^2 e^{-t} \cos(\pi x) \cos(\pi y)$$

Thus:

$$\nabla^2 u + F = -2\pi^2 e^{-t} \cos(\pi x)\cos(\pi y) + (2\pi^2 - 1) e^{-t} \cos(\pi x)\cos(\pi y)$$
$$= -e^{-t} \cos(\pi x) \cos(\pi y) = \frac{\partial u}{\partial t}$$ ✓

## Physical Interpretation

The forcing term exactly balances most of the diffusion, resulting in a slow exponential decay (with rate 1 instead of $2\pi^2$). This represents a scenario where an external heat source partially maintains the temperature pattern against diffusive spreading.

## Expected Behavior

- Initial pattern: cosine wave with maximum at corners (value 1) and minimum at center (value 1)
- Slow exponential decay: $u_{\max}(t) = e^{-t}$
- At $t = 1.0$: $u_{\max} \approx 0.368$

## Convergence Properties

For the D'Yakonov ADI scheme:

- **Spatial accuracy**: Second-order ( $O(\Delta x^2 + \Delta y^2)$ )
- **Temporal accuracy**: Second-order ( $O(\Delta t^2)$ )
- **Stability**: Unconditionally stable

## Usage

```bash
python analysis/test_case_2_standing_wave_2d.py
```

## Output

1. **Convergence study**: Error vs grid size with computed convergence rates
2. **Error analysis**: Time-series of L∞, L², and relative errors
3. **Visualization**: `analysis/test_case_2_standing_wave_2d.gif`

