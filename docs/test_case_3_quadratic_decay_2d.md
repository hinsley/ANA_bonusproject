# Test Case 3 (2D): The "Quadratic Decay"

## Overview

This test case validates the 2D D'Yakonov ADI solver using an exact analytical solution for isotropic diffusion with inhomogeneous forcing and Robin boundary conditions.

## Mathematical Formulation

### PDE

$$\frac{\partial u}{\partial t} = \nabla^2 u + F(x, y, t)$$

### Domain

Unit square: $\Omega = [0, 1] \times [0, 1]$

### Diffusion Coefficient

$$c = 1$$

### Exact Solution

$$u(x, y, t) = e^{-t} (1 + x^2 + y^2)$$

### Initial Condition

$$u_0(x, y) = 1 + x^2 + y^2$$

### Forcing Function

$$F(x, y, t) = -e^{-t} (5 + x^2 + y^2)$$

### Boundary Conditions

Robin condition on all boundaries: $u + \frac{\partial u}{\partial n} = g$

| Boundary | Location | $g(x, y, t)$ |
|----------|----------|--------------|
| Left | $x = 0$ | $e^{-t}(1 + y^2)$ |
| Right | $x = 1$ | $e^{-t}(4 + y^2)$ |
| Bottom | $y = 0$ | $e^{-t}(1 + x^2)$ |
| Top | $y = 1$ | $e^{-t}(4 + x^2)$ |

## Verification

Computing derivatives:

$$\frac{\partial u}{\partial t} = -e^{-t} (1 + x^2 + y^2)$$

$$\nabla^2 u = e^{-t} (2 + 2) = 4e^{-t}$$

Thus:

$$\nabla^2 u + F = 4e^{-t} - e^{-t}(5 + x^2 + y^2) = -e^{-t}(1 + x^2 + y^2) = \frac{\partial u}{\partial t}$$ ✓

### Robin BC Verification (Right Edge, $x = 1$)

$$u + \frac{\partial u}{\partial n} = e^{-t}(1 + 1 + y^2) + e^{-t} \cdot 2 \cdot 1 = e^{-t}(4 + y^2) = g$$ ✓

## Physical Interpretation

The solution represents a quadratic temperature distribution that decays uniformly in time. The Robin boundary conditions model a combination of convective heat transfer and prescribed temperature at the boundaries, which is common in practical heat transfer applications.

## Expected Behavior

- Initial range: $u \in [1, 3]$ (minimum at origin, maximum at $(1, 1)$)
- Uniform exponential decay: all values scale by $e^{-t}$
- At $t = 1.0$: $u \in [0.368, 1.104]$

## Convergence Properties

For the D'Yakonov ADI scheme:

- **Spatial accuracy**: Second-order ( $O(\Delta x^2 + \Delta y^2)$ )
- **Temporal accuracy**: Second-order ( $O(\Delta t^2)$ )
- **Stability**: Unconditionally stable

## Usage

```bash
python analysis/test_case_3_quadratic_decay_2d.py
```

## Output

1. **Convergence study**: Error vs grid size with computed convergence rates
2. **Error analysis**: Time-series of L∞, L², and relative errors
3. **Visualization**: `analysis/test_case_3_quadratic_decay_2d.gif`

