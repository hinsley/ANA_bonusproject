# Test Case 3 (3D): The "Quadratic Decay"

## Overview

This test case validates the 3D heat equation solver against an exact analytical solution using **Robin boundary conditions** and **inhomogeneous forcing**. This is the most complex test case, combining mixed boundary conditions with a polynomial spatial structure.

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
| Robin coefficients | $\alpha = 1$, $\beta = 1$ |

### Exact Solution

$$
u(x,y,z,t) = e^{-t} (1 + x^2 + y^2 + z^2)
$$

### Initial Condition

$$
u_0(x,y,z) = 1 + x^2 + y^2 + z^2
$$

### Forcing Function

$$
F(x,y,z,t) = -e^{-t}(7 + x^2 + y^2 + z^2)
$$

#### Derivation of Forcing

Given $u = e^{-t} (1 + x^2 + y^2 + z^2)$:

1. Time derivative: $u_t = -e^{-t} (1 + x^2 + y^2 + z^2)$
2. Second derivatives: $u_{xx} = 2e^{-t}$, $u_{yy} = 2e^{-t}$, $u_{zz} = 2e^{-t}$
3. Laplacian: $\nabla^2 u = 6e^{-t}$
4. Substituting: $F = u_t - \nabla^2 u = -e^{-t}(1 + x^2 + y^2 + z^2) - 6e^{-t}$
5. Simplifying: $F = -e^{-t}(7 + x^2 + y^2 + z^2)$

### Boundary Conditions

Robin condition on all faces: $\alpha u + \beta \frac{\partial u}{\partial n} = g$

With $\alpha = 1$ and $\beta = 1$:

$$
u + \frac{\partial u}{\partial n} = g(x,y,z,t)
$$

#### Boundary Values

| Face | Normal | $g(x,y,z,t)$ |
|------|--------|--------------|
| $x = 0$ | $-\hat{x}$ | $e^{-t}(1 + y^2 + z^2)$ |
| $x = 1$ | $+\hat{x}$ | $e^{-t}(4 + y^2 + z^2)$ |
| $y = 0$ | $-\hat{y}$ | $e^{-t}(1 + x^2 + z^2)$ |
| $y = 1$ | $+\hat{y}$ | $e^{-t}(4 + x^2 + z^2)$ |
| $z = 0$ | $-\hat{z}$ | $e^{-t}(1 + x^2 + y^2)$ |
| $z = 1$ | $+\hat{z}$ | $e^{-t}(4 + x^2 + y^2)$ |

#### Derivation of $g$ at $x = 1$

1. At $x = 1$: $u = e^{-t}(1 + 1 + y^2 + z^2) = e^{-t}(2 + y^2 + z^2)$
2. Normal derivative: $\frac{\partial u}{\partial n} = \frac{\partial u}{\partial x} = 2x \cdot e^{-t} = 2e^{-t}$
3. Robin condition: $g = u + \frac{\partial u}{\partial n} = e^{-t}(2 + y^2 + z^2) + 2e^{-t} = e^{-t}(4 + y^2 + z^2)$

#### Derivation of $g$ at $x = 0$

1. At $x = 0$: $u = e^{-t}(1 + 0 + y^2 + z^2) = e^{-t}(1 + y^2 + z^2)$
2. Outward normal is $-\hat{x}$, so: $\frac{\partial u}{\partial n} = -\frac{\partial u}{\partial x} = -2x \cdot e^{-t} = 0$
3. Robin condition: $g = u + \frac{\partial u}{\partial n} = e^{-t}(1 + y^2 + z^2) + 0 = e^{-t}(1 + y^2 + z^2)$

## Physical Interpretation

The solution represents a **quadratic temperature profile** that decays uniformly in time:

- The temperature is always positive and ranges from $e^{-t}$ (at origin) to $4e^{-t}$ (at corner $(1,1,1)$)
- At $t = 0$: Temperature ranges from 1 to 4
- At $t = 1$: Temperature ranges from $1/e \approx 0.37$ to $4/e \approx 1.47$

The Robin boundary conditions represent a convective heat transfer scenario where the heat flux is proportional to the temperature difference.

## Why This Test Case?

1. **Robin BCs**: The most general linear boundary condition type.
2. **Polynomial solution**: Tests spatial accuracy without trigonometric complications.
3. **Non-zero forcing**: Verifies correct source term handling.
4. **Spatially varying BCs**: Different $g$ values on different faces.
5. **Time-dependent BCs**: All boundary conditions evolve with $e^{-t}$.

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
- Error may be larger near boundaries due to Robin BC approximation
- Solution should remain positive throughout

## Usage

```bash
python analysis/test_case_3_quadratic_decay.py
```

## Output

- Console: Error analysis at each saved time step
- Animation: `analysis/test_case_3_quadratic_decay.gif`

## Related Files

- Python script: [`analysis/test_case_3_quadratic_decay.py`](../analysis/test_case_3_quadratic_decay.py)
- Solver: [`src/heat_solver/solver_3d.py`](../src/heat_solver/solver_3d.py)

