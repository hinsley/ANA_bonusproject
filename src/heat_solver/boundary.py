"""
Boundary condition classes for the heat equation solver.

This module provides abstractions for specifying boundary conditions:
    - Dirichlet: u = g(x, t)
    - Neumann: du/dn = g(x, t)
    - Robin: alpha*u + beta*du/dn = g(x, t)

Boundary conditions are specified per face of the domain (x_min, x_max,
y_min, y_max, and for 3D: z_min, z_max).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray


class BCType(Enum):
    """Enumeration of boundary condition types."""
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"


class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions."""
    
    @property
    @abstractmethod
    def bc_type(self) -> BCType:
        """Return the type of boundary condition."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        coords: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """
        Evaluate the boundary condition at given coordinates and time.

        Parameters
        ----------
        coords : ndarray
            Coordinates on the boundary. For 2D, shape is (n_points, 1) for
            edges. For 3D, shape is (n_points_1, n_points_2, 2) for faces.
        t : float
            Current time.

        Returns
        -------
        values : ndarray
            Boundary condition values at the specified points.
        """
        pass


@dataclass
class DirichletBC(BoundaryCondition):
    """
    Dirichlet boundary condition: u = g(x, t).

    Parameters
    ----------
    g : callable
        Function g(coords, t) -> values specifying the boundary values.
        For 2D edges: g(y_or_x, t) -> u_values.
        For 3D faces: g(coord1, coord2, t) -> u_values.
    """
    g: Callable[..., NDArray[np.float64]]
    
    @property
    def bc_type(self) -> BCType:
        return BCType.DIRICHLET
    
    def evaluate(
        self,
        coords: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Evaluate u = g(coords, t) at boundary points."""
        return self.g(coords, t)


@dataclass
class NeumannBC(BoundaryCondition):
    """
    Neumann boundary condition: du/dn = g(x, t).

    Here, n is the outward normal direction. The derivative is approximated
    using one-sided finite differences in the solver.

    Parameters
    ----------
    g : callable
        Function g(coords, t) -> values specifying the normal derivative.
        For 2D edges: g(y_or_x, t) -> du/dn values.
        For 3D faces: g(coord1, coord2, t) -> du/dn values.
    """
    g: Callable[..., NDArray[np.float64]]
    
    @property
    def bc_type(self) -> BCType:
        return BCType.NEUMANN
    
    def evaluate(
        self,
        coords: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Evaluate du/dn = g(coords, t) at boundary points."""
        return self.g(coords, t)


@dataclass
class RobinBC(BoundaryCondition):
    """
    Robin boundary condition: alpha*u + beta*du/dn = g(x, t).

    This is a linear combination of Dirichlet and Neumann conditions.
    Special cases:
        - alpha=1, beta=0: Dirichlet
        - alpha=0, beta=1: Neumann

    Parameters
    ----------
    alpha : float
        Coefficient for u in the Robin condition.
    beta : float
        Coefficient for du/dn in the Robin condition.
    g : callable
        Function g(coords, t) -> values specifying the RHS.
        For 2D edges: g(y_or_x, t) -> g values.
        For 3D faces: g(coord1, coord2, t) -> g values.
    """
    alpha: float
    beta: float
    g: Callable[..., NDArray[np.float64]]
    
    def __post_init__(self):
        if self.alpha == 0 and self.beta == 0:
            raise ValueError("At least one of alpha or beta must be nonzero.")
    
    @property
    def bc_type(self) -> BCType:
        return BCType.ROBIN
    
    def evaluate(
        self,
        coords: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Evaluate g(coords, t) for alpha*u + beta*du/dn = g."""
        return self.g(coords, t)


@dataclass
class BoundaryConditions2D:
    """
    Container for boundary conditions on a 2D rectangular domain.

    The domain is [x_min, x_max] x [y_min, y_max]. Boundary conditions
    are specified for each of the four edges.

    Parameters
    ----------
    x_min : BoundaryCondition
        BC on the left edge (x = x_min), function of (y, t).
    x_max : BoundaryCondition
        BC on the right edge (x = x_max), function of (y, t).
    y_min : BoundaryCondition
        BC on the bottom edge (y = y_min), function of (x, t).
    y_max : BoundaryCondition
        BC on the top edge (y = y_max), function of (x, t).
    """
    x_min: BoundaryCondition
    x_max: BoundaryCondition
    y_min: BoundaryCondition
    y_max: BoundaryCondition


@dataclass
class BoundaryConditions3D:
    """
    Container for boundary conditions on a 3D rectangular domain.

    The domain is [x_min, x_max] x [y_min, y_max] x [z_min, z_max].
    Boundary conditions are specified for each of the six faces.

    Parameters
    ----------
    x_min : BoundaryCondition
        BC on the x = x_min face, function of (y, z, t).
    x_max : BoundaryCondition
        BC on the x = x_max face, function of (y, z, t).
    y_min : BoundaryCondition
        BC on the y = y_min face, function of (x, z, t).
    y_max : BoundaryCondition
        BC on the y = y_max face, function of (x, z, t).
    z_min : BoundaryCondition
        BC on the z = z_min face, function of (x, y, t).
    z_max : BoundaryCondition
        BC on the z = z_max face, function of (x, y, t).
    """
    x_min: BoundaryCondition
    x_max: BoundaryCondition
    y_min: BoundaryCondition
    y_max: BoundaryCondition
    z_min: BoundaryCondition
    z_max: BoundaryCondition


# Alias for backward compatibility.
BoundaryConditions = Union[BoundaryConditions2D, BoundaryConditions3D]


def apply_dirichlet_bc_1d(
    u: NDArray[np.float64],
    bc_left: BoundaryCondition,
    bc_right: BoundaryCondition,
    coords_left: NDArray[np.float64],
    coords_right: NDArray[np.float64],
    t: float,
) -> None:
    """
    Apply Dirichlet boundary conditions to a 1D array (in-place).

    This is a helper function used when sweeping along lines in ADI methods.
    Only Dirichlet BCs directly set boundary values; Neumann/Robin are
    handled through the tridiagonal system.

    Parameters
    ----------
    u : ndarray
        Solution array to modify (1D or along one axis).
    bc_left : BoundaryCondition
        Boundary condition at the left end.
    bc_right : BoundaryCondition
        Boundary condition at the right end.
    coords_left : ndarray
        Coordinates for the left boundary.
    coords_right : ndarray
        Coordinates for the right boundary.
    t : float
        Current time.
    """
    if bc_left.bc_type == BCType.DIRICHLET:
        u[0] = bc_left.evaluate(coords_left, t)
    if bc_right.bc_type == BCType.DIRICHLET:
        u[-1] = bc_right.evaluate(coords_right, t)


def get_bc_coefficients(
    bc: BoundaryCondition,
    h: float,
    is_min_boundary: bool,
) -> tuple[float, float, float]:
    """
    Get coefficients for incorporating boundary conditions into tridiagonal system.

    For the equation at a boundary point, we modify the tridiagonal coefficients
    to incorporate the boundary condition. This returns (a, b, c) modifications.

    For Dirichlet: The boundary value is known, so we use identity row.
    For Neumann: du/dn = g => use one-sided difference.
    For Robin: alpha*u + beta*du/dn = g => combine the above.

    Parameters
    ----------
    bc : BoundaryCondition
        The boundary condition to apply.
    h : float
        Grid spacing in the normal direction.
    is_min_boundary : bool
        True if this is the min boundary (left/bottom/front), False for max.

    Returns
    -------
    a_coef : float
        Coefficient for the sub-diagonal (or modification factor).
    b_coef : float
        Coefficient for the main diagonal.
    c_coef : float
        Coefficient for the super-diagonal.
    """
    if bc.bc_type == BCType.DIRICHLET:
        # Identity row: u_boundary = g.
        return (0.0, 1.0, 0.0)
    
    elif bc.bc_type == BCType.NEUMANN:
        # du/dn = g. Using one-sided difference:
        # For min boundary (outward normal is -x): -du/dx = g => u[0] - u[1] = -h*g.
        #   => u[0] = u[1] - h*g => row: [1, -1] with RHS modification.
        # For max boundary (outward normal is +x): du/dx = g => u[n] - u[n-1] = h*g.
        #   => u[n] = u[n-1] + h*g => row: [-1, 1] with RHS modification.
        if is_min_boundary:
            return (0.0, 1.0, -1.0)
        else:
            return (-1.0, 1.0, 0.0)
    
    elif bc.bc_type == BCType.ROBIN:
        # alpha*u + beta*du/dn = g.
        # Combine Dirichlet and Neumann contributions.
        alpha = bc.alpha
        beta = bc.beta
        
        if is_min_boundary:
            # -beta*du/dx + alpha*u = g.
            # Using one-sided: -beta*(u[1]-u[0])/h + alpha*u[0] = g.
            # => (alpha + beta/h)*u[0] - (beta/h)*u[1] = g.
            b_coef = alpha + beta / h
            c_coef = -beta / h
            return (0.0, b_coef, c_coef)
        else:
            # beta*du/dx + alpha*u = g.
            # Using one-sided: beta*(u[n]-u[n-1])/h + alpha*u[n] = g.
            # => -(beta/h)*u[n-1] + (alpha + beta/h)*u[n] = g.
            a_coef = -beta / h
            b_coef = alpha + beta / h
            return (a_coef, b_coef, 0.0)
    
    else:
        raise ValueError(f"Unknown boundary condition type: {bc.bc_type}")


def get_bc_rhs_contribution(
    bc: BoundaryCondition,
    coords: NDArray[np.float64],
    t: float,
    h: float,
    is_min_boundary: bool,
) -> NDArray[np.float64]:
    """
    Get RHS contribution from boundary condition.

    Parameters
    ----------
    bc : BoundaryCondition
        The boundary condition.
    coords : ndarray
        Coordinates at the boundary.
    t : float
        Current time.
    h : float
        Grid spacing in the normal direction.
    is_min_boundary : bool
        True if this is the min boundary.

    Returns
    -------
    rhs : ndarray
        RHS contribution for the tridiagonal system.
    """
    g = bc.evaluate(coords, t)
    
    if bc.bc_type == BCType.DIRICHLET:
        return g
    
    elif bc.bc_type == BCType.NEUMANN:
        # The RHS is modified by the Neumann condition.
        # For min boundary: u[0] - u[1] = -h*g => RHS contribution is -h*g.
        # For max boundary: u[n] - u[n-1] = h*g => RHS contribution is h*g.
        if is_min_boundary:
            return -h * g
        else:
            return h * g
    
    elif bc.bc_type == BCType.ROBIN:
        # RHS is just g for the combined equation.
        return g
    
    else:
        raise ValueError(f"Unknown boundary condition type: {bc.bc_type}")

