"""
Visualization utilities for heat equation solutions.

Provides functions for:
- 2D: Contour plots, surface plots, animations.
- 3D: Orthogonal slice plots, stacked slices, isosurfaces, animations.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.cm as cm


def plot_solution_2d(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    u: NDArray[np.float64],
    t: Optional[float] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 8),
    ax: Optional[Axes] = None,
    plot_type: str = "contourf",
) -> Tuple[Figure, Axes]:
    """
    Plot a 2D solution as contour or surface plot.

    Parameters
    ----------
    X, Y : ndarray
        Meshgrid coordinate arrays.
    u : ndarray
        Solution values.
    t : float, optional
        Time value for title.
    title : str, optional
        Custom title. Overrides automatic title.
    cmap : str
        Colormap name.
    colorbar : bool
        Whether to show colorbar.
    vmin, vmax : float, optional
        Color scale limits.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Existing axes to plot on.
    plot_type : str
        Type of plot: "contourf", "contour", "surface", or "imshow".

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    if ax is None:
        if plot_type == "surface":
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    if vmin is None:
        vmin = np.min(u)
    if vmax is None:
        vmax = np.max(u)
    
    if plot_type == "contourf":
        levels = np.linspace(vmin, vmax, 50)
        cf = ax.contourf(X, Y, u, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        if colorbar:
            plt.colorbar(cf, ax=ax, label="u")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
    elif plot_type == "contour":
        levels = np.linspace(vmin, vmax, 20)
        cf = ax.contour(X, Y, u, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.clabel(cf, inline=True, fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
    elif plot_type == "imshow":
        extent = [X.min(), X.max(), Y.min(), Y.max()]
        im = ax.imshow(
            u.T, origin='lower', extent=extent, cmap=cmap,
            vmin=vmin, vmax=vmax, aspect='auto'
        )
        if colorbar:
            plt.colorbar(im, ax=ax, label="u")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
    elif plot_type == "surface":
        surf = ax.plot_surface(X, Y, u, cmap=cmap, vmin=vmin, vmax=vmax)
        if colorbar:
            fig.colorbar(surf, ax=ax, shrink=0.5, label="u")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
    
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
    
    if title is not None:
        ax.set_title(title)
    elif t is not None:
        ax.set_title(f"Solution at t = {t:.4f}")
    
    return fig, ax


def animate_solution_2d(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    times: List[float],
    solutions: List[NDArray[np.float64]],
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 8),
    interval: int = 300,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title_prefix: str = "Heat Equation",
    save_path: Optional[str] = None,
) -> animation.FuncAnimation:
    """
    Create an animation of the 2D solution over time.

    Parameters
    ----------
    X, Y : ndarray
        Meshgrid coordinate arrays.
    times : list of float
        Time values.
    solutions : list of ndarray
        Solution arrays at each time.
    cmap : str
        Colormap name.
    figsize : tuple
        Figure size.
    interval : int
        Milliseconds between frames.
    vmin, vmax : float, optional
        Color scale limits. If None, uses global min/max.
    title_prefix : str
        Prefix for animation title.
    save_path : str, optional
        If provided, save animation to this path.

    Returns
    -------
    anim : FuncAnimation
        Matplotlib animation object.
    """
    # Determine global color limits.
    if vmin is None:
        vmin = min(np.min(sol) for sol in solutions)
    if vmax is None:
        vmax = max(np.max(sol) for sol in solutions)
    
    fig, ax = plt.subplots(figsize=figsize)
    levels = np.linspace(vmin, vmax, 50)
    
    # Initial plot.
    cf = ax.contourf(X, Y, solutions[0], levels=levels, cmap=cmap)
    cbar = fig.colorbar(cf, ax=ax, label="u")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(f"{title_prefix}: t = {times[0]:.4f}")
    
    def update(frame):
        ax.clear()
        cf = ax.contourf(X, Y, solutions[frame], levels=levels, cmap=cmap)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title_prefix}: t = {times[frame]:.4f}")
        return [cf]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(times), interval=interval, blit=False
    )
    
    if save_path is not None:
        fps = max(1, int(1000 / interval))
        # Use PillowWriter with optimized settings for better color quality.
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=150)
        plt.close(fig)  # Close figure after saving to prevent display.
    
    return anim


def plot_solution_3d(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    u: NDArray[np.float64],
    t: Optional[float] = None,
    slice_axis: str = "z",
    slice_indices: Optional[List[int]] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (12, 4),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    view_mode: str = "orthogonal",
    orthogonal_indices: Optional[Tuple[int, int, int]] = None,
) -> Tuple[Figure, List[Axes]]:
    """
    Plot slices of a 3D solution with improved defaults.

    Parameters
    ----------
    X, Y, Z : ndarray
        Meshgrid coordinate arrays.
    u : ndarray
        Solution values (shape: nx, ny, nz).
    t : float, optional
        Time value for title.
    slice_axis : str
        Axis to slice along for ``view_mode='stacked'``.
    slice_indices : list of int, optional
        Indices along slice_axis to plot (stacked view).
    cmap : str
        Colormap name.
    figsize : tuple
        Figure size.
    vmin, vmax : float, optional
        Color scale limits.
    view_mode : str
        "orthogonal" (default) to show three orthogonal slices, or "stacked"
        to show multiple slices along a single axis.
    orthogonal_indices : tuple of ints, optional
        Indices for (x_idx, y_idx, z_idx) when using orthogonal view.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : list of Axes
        List of axes objects.
    """
    if vmin is None:
        vmin = float(np.min(u))
    if vmax is None:
        vmax = float(np.max(u))

    def slice_info(axis: str, idx: int) -> Tuple[NDArray, List[float], Tuple[str, str], str]:
        if axis == "x":
            data = u[idx, :, :]
            extent = [Z.min(), Z.max(), Y.min(), Y.max()]
            labels = ("z", "y")
            title = f"x = {X[idx, 0, 0]:.3f}"
        elif axis == "y":
            data = u[:, idx, :].T
            extent = [X.min(), X.max(), Z.min(), Z.max()]
            labels = ("x", "z")
            title = f"y = {Y[0, idx, 0]:.3f}"
        elif axis == "z":
            data = u[:, :, idx].T
            extent = [X.min(), X.max(), Y.min(), Y.max()]
            labels = ("x", "y")
            title = f"z = {Z[0, 0, idx]:.3f}"
        else:
            raise ValueError(f"Unknown axis: {axis}")
        return data, extent, labels, title

    axes_handles: List[Axes] = []

    if view_mode == "orthogonal":
        if orthogonal_indices is None:
            orthogonal_indices = (
                u.shape[0] // 2,
                u.shape[1] // 2,
                u.shape[2] // 2,
            )
        axes_order = ["x", "y", "z"]
        fig, axes = plt.subplots(1, 3, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        images = []
        for ax_obj, axis_name, idx in zip(axes, axes_order, orthogonal_indices):
            data, extent, labels, title = slice_info(axis_name, idx)
            im = ax_obj.imshow(
                data,
                origin='lower',
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='auto',
            )
            ax_obj.set_xlabel(labels[0])
            ax_obj.set_ylabel(labels[1])
            ax_obj.set_title(title)
            images.append(im)
        fig.colorbar(images[-1], ax=axes.tolist(), shrink=0.8, label="u")
        axes_handles = axes.tolist()
    elif view_mode == "stacked":
        if slice_axis not in ("x", "y", "z"):
            raise ValueError(f"Unknown slice_axis: {slice_axis}")
        axis_idx = "xyz".index(slice_axis)
        n_slices = u.shape[axis_idx]
        if slice_indices is None:
            slice_indices = np.linspace(0, n_slices - 1, min(5, n_slices), dtype=int).tolist()
        n_plots = len(slice_indices)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows), squeeze=False)
        axes_flat = axes.flatten()
        images = []
        for ax_obj, idx in zip(axes_flat[:n_plots], slice_indices):
            data, extent, labels, title = slice_info(slice_axis, idx)
            im = ax_obj.imshow(
                data,
                origin='lower',
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='auto',
            )
            ax_obj.set_xlabel(labels[0])
            ax_obj.set_ylabel(labels[1])
            ax_obj.set_title(title)
            images.append(im)
        for ax_obj in axes_flat[n_plots:]:
            ax_obj.set_visible(False)
        fig.colorbar(images[-1], ax=axes_flat[:n_plots].tolist(), label="u", shrink=0.8)
        axes_handles = axes_flat[:n_plots].tolist()
    else:
        raise ValueError("view_mode must be 'orthogonal' or 'stacked'")

    if t is not None:
        fig.suptitle(
            f"3D Solution at t = {t:.4f}",
            fontsize=14,
        )
    fig.tight_layout()
    return fig, axes_handles


def animate_solution_3d(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    times: List[float],
    solutions: List[NDArray[np.float64]],
    slice_axis: str = "z",
    slice_index: Optional[int] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (12, 4),
    interval: int = 300,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title_prefix: str = "Heat Equation 3D",
    save_path: Optional[str] = None,
    view_mode: str = "orthogonal",
    orthogonal_indices: Optional[Tuple[int, int, int]] = None,
) -> animation.FuncAnimation:
    """
    Create an animation of a 2D slice (or orthogonal slices) of the 3D solution.
    """

    def slice_data(axis: str, sol: NDArray, idx: int) -> Tuple[NDArray, List[float], Tuple[str, str], str]:
        if axis == "x":
            data = sol[idx, :, :]
            extent = [Z.min(), Z.max(), Y.min(), Y.max()]
            labels = ("z", "y")
            title = f"x = {X[idx, 0, 0]:.3f}"
        elif axis == "y":
            data = sol[:, idx, :].T
            extent = [X.min(), X.max(), Z.min(), Z.max()]
            labels = ("x", "z")
            title = f"y = {Y[0, idx, 0]:.3f}"
        elif axis == "z":
            data = sol[:, :, idx].T
            extent = [X.min(), X.max(), Y.min(), Y.max()]
            labels = ("x", "y")
            title = f"z = {Z[0, 0, idx]:.3f}"
        else:
            raise ValueError(f"Unknown slice axis: {axis}")
        return data, extent, labels, title

    if vmin is None:
        vmin = min(np.min(sol) for sol in solutions)
    if vmax is None:
        vmax = max(np.max(sol) for sol in solutions)

    if view_mode == "orthogonal":
        if orthogonal_indices is None:
            orthogonal_indices = (
                solutions[0].shape[0] // 2,
                solutions[0].shape[1] // 2,
                solutions[0].shape[2] // 2,
            )
        axes_order = ["x", "y", "z"]
        fig, axes = plt.subplots(1, 3, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        images = []
        for ax_obj, axis_name, idx in zip(axes, axes_order, orthogonal_indices):
            data, extent, labels, title = slice_data(axis_name, solutions[0], idx)
            im = ax_obj.imshow(
                data,
                origin='lower',
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='auto',
            )
            ax_obj.set_xlabel(labels[0])
            ax_obj.set_ylabel(labels[1])
            ax_obj.set_title(title)
            images.append(im)
        fig.colorbar(images[-1], ax=axes.tolist(), shrink=0.8, label="u")
        time_title = fig.suptitle(f"{title_prefix}: t = {times[0]:.4f}")

        def update(frame):
            sol = solutions[frame]
            for im, axis_name, idx in zip(images, axes_order, orthogonal_indices):
                data, _, _, _ = slice_data(axis_name, sol, idx)
                im.set_data(data)
            time_title.set_text(f"{title_prefix}: t = {times[frame]:.4f}")
            return images + [time_title]
    else:
        if slice_axis not in ("x", "y", "z"):
            raise ValueError(f"Unknown slice_axis: {slice_axis}")
        n_slices = solutions[0].shape["xyz".index(slice_axis)]
        if slice_index is None:
            slice_index = n_slices // 2
        fig, ax = plt.subplots(figsize=figsize)
        data, extent, labels, title = slice_data(slice_axis, solutions[0], slice_index)
        im = ax.imshow(
            data,
            origin='lower',
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto',
        )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        title_obj = ax.set_title(f"{title_prefix}: t = {times[0]:.4f}, {title}")
        fig.colorbar(im, ax=ax, label="u")

        def update(frame):
            data, _, _, title_text = slice_data(slice_axis, solutions[frame], slice_index)
            im.set_data(data)
            title_obj.set_text(f"{title_prefix}: t = {times[frame]:.4f}, {title_text}")
            return [im, title_obj]

    anim = animation.FuncAnimation(
        fig, update, frames=len(times), interval=interval, blit=False
    )

    if save_path is not None:
        fps = max(1, int(1000 / interval))
        anim.save(save_path, writer='pillow', fps=fps)

    return anim


def plot_solution_3d_volume(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    u: NDArray[np.float64],
    t: Optional[float] = None,
    isosurface_values: Optional[List[float]] = None,
    opacity: float = 0.5,
    cmap: str = "viridis",
    save_path: Optional[str] = None,
) -> None:
    """
    Create a 3D volume rendering using PyVista.

    Parameters
    ----------
    X, Y, Z : ndarray
        Meshgrid coordinate arrays.
    u : ndarray
        Solution values (shape: nx, ny, nz).
    t : float, optional
        Time value for title.
    isosurface_values : list of float, optional
        Values at which to draw isosurfaces. Default: 5 evenly spaced.
    opacity : float
        Opacity of isosurfaces (0-1).
    cmap : str
        Colormap name.

    Notes
    -----
    Requires pyvista to be installed.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista is required for 3D volume rendering. Install with: pip install pyvista")
        return
    
    # Create structured grid.
    grid = pv.StructuredGrid(X, Y, Z)
    grid["u"] = u.flatten(order='F')
    
    # Determine isosurface values.
    if isosurface_values is None:
        umin, umax = np.min(u), np.max(u)
        isosurface_values = np.linspace(umin, umax, 5)[1:-1].tolist()
    
    off_screen = save_path is not None
    plotter = pv.Plotter(off_screen=off_screen)
    
    # Add isosurfaces.
    for val in isosurface_values:
        iso = grid.contour([val], scalars="u")
        if iso.n_points > 0:
            plotter.add_mesh(iso, opacity=opacity, cmap=cmap)
    
    # Add bounding box.
    plotter.add_mesh(grid.outline(), color='black')
    
    if t is not None:
        plotter.add_title(f"3D Solution at t = {t:.4f}")
    
    plotter.add_axes()
    if off_screen:
        plotter.show(screenshot=save_path)
    else:
        plotter.show()


def save_volume_snapshots(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    times: List[float],
    solutions: List[NDArray[np.float64]],
    output_dir: str,
    cmap: str = "viridis",
    opacity: float = 0.4,
    clim: Optional[Tuple[float, float]] = None,
    title_prefix: str = "3D Heat Equation",
    image_format: str = "png",
) -> List[str]:
    """
    Save volumetric PyVista snapshots at each time step to a folder.

    Parameters
    ----------
    X, Y, Z : ndarray
        Meshgrid coordinates.
    times : list of float
        Time stamps matching `solutions`.
    solutions : list of ndarray
        3D solution fields (nx, ny, nz) for each time.
    output_dir : str
        Directory to save images. Will be created if it doesn't exist.
    cmap : str
        Colormap for the volume.
    opacity : float
        Opacity scalar for the volume (0-1).
    clim : tuple, optional
        Fixed color limits (vmin, vmax). If None, uses global min/max.
    title_prefix : str
        Text prefix drawn on each frame.
    image_format : str
        Image format extension (e.g., "png", "jpg").

    Returns
    -------
    saved_paths : list of str
        List of saved image file paths.
    """
    import os
    
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista is required for volumetric snapshots. Install with: uv pip install pyvista")
        return []

    if len(times) != len(solutions):
        raise ValueError("times and solutions must have the same length.")

    # Create output directory.
    os.makedirs(output_dir, exist_ok=True)

    grid = pv.StructuredGrid(X, Y, Z)
    if clim is None:
        clim = (
            float(min(np.min(sol) for sol in solutions)),
            float(max(np.max(sol) for sol in solutions)),
        )

    saved_paths = []

    for idx, (t, sol) in enumerate(zip(times, solutions)):
        plotter = pv.Plotter(off_screen=True)
        plotter.background_color = "white"
        
        grid["u"] = sol.ravel(order="F")
        plotter.add_volume(
            grid,
            scalars="u",
            cmap=cmap,
            opacity=opacity,
            shade=True,
            clim=clim,
        )
        plotter.add_mesh(grid.outline(), color="black")
        plotter.add_text(f"{title_prefix}: t = {t:.4f}", font_size=12, color="black")
        plotter.add_axes()
        
        # Save screenshot.
        filename = f"volume_{idx:04d}_t{t:.4f}.{image_format}"
        filepath = os.path.join(output_dir, filename)
        plotter.show(screenshot=filepath)
        plotter.close()
        
        saved_paths.append(filepath)
        print(f"  Saved: {filepath}")

    return saved_paths


def create_volume_animation(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    times: List[float],
    solutions: List[NDArray[np.float64]],
    output_path: str = "volume_animation.gif",
    cmap: str = "viridis",
    opacity: Union[float, str] = 0.6,
    clim: Optional[Tuple[float, float]] = None,
    title_prefix: str = "3D Heat Equation",
    fps: int = 10,
    window_size: Tuple[int, int] = (1024, 768),
    camera_position: Optional[Union[str, List]] = "iso",
) -> str:
    """
    Create a 3D volumetric animation (gif) using PyVista.

    Renders the full 3D volume at each time step and combines them into a gif.
    No interactive preview is shown during rendering.

    Parameters
    ----------
    X, Y, Z : ndarray
        Meshgrid coordinates.
    times : list of float
        Time stamps matching `solutions`.
    solutions : list of ndarray
        3D solution fields (nx, ny, nz) for each time.
    output_path : str
        Output file path. Defaults to "volume_animation.gif".
    cmap : str
        Colormap for the volume rendering.
    opacity : float or str
        Opacity for volume rendering. Can be a scalar (0-1), or a string
        like "linear", "sigmoid", "geom" for opacity transfer functions.
        Default is 0.6 for good visibility.
    clim : tuple, optional
        Fixed color limits (vmin, vmax). If None, uses global min/max.
    title_prefix : str
        Text prefix drawn on each frame.
    fps : int
        Frames per second for the gif.
    window_size : tuple
        Size of the rendering window (width, height).
    camera_position : str or list, optional
        Camera position. Use "iso", "xy", "xz", "yz" for preset views, or
        a list of [(camera_x, camera_y, camera_z), (focal_x, focal_y, focal_z),
        (up_x, up_y, up_z)] for custom positioning.

    Returns
    -------
    output_path : str
        Path to the saved gif file.

    Notes
    -----
    Requires pyvista and pillow to be installed.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "PyVista is required for volumetric animations. "
            "Install with: pip install pyvista"
        )

    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for gif output. Install with: pip install pillow"
        )

    import os
    import tempfile
    import shutil

    if len(times) != len(solutions):
        raise ValueError("times and solutions must have the same length.")

    # Ensure output has .gif extension.
    if not output_path.lower().endswith(".gif"):
        output_path = os.path.splitext(output_path)[0] + ".gif"

    # Create structured grid.
    grid = pv.StructuredGrid(X, Y, Z)

    # Determine global color limits.
    if clim is None:
        clim = (
            float(min(np.min(sol) for sol in solutions)),
            float(max(np.max(sol) for sol in solutions)),
        )

    # Disable interactive rendering.
    pv.OFF_SCREEN = True

    # Create a temporary directory for frames.
    temp_dir = tempfile.mkdtemp(prefix="pyvista_anim_")
    frame_paths = []

    # Determine camera position once for consistency across frames.
    # We'll capture it from the first frame's plotter.
    saved_camera_position = None

    try:
        # Render each frame with a fresh plotter (required for volume rendering).
        for idx, (t, sol) in enumerate(zip(times, solutions)):
            # Create fresh plotter for each frame.
            plotter = pv.Plotter(off_screen=True, window_size=window_size)
            plotter.background_color = "white"

            # Update grid with current solution.
            grid["u"] = sol.ravel(order="F")

            # Add volume rendering with current data.
            plotter.add_volume(
                grid,
                scalars="u",
                cmap=cmap,
                opacity=opacity,
                shade=True,
                clim=clim,
            )

            # Add outline and axes.
            plotter.add_mesh(grid.outline(), color="black", line_width=2)
            plotter.add_axes()

            # Set camera position.
            if saved_camera_position is not None:
                # Use saved camera position for consistency.
                plotter.camera_position = saved_camera_position
            elif camera_position == "iso":
                plotter.camera_position = "iso"
                plotter.camera.azimuth = 45
                plotter.camera.elevation = 30
                plotter.reset_camera()
                saved_camera_position = plotter.camera_position
            elif camera_position in ("xy", "xz", "yz"):
                plotter.camera_position = camera_position
                plotter.reset_camera()
                saved_camera_position = plotter.camera_position
            elif camera_position is not None:
                plotter.camera_position = camera_position
                saved_camera_position = plotter.camera_position
            else:
                plotter.reset_camera()
                saved_camera_position = plotter.camera_position

            # Add title with current time.
            plotter.add_text(
                f"{title_prefix}: t = {t:.4f}",
                font_size=14,
                color="black",
                position="upper_left",
            )

            # Save frame.
            frame_path = os.path.join(temp_dir, f"frame_{idx:04d}.png")
            plotter.screenshot(frame_path)
            plotter.close()
            frame_paths.append(frame_path)
            print(f"  Rendered frame {idx + 1}/{len(times)} (t = {t:.4f})")

        # Combine frames into gif using PIL with high-quality palette.
        duration = int(1000 / fps)  # Milliseconds per frame.
        
        # Load and convert images to use adaptive 256-color palettes with dithering.
        # This produces much better color quality than the default.
        images = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")  # Convert to RGB (no alpha).
            # Quantize with maximum colors and Floyd-Steinberg dithering.
            img_p = img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=1)
            images.append(img_p)
        
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=False,  # Don't reduce colors further.
        )

    finally:
        # Clean up temporary directory.
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    print(f"\nAnimation saved to: {output_path}")
    return output_path


def save_solution(
    filepath: str,
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    u: NDArray[np.float64],
    t: float,
    Z: Optional[NDArray[np.float64]] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save solution to a NumPy file.

    Parameters
    ----------
    filepath : str
        Output file path (.npz).
    X, Y : ndarray
        Coordinate arrays.
    u : ndarray
        Solution array.
    t : float
        Time value.
    Z : ndarray, optional
        Z coordinate array for 3D solutions.
    metadata : dict, optional
        Additional metadata to save.
    """
    data = {
        'X': X,
        'Y': Y,
        'u': u,
        't': np.array([t]),
    }
    
    if Z is not None:
        data['Z'] = Z
    
    if metadata is not None:
        for key, value in metadata.items():
            data[f'meta_{key}'] = np.array([value]) if np.isscalar(value) else value
    
    np.savez(filepath, **data)


def load_solution(filepath: str) -> dict:
    """
    Load solution from a NumPy file.

    Parameters
    ----------
    filepath : str
        Input file path (.npz).

    Returns
    -------
    data : dict
        Dictionary with X, Y, u, t, and optionally Z and metadata.
    """
    loaded = np.load(filepath)
    
    data = {
        'X': loaded['X'],
        'Y': loaded['Y'],
        'u': loaded['u'],
        't': loaded['t'][0] if loaded['t'].size == 1 else loaded['t'],
    }
    
    if 'Z' in loaded:
        data['Z'] = loaded['Z']
    
    # Extract metadata.
    metadata = {}
    for key in loaded.keys():
        if key.startswith('meta_'):
            meta_key = key[5:]
            val = loaded[key]
            metadata[meta_key] = val[0] if val.size == 1 else val
    
    if metadata:
        data['metadata'] = metadata
    
    return data


def save_solution_series(
    filepath: str,
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    times: List[float],
    solutions: List[NDArray[np.float64]],
    Z: Optional[NDArray[np.float64]] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save a time series of solutions to a NumPy file.

    Parameters
    ----------
    filepath : str
        Output file path (.npz).
    X, Y : ndarray
        Coordinate arrays.
    times : list of float
        Time values.
    solutions : list of ndarray
        Solution arrays.
    Z : ndarray, optional
        Z coordinate array for 3D solutions.
    metadata : dict, optional
        Additional metadata to save.
    """
    data = {
        'X': X,
        'Y': Y,
        'times': np.array(times),
        'solutions': np.array(solutions),
    }
    
    if Z is not None:
        data['Z'] = Z
    
    if metadata is not None:
        for key, value in metadata.items():
            data[f'meta_{key}'] = np.array([value]) if np.isscalar(value) else value
    
    np.savez(filepath, **data)

