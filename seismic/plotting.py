import numpy as np
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mpl.rc('font', size=16)
    mpl.rc('figure', figsize=(8, 6))
except:
    plt = None
    cm = None


def plot_perturbation(model, model1, colorbar=True):
    """
    Plot a two-dimensional velocity perturbation from two seismic `Model`
    objects.

    Parameters
    ----------
    model : Model
        The first velocity model.
    model1 : Model
        The second velocity model.
    colorbar : bool
        Option to plot the colorbar.
    """
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]
    dv = model.vp.data.T - model1.vp.data.T

    scale = max(dv.reshape(-1))/5
    plot = plt.imshow(dv, animated=True, cmap=cm.Greys,
                      vmin=-scale, vmax=scale,
                      extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity perturbation (km/s)')
    plt.show()


def plot_velocity(model, source=None, receiver=None, colorbar=True, cmap="jet"):
    """
    Plot a two-dimensional velocity field from a seismic `Model`
    object. Optionally also includes point markers for sources and receivers.

    Parameters
    ----------
    model : Model
        Object that holds the velocity model.
    source : array_like or float
        Coordinates of the source point.
    receiver : array_like or float
        Coordinates of the receiver points.
    colorbar : bool
        Option to plot the colorbar.
    """
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    field = (getattr(model, 'vp', None) or getattr(model, 'lam')).data[slices]
    plot = plt.imshow(np.transpose(field), animated=True, cmap=cmap,
                      vmin=np.min(field), vmax=np.max(field),
                      extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')

    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(1e-3*receiver[:, 0], 1e-3*receiver[:, 1],
                    s=25, c='green', marker='D')

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(1e-3*source[:, 0], 1e-3*source[:, 1],
                    s=25, c='red', marker='o')

    # Ensure axis limits
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity (km/s)')
    plt.show()


def plot_shotrecord(rec, model, t0, tn, colorbar=True):
    """
    Plot a shot record (receiver values over time).

    Parameters
    ----------
    rec :
        Receiver data with shape (time, points).
    model : Model
        object that holds the velocity model.
    t0 : int
        Start of time dimension to plot.
    tn : int
        End of time dimension to plot.
    """
    scale = np.max(rec) / 5.
    extent = [model.origin[0], model.origin[0] + 1e-3*model.domain_size[0],
              1e-3*tn, t0]

    plot = plt.imshow(rec, vmin=-scale, vmax=scale, cmap=cm.Greys, extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Time (s)')

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.show()


def plot_image(im, colorbar=True, vmin=None, vmax=None,
               cmap="Greys", depth_scaling=False, diff=False):
    """
    Plot image data, such as RTM images or FWI gradients.

    Parameters
    ----------
    data : ndarray
        Image data to plot.
    cmap : str
        Choice of colormap. Defaults to gray scale for images as a
        seismic convention.
    """
    extent = [0, 1, 1, 0]
    if getattr(im, 'is_Function', False):
        extent = [im.grid.origin[0].data, im.grid.extent[0],
                  im.grid.extent[1], im.grid.origin[1].data]
        im = im.data
    if diff:
        im = np.gradient(im)[-1]

    if depth_scaling:
        depth = np.diag(np.linspace(1.0, 2.0, im.shape[-1])**1.4)
        im = np.dot(depth, im)

    scale = np.max(im)
    plot = plt.imshow(im.T, extent=extent, vmin=vmin or -scale,
                      vmax=vmax or scale, cmap=cmap)
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.show()
