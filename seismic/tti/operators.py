from sympy import cos, sin, sqrt

from devito import Eq, Operator, TimeFunction, NODE
from seismic import PointSource, Receiver
from seismic.tti.rotated_fd import rotated_weighted_lap


def second_order_stencil(model, u, v, space_order):
    """
    Creates the stencil corresponding to the second order TTI wave equation
    u.dt2 =  (epsilon * H0 + delta * Hz) - damp * u.dt
    v.dt2 =  (delta * H0 + Hz) - damp * v.dt
    """
    # Tilt and azymuth setup
    costheta = cos(model.theta)
    sintheta = sin(model.theta)
    cosphi = cos(model.phi)
    sinphi = sin(model.phi)

    H0, Hz = rotated_weighted_lap(u, v, costheta, sintheta, cosphi, sinphi,
                                  model.epsilon, model.delta, model.irho)
    # Stencils
    m, damp = model.m, model.damp
    m = m * model.irho
    s = model.grid.stepping_dim.spacing

    stencilp = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * u + (s * damp - 2.0 * m) *
         u.backward + 2.0 * s ** 2 * H0)
    stencilr = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * v + (s * damp - 2.0 * m) *
         v.backward + 2.0 * s ** 2 * Hz)
    first_stencil = Eq(u.forward, stencilp)
    second_stencil = Eq(v.forward, stencilr)
    stencils = [first_stencil, second_stencil]
    return stencils

def particle_velocity_fields(model, space_order):
    """
    Initialize particle velocity fields for staggered TTI.
    """
    if model.grid.dim == 2:
        x, z = model.space_dimensions
        stagg_x = x
        stagg_z = z
        x, z = model.grid.dimensions
        # Create symbols for forward wavefield, source and receivers
        vx = TimeFunction(name='vx', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order)
        vz = TimeFunction(name='vz', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order)
        vy = None
    elif model.grid.dim == 3:
        x, y, z = model.space_dimensions
        stagg_x = x
        stagg_y = y
        stagg_z = z
        x, y, z = model.grid.dimensions
        # Create symbols for forward wavefield, source and receivers
        vx = TimeFunction(name='vx', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order)
        vy = TimeFunction(name='vy', grid=model.grid, staggered=stagg_y,
                          time_order=1, space_order=space_order)
        vz = TimeFunction(name='vz', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order)

    return vx, vz, vy


def kernel_staggered_2d(model, u, v, space_order):
    """
    TTI finite difference. The equation solved is:
    vx.dt = - u.dx
    vz.dt = - v.dx
    m * v.dt = - sqrt(1 + 2 delta) vx.dx - vz.dz + Fh
    m * u.dt = - (1 + 2 epsilon) vx.dx - sqrt(1 + 2 delta) vz.dz + Fv
    """
    dampl = 1 - model.damp
    m, epsilon, delta, theta = (model.m, model.epsilon, model.delta, model.theta)
    epsilon = 1 + 2 * epsilon
    delta = sqrt(1 + 2 * delta)
    s = model.grid.stepping_dim.spacing
    x, z = model.grid.dimensions
    # Staggered setup
    vx, vz, _ = particle_velocity_fields(model, space_order)

    # Stencils
    phdx = cos(theta) * u.dx - sin(theta) * u.dy
    u_vx = Eq(vx.forward, dampl * vx - dampl * s * phdx)

    pvdz = sin(theta) * v.dx + cos(theta) * v.dy
    u_vz = Eq(vz.forward, dampl * vz - dampl * s * pvdz)

    dvx = cos(theta) * vx.forward.dx - sin(theta) * vx.forward.dy
    dvz = sin(theta) * vz.forward.dx + cos(theta) * vz.forward.dy

    # u and v equations
    pv_eq = Eq(v.forward, dampl * (v - s / m * (delta * dvx + dvz)))

    ph_eq = Eq(u.forward, dampl * (u - s / m * (epsilon * dvx + delta * dvz)))

    return [u_vx, u_vz] + [pv_eq, ph_eq]


def kernel_staggered_3d(model, u, v, space_order):
    """
    TTI finite difference. The equation solved is:
    vx.dt = - u.dx
    vy.dt = - u.dx
    vz.dt = - v.dx
    m * v.dt = - sqrt(1 + 2 delta) (vx.dx + vy.dy) - vz.dz + Fh
    m * u.dt = - (1 + 2 epsilon) (vx.dx + vy.dy) - sqrt(1 + 2 delta) vz.dz + Fv
    """
    dampl = 1 - model.damp
    m, epsilon, delta, theta, phi = (model.m, model.epsilon, model.delta,
                                     model.theta, model.phi)
    epsilon = 1 + 2 * epsilon
    delta = sqrt(1 + 2 * delta)
    s = model.grid.stepping_dim.spacing
    x, y, z = model.grid.dimensions
    # Staggered setup
    vx, vz, vy = particle_velocity_fields(model, space_order)
    # Stencils
    phdx = (cos(theta) * cos(phi) * u.dx +
            cos(theta) * sin(phi) * u.dyc -
            sin(theta) * u.dzc)
    u_vx = Eq(vx.forward, dampl * vx - dampl * s * phdx)

    phdy = -sin(phi) * u.dxc + cos(phi) * u.dy
    u_vy = Eq(vy.forward, dampl * vy - dampl * s * phdy)

    pvdz = (sin(theta) * cos(phi) * v.dxc +
            sin(theta) * sin(phi) * v.dyc +
            cos(theta) * v.dz)
    u_vz = Eq(vz.forward, dampl * vz - dampl * s * pvdz)

    dvx = (cos(theta) * cos(phi) * vx.forward.dx +
           cos(theta) * sin(phi) * vx.forward.dyc -
           sin(theta) * vx.forward.dzc)
    dvy = -sin(phi) * vy.forward.dxc + cos(phi) * vy.forward.dy
    dvz = (sin(theta) * cos(phi) * vz.forward.dxc +
           sin(theta) * sin(phi) * vz.forward.dyc +
           cos(theta) * vz.forward.dz)
    # u and v equations
    pv_eq = Eq(v.forward, dampl * (v - s / m * (delta * (dvx + dvy) + dvz)))

    ph_eq = Eq(u.forward, dampl * (u - s / m * (epsilon * (dvx + dvy) +
                                                delta * dvz)))

    return [u_vx, u_vy, u_vz] + [pv_eq, ph_eq]


def ForwardOperator(model, geometry, space_order=4,
                    save=False, kernel='centered', **kwargs):
    """
    Construct an forward modelling operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    data : ndarray
        IShot() object containing the acquisition geometry and field data.
    time_order : int
        Time discretization order.
    space_order : int
        Space discretization order.
    """

    dt = model.grid.time_dim.spacing
    m = model.m
    time_order = 1 if kernel == 'staggered' else 2
    if kernel == 'staggered':
        stagg_u = stagg_v = NODE
    else:
        stagg_u = stagg_v = None

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid, staggered=stagg_u,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid, staggered=stagg_v,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)
    src = PointSource(name='src', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # FD kernels of the PDE
    FD_kernel = kernels[(kernel, len(model.shape))]
    stencils = FD_kernel(model, u, v, space_order)

    # Source and receivers
    stencils += src.inject(field=u.forward, expr=src * dt**2 / m)
    stencils += src.inject(field=v.forward, expr=src * dt**2 / m)
    stencils += rec.interpolate(expr=u + v)

    # Substitute spacing terms to reduce flops
    return Operator(stencils, subs=model.spacing_map, name='ForwardTTI', **kwargs)


kernels = {('centered', 3): second_order_stencil, ('centered', 2): second_order_stencil,
           ('staggered', 3): kernel_staggered_3d, ('staggered', 2): kernel_staggered_2d}
