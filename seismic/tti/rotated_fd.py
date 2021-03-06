from sympy import sqrt
from devito import  centered, first_derivative, transpose

def laplacian(v, irho):
    """
    Laplacian with density div( 1/rho grad) (u)
    """
    if irho is None or irho == 1:
        Lap = v.laplace
    else:
        if getattr(irho, 'is_Function', False):
            Lap = grad(irho).T * grad(v) + irho * v.laplace
        else:
            Lap = irho * v.laplace

    return Lap


def rotated_weighted_lap(u, v, costheta, sintheta, cosphi, sinphi,
                         epsilon, delta, irho, fw=True):
    """
    TTI finite difference kernel. The equation we solve is:
    u.dt2 = (1+2 *epsilon) (Gxx(u)) + sqrt(1+ 2*delta) Gzz(v)
    v.dt2 = sqrt(1+ 2*delta) (Gxx(u)) +  Gzz(v)
    where epsilon and delta are the thomsen parameters. This function computes
    H0 = Gxx(u) + Gyy(u)
    Hz = Gzz(v)

    Parameters
    ----------
    u: first TTI field
    v: second TTI field
    costheta: cosine of the tilt angle
    sintheta:  sine of the tilt angle
    cosphi: cosine of the azymuth angle, has to be 0 in 2D
    sinphi: sine of the azymuth angle, has to be 0 in 2D
    space_order: discretization order

    Returns
    -------
    u and v component of the rotated Laplacian in 2D
    """
    epsilon = 1 + 2 * epsilon
    delta = sqrt(1 * 2 * delta)
    if fw:
        Gxx = Gxxyy(u, costheta, sintheta, cosphi, sinphi, irho)
        Gzzr = Gzz(v, costheta, sintheta, cosphi, sinphi, irho)
        return (epsilon * Gxx + delta * Gzzr, delta * Gxx + Gzzr)
    else:
        a = epsilon * u + delta * v
        b = delta * u + v
        H0 = Gxxyy(a, costheta, sintheta, cosphi, sinphi, irho)
        H1 = Gzz(b, costheta, sintheta, cosphi, sinphi, irho)
        return H0, H1


def Gzz(field, costheta, sintheta, cosphi, sinphi, irho):
    """
    3D rotated second order derivative in the direction z

    Parameters
    ----------
    field: symbolic data whose derivative we are computing
    costheta: cosine of the tilt angle
    sintheta:  sine of the tilt angle
    cosphi: cosine of the azymuth angle
    sinphi: sine of the azymuth angle
    space_order: discretization order

    Returns
    -------
    rotated second order derivative wrt z
    """
    if field.grid.dim == 2:
        return Gzz2d(field, costheta, sintheta, irho)

    order1 = field.space_order // 2
    x, y, z = field.grid.dimensions
    Gz = -(sintheta * cosphi * first_derivative(field, dim=x, side=centered,
                                                fd_order=order1) +
           sintheta * sinphi * first_derivative(field, dim=y, side=centered,
                                                fd_order=order1) +
           costheta * first_derivative(field, dim=z, side=centered,
                                       fd_order=order1))
    Gzz = (first_derivative(Gz * sintheta * cosphi * irho,
                            dim=x, side=centered, fd_order=order1,
                            matvec=transpose) +
           first_derivative(Gz * sintheta * sinphi * irho,
                            dim=y, side=centered, fd_order=order1,
                            matvec=transpose) +
           first_derivative(Gz * costheta * irho,
                            dim=z, side=centered, fd_order=order1,
                            matvec=transpose))
    return Gzz


def Gzz2d(field, costheta, sintheta, irho):
    """
    3D rotated second order derivative in the direction z

    Parameters
    ----------
    field: symbolic data whose derivative we are computing
    costheta: cosine of the tilt angle
    sintheta:  sine of the tilt angle
    cosphi: cosine of the azymuth angle
    sinphi: sine of the azymuth angle
    space_order: discretization order

    Returns
    -------
    rotated second order derivative wrt ztranspose
    """
    if sintheta == 0:
        return getattr(field, 'd%s2'%field.grid.dimensions[-1])

    order1 = field.space_order // 2
    x, z = field.grid.dimensions
    Gz = -(sintheta * first_derivative(field, dim=x, side=centered, fd_order=order1) +
           costheta * first_derivative(field, dim=z, side=centered, fd_order=order1))
    Gzz = (first_derivative(Gz * sintheta * irho, dim=x, side=centered,
                            fd_order=order1, matvec=transpose) +
           first_derivative(Gz * costheta * irho, dim=z, side=centered,
                            fd_order=order1, matvec=transpose))
    return Gzz


# Centered case produces directly Gxx + Gyy
def Gxxyy(field, costheta, sintheta, cosphi, sinphi, irho):
    """
    Sum of the 3D rotated second order derivative in the direction x and y.
    As the Laplacian is rotation invariant, it is computed as the conventional
    Laplacian minus the second order rotated second order derivative in the direction z
    Gxx + Gyy = field.laplace - Gzz

    Parameters
    ----------
    field: symbolic data whose derivative we are computing
    costheta: cosine of the tilt angle
    sintheta:  sine of the tilt angle
    cosphi: cosine of the azymuth angle
    sinphi: sine of the azymuth angle
    space_order: discretization order

    Returns
    -------
    Sum of the 3D rotated second order derivative in the direction x and y
    """
    if sintheta == 0 and sinphi == 0:
        return sum([getattr(field, 'd%s2'%d) for d in field.grid.dimensions[:-1]])

    lap = laplacian(field, irho)
    if field.grid.dim == 2:
        Gzzr = Gzz2d(field, costheta, sintheta, irho)
    else:
        Gzzr = Gzz(field, costheta, sintheta, cosphi, sinphi, irho)
    return lap - Gzzr
