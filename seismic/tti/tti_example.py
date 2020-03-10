from argparse import ArgumentParser
from devito import configuration
from seismic import demo_model, setup_geometry
from seismic.tti import AnisotropicWaveSolver


def tti_setup(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
              space_order=4, nbl=10, preset='layers-tti', density=False,
              vti=False, **kwargs):

    # Two layer model for true velocity
    model = demo_model(preset, shape=shape, spacing=spacing, nbl=nbl,
                       density=density, vti=vti)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    return AnisotropicWaveSolver(model, geometry,
                                 space_order=space_order, **kwargs)


def run(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
        autotune=False, time_order=2, space_order=4, nbl=10,
        kernel='centered', density=False, vti=False, **kwargs):

    solver = tti_setup(shape, spacing, tn, space_order, nbl,
                       density=density, vti=vti, **kwargs)

    rec, u, v, summary = solver.forward(autotune=autotune, kernel=kernel)

    return summary.gflopss, summary.oi, summary.timings, [rec, u, v]


if __name__ == "__main__":
    description = ("Example script to execute a TTI forward operator.")
    parser = ArgumentParser(description=description)
    parser.add_argument('-nd', dest='ndim', default=3, type=int,
                        help="Preset to determine the number of dimensions")
    parser.add_argument('--noazimuth', dest='azi', default=False, action='store_true',
                        help="Whether or not to use an azimuth angle")
    parser.add_argument('-rho', '--density', default=False, action='store_true',
                        help="Whether to include density")
    parser.add_argument('-vti', '--vti', default=False, action='store_true',
                        help="VTI modeling, no tilt or azimuth")
    parser.add_argument('-a', '--autotune', default='off',
                        choices=(configuration._accepted['autotuning']),
                        help="Operator auto-tuning mode")
    parser.add_argument("-so", "--space_order", default=4,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbl", default=40,
                        type=int, help="Number of boundary layers around the domain")
    parser.add_argument("-k", dest="kernel", default='centered',
                        choices=['centered', 'staggered'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("-dse", default="advanced",
                        choices=["noop", "basic", "advanced", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced", choices=["noop", "advanced"],
                        help="Devito loop engine (DLE) mode")
    args = parser.parse_args()

    preset = 'layers-tti-noazimuth' if args.azi else 'layers-tti'
    # 3D preset parameters
    if args.ndim == 2:
        shape = (150, 150)
        spacing = (10.0, 10.0)
        tn = 750.0
    elif args.ndim == 3:
        shape = (150, 150, 150)
        spacing = (10.0, 10.0, 10.0)
        tn = 750.0
    else:
        ValueError("One dimensional tti wave equation does not exist")

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn,
        space_order=args.space_order, autotune=args.autotune, dse=args.dse,
        dle=args.dle, kernel=args.kernel, preset=preset, rho=args.density,
        vti=args.vti)
