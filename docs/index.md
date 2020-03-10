# Devito-Examples

Documentation for the seismic modeling and inversion examples developed at Georgia Institute of Technology with contribution from the rest of the Devito team at Imperial College London (in particular F. Luporini, R. Nelson and G. Bismas).


## Overview

This repository contains a set of examples and tutorials for seismic modeling and inversion using [Devito].
These examples use four different wave equations, namely

- The acoustic isotropic wave equation in`seismic/acoustic`
- The TTI pseudo-acoustic wave equation in `seismic/tti`
- The elastic isotropic wave equation in `seismic/elastic`
- The viscoelastic isotropic wave equation in `seismic/elastic`

Currently, the acoustic isotropic wave equation solver also contains the propagator associated with the adjoint and linearized (Born) wave-equation solution and the gradient of the FWI objective (application of the Jacobian to data residual)

## Disclaimer

A good part of these examples can also be found in the [Devito] examples directory as a fork of this one. These examples for seismic applications have been developed and implemented by [Mathias Louboutin](https://slim.gatech.edu/people/mathias-louboutin) at Georgian Institute of Technology. Some extra examples are also included over there, such as tutorials on the [Devito] compiler as these examples have been developed primarily at Imperial College London.

## Installation

To install this set of examples with its dependencies run in your terminal (OSX, Ubuntu):

```
git clone https://github.com/slimgroup/Devito-Examples
cd Devito-Examples
pip install -e .
```

This command will install all dependencies including [Devito] and will allow you to run the examples. To verify your installation you can run:

```
python seismic/acoustic/acoustic_example.py -nd 1
```

Some of the examples require velocity models such as the marmousi-ii model. These models can be downloaded at [devito-data](https://github.com/devitocodes/data) to be used in the tutorials.


## Related literature

Some of these examples are described in the following papers:

[Devito's symbolic API](https://slim.gatech.edu/content/devito-embedded-domain-specific-language-finite-differences-and-geophysical-exploration)\
[TTI imaging](https://slim.gatech.edu/content/effects-wrong-adjoints-rtm-tti-media)\
[Mathias Louboutin's thesis](https://slim.gatech.edu/content/modeling-inversion-exploration-geophysics)

More advanced geophysical application can be found in the [JUDI] repository. [JUDI] is a linear algebra DSL built on top of [Devito] for large scale inverse problems and includes abstractions for source/receivers and handles large SEG-Y datasets with [SegyIO](https://github.com/slimgroup/SegyIO.jl). A complete description of [JUDI] and the related seismic inversion application can be found in [Philipp Witte's thesis](https://slim.gatech.edu/content/modeling-inversion-exploration-geophysics).

[JUDI]:https://github.com/slimgroup/JUDI.jl
[Devito]:https://www.devitoproject.org
