# Devito-Examples


This repository contains a set of examples and tutorials for seismic modeling and inversion using Devito.
These examples use different wave equation namely:
- The acoustic isotropic wave equation in`seismic/acoustic`
- The TTI pseudo-acoustic wave equation in `seismic/tti`
- The elastic isotropic wave equation in `seismic/elastic`

## How to navigate this directory

Examples and tutorials are provided in the form of single Python files and as Jupyter
notebooks.

Jupyter notebooks are files with extension `.ipynb`. To execute these, run
`jupyter notebook`, and then click on the desired notebook in the window that
pops up in your browser.


A set of more advanced examples are available in `seismic`:

* `seismic/tutorials`: A series of Jupyter notebooks of incremental complexity,
  showing a variety of Devito features in the context of seismic inversion
  operators. Among the discussed features are custom stencils, staggered
  grids, and tensor notation.
* `seismic/acoustic`: Example implementations of isotropic acoustic forward,
  adjoint, gradient and born operators, suitable for full-waveform inversion
  methods (FWI).
* `seismic/tti`: Example implementations of several anisotropic acoustic
  forward operators (TTI).
* `seismic/elastic`: Example implementation of an isotropic elastic forward
  operator. `elastic`, unlike `acoustic` and `tti`, fully exploits the
  tensorial nature of the Devito symbolic language.
* `seismic/viscoelastic`: Example implementation of an isotropic viscoelastic
  forward operator. Like `elastic`, `viscoelastic` exploits tensor functions
  for a neat and compact representation of the discretized partial differential
  equations.
