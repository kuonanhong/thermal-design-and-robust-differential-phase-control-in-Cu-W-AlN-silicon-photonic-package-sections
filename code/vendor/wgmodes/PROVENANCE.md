# WGMODES source provenance

- Original project: https://github.com/thomas-e-murphy/modesolver
- Official research-group page: https://photonics.umd.edu/software/wgmodes/
- Git commit: `6f54c5e82a8de9d7c78a1c86eef2d658b45bc443`
- Installed distribution metadata: `modesolver 3.0.2`.
- Upstream `modesolver/__init__.py` reports `3.0.0`; the pinned commit and distribution metadata identify the actual source. This upstream inconsistency is preserved unchanged.
- Retrieved 2026-09-25 via pip from the original authors' repository.
- Copyright (c) 2025 Thomas E. Murphy; MIT license, supplied as LICENSE.md.
- All `modesolver` source files in this directory are copied unchanged from the installed distribution. Python bytecode is excluded. The application imports this vendored source directly.
- The wrapper in `code/vector_mode.py` sets a reproducible ARPACK starting vector and records the actual eigenpair residual through the module's imported `eigs` callable; the original solver files are not edited.
- Algorithm reference: A. B. Fallahkhair, K. S. Li, T. E. Murphy, “Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides,” Journal of Lightwave Technology 26 (2008) 1423–1431. https://doi.org/10.1109/JLT.2008.923643

Source checksums are recorded in SOURCE_SHA256.json. The MIT copyright and permission notice must accompany redistribution.
