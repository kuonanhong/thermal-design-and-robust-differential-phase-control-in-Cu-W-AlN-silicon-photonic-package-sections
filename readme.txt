Workload-aware thermal design and robust differential-phase control
in Cu/W–AlN silicon-photonic package sections

Repository purpose

This package supplies Python source, executed numerical data, figures,
and English / Traditional Chinese documents for a silicon-photonic
package co-design study. All device-performance results are numerical
model predictions. There are no measurements of fabricated devices.

Prepared repository URL:
https://github.com/kuonanhong/thermal-design-and-robust-differential-phase-control-in-Cu-W-AlN-silicon-photonic-package-sections

This archive is prepared for the owner's upload. It does not itself
publish files, create a release, submit a paper, or establish a DOI.

Scientific scope

1. Conservative thermal conduction with finite AlN interfaces and
   plane-strain thermoelasticity in an invariant two-dimensional section.
2. EIM optical projection used consistently for background phase and
   heater response, followed by finite-resolution robust phase control.
3. Independent EIC and PIC sources and a fixed command for a declared
   EIC 200–300 mW / PIC 20–30 mW workload box.
4. A separate local full-vector Maxwell benchmark and nominal-load
   phase/control reprojection using vendored WGMODES.

The workload-box and full-vector studies are separate. They do not
supply a combined full-vector workload-box guarantee, a transient
feedback controller, an experimental validation, or a 3D package model.

Installation and reproduction

Use Python 3.12. From the extracted repository root:

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r code/requirements.txt
    python code/reproduce_all.py --verify-only

The compatible requirements file permits supported package versions.
code/requirements-reproduced.txt records the supplied run's versions.
No GPU, COMSOL, proprietary solver, or API key is required. Dependencies
need internet during installation; scientific inputs and solver source
are included. Intel Mac execution uses the CPU, not the AMD GPU.

Verification-only reconstructs supplied optical/control exports and
checks certificates and tabulations. It does not rerun all thermal,
elastic, or Maxwell solves. A fresh run records its own logs and
results/reproduction_verify.json.

Complete scientific regeneration:

    python code/reproduce_all.py

This overwrites generated files in the working copy. Preserve the
original archive or use a separate checkout. The complete run writes
results/reproduction_manifest.json and results/reproduction_logs/.
Read all_passed and step return codes; an old output file does not
prove that a failed new run succeeded.

Programs

code/core.py                     Geometry and physical operators
code/optical_mode.py             Slab modes, EIM and optical coefficients
code/physical.py                 Main physical parameter sweep
code/placement_analysis.py       Guard positions and selected checks
code/optical_sensitivity.py      EIM and photoelastic scenarios
code/control_analysis.py         Nominal/robust LPs and DAC certificates
code/placement_control.py        Placement control comparisons
code/plot_physical.py            Physical scientific figures
code/plot_placement.py           Guard-placement scientific figure
code/workload_robust.py          Source-box commands and lower bounds
code/workload_independent_check.py  Independent workload reconstruction
code/vector_mode.py              Local full-vector Maxwell benchmark
code/vector_target_study.py      Vector 0.25-rad target commands
code/vector_independent_check.py Independent exported vector checks
research/independent_checks.py   Independent base-study checks
code/reproduce_all.py            Ordered scientific reproduction

Preserve code/vendor/wgmodes/, the coefficient JSON files, and all
results directories. Do not upload the .py files alone.

Documents and build

English manuscript:
    manuscript/EN/OLT_Manuscript_EN.pdf
    manuscript/EN/OLT_Manuscript_EN.tex
Traditional Chinese manuscript:
    manuscript/ZH/OLT_Manuscript_ZH.pdf
    manuscript/ZH/OLT_Manuscript_ZH.tex
English technical report:
    reports/EN/OLT_Technical_Report_EN.pdf
    reports/EN/OLT_Technical_Report_EN.tex
Traditional Chinese technical report:
    reports/ZH/OLT_Technical_Report_ZH.pdf
    reports/ZH/OLT_Technical_Report_ZH.tex

After installing XeLaTeX, BibTeX and the required TeX packages:

    python tools/build_documents.py

Fonts and their OFL license are under assets/fonts/. LaTeX is needed
for document/schematic builds, not for the numerical Python pipeline.

Interpretation

C_pre includes sufficient DAC rounding margins; C_post is the exact
interval-box certificate for a specified rounded command. A feasible
command is not a proof of globally minimum integer-DAC power. Separate
vertex commands are not an implemented feedback policy. The optical
efficiency bound excludes propagation/coupling losses and imbalance.

Changing geometry or optical/material parameters requires regenerating
both background phase and H. The principal uncertainty boxes are
deterministic assumptions, not measured distributions or yield data.

License, attribution and citation

Original project software, software documentation and generated
numerical datasets use the MIT license in LICENSE and license.txt.
Manuscript and report prose is outside that software grant; see RIGHTS.md.
WGMODES retains its original MIT copyright, source hashes and provenance.
Noto fonts retain SIL OFL terms. Keep THIRD_PARTY_NOTICES.md and the
original third-party license files when redistributing.

CITATION.cff identifies Nan-Hong Kuo and this software project. Cite the
exact commit/release used, plus the article when it is publicly available.
No unverified affiliation, author list, acceptance, DOI or release date
is asserted by that file.

SHA256SUMS.json records distributed file bytes. Regeneration may change
hashes, timestamps, plots and floating-point formatting; retain the new
run's environment and verification records. docs/final_validation.json
records the checks actually performed for this revised package.