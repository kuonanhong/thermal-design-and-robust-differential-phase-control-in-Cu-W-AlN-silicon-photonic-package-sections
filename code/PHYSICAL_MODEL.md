# Reproducible package model

`python physical.py` produces every package and heater response used in the new article. The main output `../results/control_inputs.npz` contains 48 cases: 18 interface cases, 18 AlN-conductivity cases and 12 mesh-refinement cases. Each case requires ten distinct load solutions: one passive EIC/PIC heat source and nine unit heater sources. The file `physical_inputs.npz` retains temperature, signed stress and total/elastic strain samples independently of the optical model, allowing optical sensitivity changes without rerunning mechanics.

## Coordinates and optical interpretation

The solved section spans x=0..1.2 mm and y=0..0.6 mm. The medium and sources are invariant in z. The nine sampled optical channels lie at y=350 µm, x=420,465,...,780 µm and propagate in z for an assumed 1 mm length. A plane-strain displacement solution imposes total εzz=0. Therefore the optical path-length perturbation is zero; εxx must **not** be treated as longitudinal waveguide elongation. The local 450 nm × 220 nm Si waveguide is an analytical submodel; the package grid does not resolve its core or oxide cladding.

The optical model supplies derivatives for stress-free material temperature, signed elastic strain and affine cross-section dimensions. Elastic strain is εe=εtotal−αSi ΔT I, including εezz=−αSiΔT. This avoids counting stress-free thermal expansion twice in the material thermo-optic index response. The waveguide width and height changes use total εxx and εyy respectively. Affine transfer from package strain to the submicrometre core is a modeling assumption; a resolved anisotropic local mechanical submodel is not claimed.

## Geometry and loading

Substrate: y<100 µm; Si interposer:100..300 µm; oxide layer:300..320 µm. PIC:x=240..840 µm,y=320..400 µm; EIC:x=340..740 µm,y=420..520 µm; interchip oxide:y=400..420 µm over the PIC footprint. Eight 40 µm-wide TSV strips extend through y=100..300 µm at x=100,240,380,520,660,800,940,1080 µm. These are 2D strips, not resolved circular 3D cylinders. The graded design uses W fractions 1 for |x−600 µm|<100 µm, 0.5 for 100..250 µm and 0 beyond. Arithmetic interpolation of k,E,ν,α is an illustrative effective-property assumption, not a qualified Cu/W process recipe.

AlN walls occupy x=300..340 and 860..900 µm, y=220..320 µm. The layout is asymmetric relative to the EIC/PIC heat source centered at x=540 µm; its differential-phase findings are layout dependent. Their volume replaces underlying Si and oxide; these are substantial 40×100 µm heat paths, not a thin conformal AlN film. The cross-section is inherited to make ablations interpretable, not optimized a priori for differential optical phase.

The EIC and PIC supply 250 and 25 W/m along z. For the assumed 1 mm depth these correspond to 250 and 25 mW. The bottom surface has effective Robin conductance 100000 W/(m² K), the top 500 W/(m² K), and sides are adiabatic. Ambient and stress-free reference are 25 °C. These boundary values are declared scenario assumptions and must be measured for a specific package.

## Conservative thermal interfaces

For adjacent cells a,b with face area per unit depth Af and distances da,db from centers to the face, transmissibility is

Gf = Af / (da/ka + RK + db/kb).

RK is applied once only when exactly one adjacent cell is AlN. All other interfaces are ideal. Thus the contact resistance is between AlN and neighboring material, not a blanket layer resistance, and mechanical interfaces stay perfectly bonded. RK spans ideal contact, 10⁻¹⁰,10⁻⁹,10⁻⁸,3×10⁻⁸,10⁻⁷,3×10⁻⁷,10⁻⁶ m² K/W. The wide range is a sensitivity range, not a measured distribution of this device. AlN k=36,80,170,260 W/(m K) probes film/process variation; the chosen geometry is not itself a thin-film characterization sample.

## Mechanics

Bilinear Q4 elements use 2×2 Gauss integration, isotropic plane-strain D and the correct thermal load β=EαΔT/(1−2ν). The bottom is constrained only in uy; one lower-left ux degree removes rigid translation. Other displacement boundaries are traction free. σzz is recovered as λ(εxx+εyy)−β; von Mises stress includes σzz. Material junction peaks can be singular and are not used as converged fracture or reliability metrics. No cure residual stress, plasticity, delamination or creep is modeled.

## Physical heater matrix

Each basis heater injects precisely 1 W/m along z, using a normalized Gaussian in x with σ=12 µm centered above its channel and a uniform y=380..400 µm source band inside the PIC. Exact cell integrals preserve injected power at every mesh. This is an equivalent distributed heat source, not resolved metal resistivity or optical-absorption geometry. For Lz=1 mm, 1 W/m equals 1 mW, so the computed phase response is numerically H in rad/mW. Each H column includes thermal expansion, signed photoelasticity and affine cross-section change from its own solved temperature field. With temperature-independent properties, φ(p)=φ0+Hp is an exact superposition of this linear reduced model; no arbitrary cross-talk matrix is introduced.

The heater system can change mean temperature and stress significantly. The power-limited controller uses nonnegative input powers; it does not assert an ideal cooler or subtract arbitrary phase by signed actuation. The meaningful optical error is differential phase (max φ−min φ or specified pairwise MZI mismatch), not the background mean phase.

## Verification and model limits

The interface solver reproduces an exact series-resistance slab test to 3.27×10⁻¹³ relative error. Ideal-contact temperatures reproduce the original conservative solver to 8.00×10⁻¹⁵. Cached mechanics reproduces direct assembly to 3.52×10⁻¹⁵. An affine mechanical patch test yields 1.10×10⁻¹⁵ displacement error; a uniform-heating plane-strain test yields 8.10×10⁻¹⁴ relative strain error. Thermal superposition agrees to 4.50×10⁻¹⁶ and the largest heater power normalization error is 2.22×10⁻¹⁶ W/m. These are numerical-verification results, not experimental validation.

Mesh refinement uses 60×30,120×60,180×90,240×120 cells with aligned layer/TSV/wall boundaries. Extracted temperatures, phase spread and heater response are reported at each grid. Solver residual or energy conservation alone does not establish mesh convergence; corner-stress maxima are not claimed to converge. The model excludes 3D end effects, resolved waveguide mechanics, anisotropic package stiffness, transient control, fabrication residual stresses, and experimentally calibrated film/interface/photoelastic parameters.

## Constructive placement study and final dataset

`python placement_analysis.py` solves eighteen additional reported cases: AlN or graded-W+AlN, left wall center 180/320/460 µm and right wall center 600/740/880 µm. Both wall widths remain 40 µm, with equal total AlN area and fixed y=220..320 µm. Every candidate must avoid all TSV cells; overlapping candidates raise an exception. All eighteen use kAlN=170 W/(m K), RK=10⁻⁸ m² K/W and 120×60 cells. The two original (320,880) configurations duplicate main-study cases as a cross-check. This is a finite candidate comparison; no global placement optimum is claimed.

`python placement_analysis.py --validation` adds twenty-four cases: the three selected positions (320,600), (460,740), (180,880), each with both material assignments, on 180×90 and 240×120 grids (twelve cases); the phase-selected (320,600) position with both material assignments at RK=10⁻⁷ and 10⁻⁶ (four cases); and the same position at kAlN=36 or 260 W/(m K), RK=10⁻⁸ or 10⁻⁷ (eight cases). The complete package computation therefore reports ninety configurations and nine hundred solved load cases, or eighty-eight unique physical configurations after removing the two duplicate original-placement checks.

The AlN-only (320,600) candidate has mean channel temperature 30.41948 °C and passive phase spread 0.2292537 rad, versus Cu 34.83179 °C and 0.3944732 rad. The passive phase spread is reduced by 41.8836%, while mean channel temperature is reduced by 4.41231 K. The temperature-selected AlN-only (460,740) candidate is cooler at 30.16322 °C but has phase spread 0.5587925 rad. The original AlN-only placement (320,880) has temperature 32.20972 °C and phase spread 0.7698804 rad. This explicitly separates mean temperature, passive differential phase and actuator-aware control objectives. The low-temperature candidate is not automatically the low-phase-error candidate.

For AlN-only (320,600), the 120/180/240-mesh phase spreads are 0.2292537/0.2283612/0.2299269 rad. For AlN-only (460,740), they are 0.5587925/0.5563873/0.5551348 rad. For graded-W+AlN (180,880), they are 0.3536241/0.3536446/0.3536357 rad. Across the six selected placement/material combinations, all 120-to-240 phase-spread changes are below 0.73% and mean diagonal heater-response changes are below 0.60%. The nonmonotone phase convergence of the first position does not justify an asymptotic Richardson-order claim.

Finite interface resistance is not invariably detrimental to differential phase: the phase-selected graded-W+AlN (320,600) candidate changes from 0.2403207 rad at RK=10⁻⁸ to 0.2088279 at 10⁻⁷ and 0.1065697 at 10⁻⁶, while mean temperature rises. These are deterministic model cases and do not prescribe an unmeasured fabrication interface or establish reliability.

## Optical model-form sensitivity

`python optical_sensitivity.py` reuses the solved temperature/strain fields. It compares vertical-first and horizontal-first two-step effective-index approximations and scales the nominal photoelastic coefficients by 0.5, 1.0 or 1.5, without changing thermo-optic or geometric derivatives except as implied by each effective-index ordering. All four main material ablations and eighteen placement entries are included: 132 reported evaluations (120 unique physical/optical combinations after duplicate geometry checks). Every evaluation saves phase, individual thermal/photoelastic/shape terms and the complete physical heater matrix, so the controller can be reevaluated consistently.

AlN-only (320,600) reduces passive phase spread relative to Cu by 32.8756–47.0381% across all six deterministic optical assumptions. The placement (320,600) has the smallest passive phase spread among the nine sampled positions in every assumption set. The preferred material assignment changes: vertical-first EIM with half photoelastic coefficients gives 0.2483797 rad for graded-W+AlN and 0.2498008 rad for AlN-only; the remaining five assumptions favor AlN-only. Thus the placement result persists across these tests, while a universal advantage of graded W is not supported. These sensitivity ranges are neither confidence intervals nor manufacturing-yield estimates.

## Reproduction sequence and figures

Run from this directory:

```bash
python optical_mode.py
python physical.py
python placement_analysis.py
python placement_analysis.py --validation
python optical_sensitivity.py
python plot_physical.py
python plot_placement.py
```

The independent controller script uses `control_inputs.npz`, `placement_inputs.npz`, `placement_validation_inputs.npz` and, optionally, `optical_sensitivity_inputs.npz`. All four archives express H in rad/mW for the assumed 1 mm depth. Thermal/strain arrays remain available in the corresponding `physical_inputs` archives for future recalibration.

Figures are `physical_geometry`, `physical_fields`, `physical_passive_tradeoff`, `physical_contact_sweep`, `physical_heater_matrix`, `physical_mesh_convergence`, and `physical_placement`, each supplied as vector PDF and high-resolution PNG. The stress contour scale is explicitly clipped at the 99th percentile for readability and must not be interpreted as a maximum-stress bound.
