# Full-vector optical cross-check / 全向量光學交叉檢查

Run after the main thermal/mechanical study:

```bash
python -m pip install -r code/requirements-vector.txt
OPENBLAS_NUM_THREADS=1 python code/vector_mode.py
python code/vector_target_study.py
```

The wrapper imports the original MIT-licensed WGMODES source supplied in `code/vendor/wgmodes`; the upstream modesolver package does not need a separate installation. A git-pinned upstream installation command is documented in `requirements-vector.txt` if desired. `PROVENANCE.md`, `LICENSE.md` and source checksums accompany the complete original Python module.

This is a genuine two-dimensional full-vector local waveguide eigenmode calculation. It is not a three-dimensional package simulation. It keeps the same nominal material constants, extrusion and plane-strain assumptions. The original EIM dataset remains unchanged; the independent comparison lives only in `results/vector`.

## Numerical formulation

- Original Fallahkhair–Li–Murphy transverse-H finite-difference algorithm, DOI 10.1109/JLT.2008.923643. The solver returns all electric/magnetic components.
- Optical wavelength 1.55 µm; 0.45 × 0.22 µm silicon core; silicon n=3.476 and oxide n=1.444.
- Material boundaries are exactly aligned with cell boundaries. Width/height central differences stretch core-cell coordinates with fixed cell counts, so material masks do not jump.
- Full computational window uses 1.2 µm oxide padding on all faces and hard zero Hx/Hy outside the boundary. Padding 0.4/0.8/1.2/1.6 µm is checked independently.
- Core grid 10/5/2.5/1.25 nm. The whole cladding mesh refines with the core: at core spacing d in µm, initial outside spacing d grows by 1.10^(d/0.01) and is capped at 5d. Holding the grading ratio fixed under refinement is deliberately avoided, because it leaves a cladding discretization floor.
- Quasi-TE is identified by its Ex electric-intensity fraction >0.5; perturbed solutions track normalized electric-field overlap. The reported fraction is not a power or confinement fraction.
- Original eigensolver matrices are unmodified. The wrapper uses a deterministic ARPACK initial vector and computes ||Av−λv||/||Av|| for every returned eigenpair.
- Uniform-medium constant-H and exact analytical TE/TM slab tests are independent of the 2D rectangular benchmark. Rectangular mesh changes are reported as numerical changes, not rigorous continuum error bounds.

## Optical coefficients

The final map uses the 2.5 nm core grid. Its derivatives include silicon and cladding scalar refractive index, separate silicon nx/ny/nz, width and height. Principal-index derivatives are contracted with the same p11/p12 photoelastic matrix to obtain all three normal elastic-strain coefficients. An epsxy=epsyx perturbation checks the first-order shear-index response, which vanishes by symmetry to numerical precision. Sheared core boundary geometry and oxide photoelasticity are not modeled. Nominal photoelastic constants remain uncalibrated assumptions.

Relative finite-difference steps 1e-4 and 5e-5 and mesh refinement 5 → 2.5 nm are checked separately. The sum of the three principal-index derivatives is checked against the independent isotropic core-index derivative.

## Outputs and mapping

- `vector_coefficients.json`: selected full-vector optical coefficient map.
- `vector_benchmarks.csv`: exact uniform and TE/TM slab comparisons.
- `vector_mesh_convergence.csv`, `vector_padding_convergence.csv`, `vector_derivative_convergence.csv`: numerical verification.
- `vector_nominal_fields.npz`: core-aligned node axes in µm, three electric/magnetic components, selected index and branch tracking metadata. The three stored electric components are (Ex, Ey, jEz), so magnitude-squared intensity is computed directly; this convention is inherited from the original solver.
- `vector_reprojected_inputs.npz`: 22 cases × 9 channel background phases, 22 × 9 × 9 physical heater response matrices and actual quantized commands. Response axes are [case, sensed channel, heater]; units rad and rad/mW. Thermal source NPZ heater responses originally have [case, heater, channel] and are transposed consistently.
- `vector_reprojected_cases.csv`, `vector_control_commands.csv`: passive/control metrics and commands.
- `vector_summary.json`: numerical checks, selected comparisons, provenance and limitations.
- `vector_target025.csv`, `vector_target025_commands.csv`, `vector_target025_summary.json`: explicit zero-power and minimum-power 0.25-rad target checks on 10/5/2.5-nm derivative meshes.
- `vector_verification.png/.pdf`: four-panel figure for paper/slides. The normalized |E|² image uses a logarithmic color scale to resolve the dynamic range near high-index corners; it is not a Poynting-flux or confinement-fraction map.

Both background phase and the entire heater-response matrix are reprojected from `physical_inputs.npz` and `placement_physical_inputs.npz`; reusing an EIM heater matrix with vector background phases would be inconsistent and is not done. Four nominal ablations plus 18 placements are reported. Two nominal placements repeat two ablations, yielding 20 distinct physical configurations.

Controller assumptions are unchanged: one unwrapped phase branch, nonnegative heaters, 20 mW per heater, 100 mW physical total budget with pre-round reserve, 8-bit endpoint DAC, independent response-entry intervals ±2%, and channel sensing intervals ±0.002 rad. The 2% interval represents a separately assumed calibration-error set and does not cover EIM-to-vector model differences.

重現時請先完成原熱－機模型，再執行上述指令。全向量檢查會一致重算背景相位與所有加熱器響應，保留 EIM 原始結果供比較。此為局部二維波導驗證，不能宣稱為三維封裝或實驗驗證；材料光彈參數仍須校正。
