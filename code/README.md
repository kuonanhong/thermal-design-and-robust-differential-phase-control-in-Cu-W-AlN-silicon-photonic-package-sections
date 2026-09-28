# 完整可重現模擬 / Complete reproducible simulation

本套程式對應新 OLT 論文的全部數據，不使用舊海報結果填補新模型。所有數據都是已執行的數值模型結果，並非製造樣品或實驗量測。

This source reproduces the OLT manuscript's new numerical dataset. It does not substitute historical poster values for new model outputs. There are no fabricated-device measurements.

## 執行 / Run

建議 Python 3.12；一般 CPU 即可，無須 GPU、COMSOL 或其他商用求解器。下列命令於解壓縮套件根目錄執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
python code/reproduce_all.py
```

For exact package versions from the executed environment, install `code/requirements-reproduced.txt`. NumPy/SciPy/Shapely wheel availability depends on OS and Python version. The compatible range file allows another supported environment. The supplied execution used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8. Runtime is hardware-dependent; the recorded run is not a performance claim for the user's Mac.

只檢查隨附結果、不重跑所有物理場 / Check supplied results without repeating all field solves:

```bash
python code/reproduce_all.py --verify-only
```

## 程式與責任 / Modules

| Module | Role / 功能 |
|---|---|
| `core.py` | Geometry, conservative thermal faces, sparse Q4 thermoelasticity, heater normalization / 幾何、保守傳熱、Q4 熱彈性、熱源 |
| `optical_mode.py` | TE/TM slab equations, quasi-TE EIM, derivative and algebra checks / 模態、導數與解析驗證 |
| `physical.py` | 48 material/contact/conductivity/grid cases; background + nine heaters / 主物理掃描 |
| `placement_analysis.py` | 18 fixed-volume placements and 24 selected checks / 固定體積位置與細化 |
| `optical_sensitivity.py` | 132 model-form/photoelastic evaluations / 光學模型形式與參數敏感度 |
| `control_analysis.py` | Pairwise nominal/robust LP, DAC, uncertainty certificates / 差分相位控制與量化上界 |
| `placement_control.py` | Same controller across placement cases / 位置配置控制比較 |
| `plot_physical.py`, `plot_placement.py` | Scientific PNG and vector PDF figures / 科學圖 |
| `research/independent_checks.py` | Independent reconstruction without importing solver implementation / 獨立重算 |
| `workload_robust.py` | Separate EIC/PIC source bases, exact source-box certificates, fixed/vertex controls and continuous lower bounds / 熱源分解與負載區間 |
| `workload_independent_check.py` | Independently reconstructs workload projections, LPs, bounds and crossover / 獨立負載檢查 |
| `vector_mode.py` | Full-vector Maxwell mode, global grid/padding/derivative checks and consistent phase/H reprojection / 全向量光學 |
| `vector_target_study.py` | 0.25 rad target LP under three vector meshes / 全向量目標功率 |
| `vector_independent_check.py` | Independent modal and controller reconstruction / 獨立向量結果檢查 |
| `reproduce_all.py` | Ordered execution and reproducibility log / 一次執行全部流程 |

Changing a parameter defines a new model scenario. Regenerate both background phases and the heater response matrix; do not reuse an old H after modifying material/optical assumptions. Details are in `PHYSICAL_MODEL.md`, `CONTROL_METHOD.md`, and `research/MODAL_METHOD.md`.

## 數據結構 / Data

Each physical case uses ten load solutions: one background source and nine unit heaters. There are 90 reported configurations (88 unique, with two intentional duplicate placement checks), totalling 900 solved loads. The main control grid has 6,480 rows, including explicitly infeasible targets.

- `results/physical_inputs.npz`: case × channel background temperature/strain/stress; case × heater × channel unit-heater fields.
- `results/control_inputs.npz`: `phi_rad[case,channel]`, `H_rad_per_mW[case,channel,heater]`; case metadata and separated optical contributions.
- `results/placement*_inputs.npz`: corresponding fixed-volume placement and mesh/contact/conductivity checks.
- `results/optical_sensitivity_inputs.npz`: consistently reprojected phase/H under six optical assumptions.
- `results/control_results.csv`: scenario, feasibility, nominal rounded phase, pre-bound, exact posterior interval bound, physical power, phase-only MZI penalty.
- `results/control_commands.csv`: continuous and rounded heater commands.
- `results/control_sensitivity.csv`: entrywise versus column-correlated uncertainty and additional target tests.
- `results/placement_control*.csv` and `results/placement_control_validation/`: placement commands/certificates.
- `results/reproduction_manifest.json`, `research/independent_checks.json`: executed verification records.

NPZ example:

```python
from pathlib import Path
import numpy as np
p = Path('results/control_inputs.npz')
with np.load(p, allow_pickle=False) as a:
    i = int(np.where(a['case_id'] == 'Cu_R0e+00_k170_n120')[0][0])
    phi = a['phi_rad'][i]          # rad; nine parallel 1 mm segments
    H = a['H_rad_per_mW'][i]      # rad/mW; observation × heater
    power = np.zeros(9)           # nonnegative physical heater powers, mW
    controlled = phi + H @ power
    print('Passive pairwise phase range:', np.ptp(controlled))
```

The model is a steady, invariant-z package section with plane strain and a reduced EIM optical submodel. The 1 mm length defines the line-power-to-mW conversion; it is not a resolved 3D optical circuit. The uncertainty boxes are declared calibration scenarios, not production yield. `feasible=False` and missing commands are not zero-power solutions. An LP certificate failure is not a global proof that all discrete or phase-wrapped controllers fail.

## 本次新增資料 / Revision datasets

`results/workload_robust_physical_fields.npz` retains five geometries, each with independently normalized EIC and PIC loads, nine heater loads and a nominal direct check. `workload_robust_inputs.npz` contains G, H, source corners and the actual command arrays. The main source box is EIC 200–300 mW and PIC 20–30 mW. The EIM optical map is used for this study; those bounds do not include vector-model uncertainty.

`results/vector/` contains globally refined Maxwell benchmarks, mode fields, tensor/thermal/shape derivatives, 22 nominal-load reprojected layouts, commands and 0.25-rad target solves. The exact upstream MIT source and provenance are included under `code/vendor/wgmodes/`; no upstream network fetch is needed after installing the listed public dependencies. `requirements-vector.txt` gives the executed environment for that component.

The expanded end-to-end workflow was rerun on 27 September 2026: all 17 stages passed in 393.65 seconds, including the base study's 900 thermomechanical loads, 60 additional source-resolved loads, and 86 Maxwell eigenproblems. This is not a runtime promise for another machine. All 52 supplied NPZ/CSV files and numerical fields in seven modal-coefficient JSON files matched the regenerated numerical values exactly. See `results/reproduction_manifest.json` and `docs/code_audit.json` for the actual execution and comparison records. `--verify-only` reconstructs the supplied old and new results without repeating the physical field solves or Maxwell eigenproblems.

The revised workload checker also exports primal–dual residual checks and an exact-rational dual lower bound for each assembled numerical relaxation LP. These support fixed-command exclusions within the declared numerical model; they do not certify experimental accuracy or continuum discretization error.
