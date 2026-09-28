# Workload-aware thermal design and robust differential-phase control in Cu/W–AlN silicon-photonic package sections

Python source, executed numerical data, scientific figures, and English/Traditional Chinese documents for a study of package thermal design and differential optical phase control.

Repository: <https://github.com/kuonanhong/thermal-design-and-robust-differential-phase-control-in-Cu-W-AlN-silicon-photonic-package-sections>

This archive is prepared for that repository. Preparing this archive does not publish or upload its contents. It contains simulation results, not measurements from fabricated photonic devices.

## Repository purpose

A passive thermal modification changes both the background phase disturbance and the heater response used to correct it. This study computes those responses consistently, then asks how a finite nonnegative heater budget, DAC resolution, calibration intervals, source composition, and optical model affect the design decision.

The package contains three related calculations:

1. A steady two-dimensional package model with conservative heat conduction, finite AlN interface resistance, plane-strain thermoelasticity, and a quasi-TE effective-index optical model (EIM).
2. Independent EIC/PIC heat-source bases and one fixed heater command optimized for a declared workload box. Continuous relaxations give lower bounds for all feasible quantized fixed commands under the same model.
3. A separate local full-vector Maxwell benchmark using the vendored WGMODES solver, with consistent reprojection of background phase and heater response at nominal load.

The EIM workload study and the nominal-load full-vector study are separate analyses. The supplied results do not establish a combined full-vector workload-box guarantee, transient feedback performance, fabrication yield, or experimental accuracy.

## Quick start

Use Python 3.12 on a CPU. No GPU, COMSOL, proprietary electromagnetic solver, API key, or external data service is needed for the supplied calculations. Internet access is needed to install public Python dependencies; the solver source and scientific inputs are then local.

From the extracted repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
python code/reproduce_all.py --verify-only
```

`--verify-only` checks the supplied exports and reconstructs optical quantities, optimization problems, interval certificates, and tabulations. It does not rerun the full package field sweep or the Maxwell eigensolves. It writes verification JSON files and logs beneath `results/` and `research/`.

To regenerate the complete scientific dataset and figures:

```bash
python code/reproduce_all.py
```

This command overwrites generated files in the working copy. Keep the distributed archive or use a separate checkout when comparing a new parameter set against the supplied results. A successful complete run writes `results/reproduction_manifest.json`; a verification-only run writes `results/reproduction_verify.json`. Read their `all_passed`, environment, step, and log fields. A historical manifest is evidence of its recorded run, not evidence that a new local run has completed.

For the versions recorded in the supplied execution environment:

```bash
python -m pip install -r code/requirements-reproduced.txt
```

The compatible-range file supports installations with different available wheels; the pinned file records a specific environment. Exact floating-point output and execution time can vary across operating systems and BLAS/SciPy builds. See [the repository guide](docs/repository_guide.md) for Intel Mac instructions and component runs.

## Documents

| Document | English | Traditional Chinese |
|---|---|---|
| Manuscript | [PDF](manuscript/EN/OLT_Manuscript_EN.pdf) · [LaTeX](manuscript/EN/OLT_Manuscript_EN.tex) | [PDF](manuscript/ZH/OLT_Manuscript_ZH.pdf) · [LaTeX](manuscript/ZH/OLT_Manuscript_ZH.tex) |
| Technical report | [PDF](reports/EN/OLT_Technical_Report_EN.pdf) · [LaTeX](reports/EN/OLT_Technical_Report_EN.tex) | [PDF](reports/ZH/OLT_Technical_Report_ZH.pdf) · [LaTeX](reports/ZH/OLT_Technical_Report_ZH.tex) |

The technical reports explain the program-to-output mapping, the manuscript figures and tables, scientific scope, and the revision audit. The English manuscript is the submission manuscript; the Chinese document is a corresponding research translation. Repository citation metadata is supplied in [CITATION.cff](CITATION.cff). It does not assert a journal acceptance, article DOI, or unverified author affiliation.

## Programs and principal outputs

All paths below are relative to the repository root. Numerical files are retained so that a figure can be traced to its calculation rather than to an image alone.

| Program | Responsibility | Principal outputs |
|---|---|---|
| `code/core.py` | Geometry, normalized sources, conservative thermal operator, Q4 elasticity, field sampling | Imported operators; no standalone manuscript dataset |
| `code/optical_mode.py` | Slab dispersion, EIM coefficients, derivative and slab checks | `code/modal_coefficients*.json`, `results/modal/` |
| `code/physical.py` | Material, contact, conductivity, and package-mesh sweeps | `results/physical_*`, `results/control_inputs.npz`, field NPZ files |
| `code/placement_analysis.py` | Fixed-volume guard placements and selected validation cases | `results/placement*_inputs.npz`, placement summaries and verification records |
| `code/optical_sensitivity.py` | EIM-order and photoelastic scenarios using consistent phase/actuator reprojection | `results/optical_sensitivity*` |
| `code/control_analysis.py` | Nominal/robust LPs, DAC rounding, uncertainty and target scans | `results/control_*.csv`, control summaries/checks, `control_tradeoff.pdf` |
| `code/placement_control.py` | Control for the placement and validation datasets | `results/placement_control*`, `placement_optical_control.pdf` |
| `code/plot_physical.py` | Plot physical geometry, fields, passive response, interfaces, actuator matrices, and mesh comparisons | `figures/physical_*.pdf` and PNG equivalents |
| `code/plot_placement.py` | Plot equal-volume placement comparison | `figures/physical_placement.pdf` and PNG |
| `code/workload_robust.py` | EIC/PIC source bases, fixed workload-box commands, support-function certificates, relaxed lower bounds | `results/workload_robust_*`, `workload_source_composition.pdf` |
| `code/workload_independent_check.py` | Independently reconstruct workload projections, ordered-pair LPs, attained uncertainty corners, and exported summaries | `results/workload_independent_verification.json` |
| `code/vector_mode.py` | Full-vector mode, mesh/padding/derivative checks, phase and heater reprojection | `results/vector/`, including `vector_verification.pdf` |
| `code/vector_target_study.py` | Target-feasible quantized commands for selected vector meshes | `results/vector/vector_target025.csv` and command/summary exports |
| `code/vector_independent_check.py` | Independently reconstruct exported vector optical/control quantities | `results/vector_independent_verification.json` |
| `research/independent_checks.py` | Independent reconstruction of base physical, optical, and control exports | `research/independent_checks.json` |
| `code/reproduce_all.py` | Run the scientific stages in dependency order and retain logs | `results/reproduction_logs/`, reproduction manifests |

The independent checks use separately assembled calculations on exported data. In particular, the vector checker does not independently rerun the Maxwell solver, and exported-data checks do not establish the physical fidelity of the package model.

## Reading the data correctly

- `phi_rad[case, channel]` is a background phase vector. `H_rad_per_mW[case, channel, heater]` maps nonnegative heater power into phase. For a declared command `p`, the nominal phase is `phi + H @ p`.
- The invariant optical length is 1 mm: a 1 W/m line source corresponds numerically to 1 mW over that length. The package calculation does not resolve a three-dimensional routed optical circuit.
- The principal workload box is EIC 200–300 mW and PIC 20–30 mW, with fixed source shapes and constant material properties. It is a deterministic sensitivity envelope, not a measured workload distribution.
- `C_pre` is the sufficient design bound including DAC rounding margins. `C_post` is the exact independent interval-box certificate for the actual rounded command. They answer different questions and should not be interchanged.
- A feasible target command is not a proof of globally minimum integer-DAC power. Failure of a sufficient target LP is not itself proof that every possible quantized command fails. A separately solved continuous relaxation can provide the relevant fixed-command lower bound.
- Separate vertex-specific commands are steady operating-point comparisons. They are not an implemented feedback policy or an interior-load interpolation guarantee.
- A phase-only efficiency bound assumes the specified equal-amplitude, lossless optical combiner. It excludes coupling loss, propagation loss, and amplitude imbalance.

Example inspection:

```python
from pathlib import Path
import numpy as np

with np.load(Path("results/control_inputs.npz"), allow_pickle=False) as data:
    case = int(np.flatnonzero(data["case_id"] == "Cu_R0e+00_k170_n120")[0])
    phi = data["phi_rad"][case]
    H = data["H_rad_per_mW"][case]
    power_mW = np.zeros(H.shape[1])
    print("Passive phase range (rad):", np.ptp(phi + H @ power_mW))
```

## Reuse, provenance, and citation

Original project software, its software documentation, and generated numerical datasets are provided under the MIT terms in [LICENSE](LICENSE); [license.txt](license.txt) is an identical convenience copy. Manuscript and technical-report prose are outside that software license unless their individual files state otherwise. Cite the research and identify the exact repository commit or archived release used in any scientific reuse.

The vendored WGMODES source has its own retained MIT notice, copyright, pinned commit, and source hashes. Bundled Noto fonts retain their SIL Open Font License. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [RIGHTS.md](RIGHTS.md). These notices must remain with redistributed third-party files.

`SHA256SUMS.json` records the distributed file bytes. After regenerating a scientific dataset, hashes may change; successful numerical checks and a record of the new environment are then the appropriate comparison. `docs/final_validation.json` records the checks actually performed for this revision package.

## 繁體中文摘要

本套件提供 Cu/W–AlN 矽光子封裝之穩態數值研究。核心問題是：被動降溫同時改變未補償相位與加熱器矩陣，因此不能僅以平均溫度判斷光學控制效益。

完整流程採二維封裝熱傳與平面應變力學；主掃描與工作負載盒採 EIM，另以局部完整向量 Maxwell 模態檢查名目負載。兩項研究沒有合併成「完整向量工作負載盒」的保證。所有元件性能數據皆為模型結果，沒有實際樣品量測。

Intel Mac 可在 Python 3.12 虛擬環境執行，無須使用 AMD GPU。先以 `python code/reproduce_all.py --verify-only` 檢查隨附資料，再以 `python code/reproduce_all.py` 重算全部。完整重算會覆寫工作目錄內的生成結果；建議保留原始壓縮檔供比對。程式、輸出、圖表與數學限制詳見中英文技術報告及 [repository_guide.md](docs/repository_guide.md)。
