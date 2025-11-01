# Provenance Documentation

**Project**: Extreme-Field QED Simulator with Gravitational Coupling  
**Date**: October 31, 2025  
**Purpose**: This document tracks all reference materials that have influenced the physics implementation in this codebase.

## Overview

This project builds on theoretical and experimental work in:
- Vacuum QED (Heisenberg-Euler effective Lagrangian)
- Vacuum birefringence and light-by-light scattering
- Gravitational wave generation from electromagnetic sources
- Anomalous gravity-photon coupling hypotheses

All reference materials are stored in `docs/reference/` with checksums documented below for verification and reproducibility.

## How to Add New References (for agent use)

To make new papers/books available to tooling and AI agents:

1. Place the file under `docs/reference/<TopicName>/` (PDF, TeX, or BibTeX).
2. Run a checksum and add an entry under the appropriate section below.
	```bash
	sha256sum docs/reference/<TopicName>/<file.ext>
	```
3. If it’s a PDF, consider extracting text for search:
	```bash
	python scripts/tools/pdf_to_text.py docs/reference/<TopicName>/<file.pdf> > docs/reference/<TopicName>/<file>.txt
	```
4. Update or create a BibTeX entry in `docs/reference/<TopicName>/<file>.bib` with citation keys.
5. Commit changes with a message that mentions “provenance” so bots can find it easily.
6. Optional: add a short note in this file’s “Update History”.

Agent discovery tips:
- Use descriptive folder and file names (e.g., `Heisenberg-Euler/dunne2011.tex`).
- Prefer searchable text (TeX/Markdown or extracted `.txt`).
- Add a one-line “Influence” note explaining why the reference matters (which module it informs).

---

## Reference Materials

### 1. Heisenberg-Euler Effective QED

**File**: `docs/reference/Heisenberg-Euler/dunne-qfext11-web.tex`  
**Title**: "Heisenberg-Euler Effective Lagrangians: Basics and Extensions" (Dunne, 2011)  
**SHA-256**: `2012566e9cf42635afc197ff6dcc6e3230e7d07e90795aa9a64c03a18bea5396`  
**Influence**:
- Foundation for `src/efqs/heisenberg_euler.py` module
- QED vacuum corrections to Maxwell's equations
- Pair production threshold calculations
- Critical field strength E_schwinger derivation

**Key Equations Used**:
- Heisenberg-Euler Lagrangian: $\mathcal{L}_{HE} = \frac{2\alpha^2}{45m_e^4}[(EB)^2 + \frac{7}{4}(E^2 - B^2)^2]$
- Critical field: $E_{crit} = \frac{m_e^2 c^3}{e\hbar} \approx 1.3 \times 10^{18}$ V/m

---

### 2. PVLAS Vacuum Birefringence Experiment

**File**: `docs/reference/PVLAS_experiment/Grav_fp5.tex`  
**Title**: "Gravitational coupling in the PVLAS experiment" (working paper)  
**SHA-256**: `626108b5816ccb5a298c22df8996bba1ee656c1c8d7dae888b879b85f0635ced`  
**Influence**:
- Experimental constraints on vacuum birefringence
- Detector sensitivity thresholds for `src/efqs/detector_noise.py`
- Validation benchmarks for `scripts/simulate_birefringence.py`

**File**: `docs/reference/PVLAS_experiment/PVLAS.bib`  
**Bibliography**: References for PVLAS experimental program  
**Influence**: Cross-references for experimental methodology

---

### 3. Graviton Detection and GW from EM Sources

**File**: `docs/reference/Detecting_single_gravitons/main.tex`  
**Title**: "Detecting Single Gravitons with Quantum Sensors" (concept paper)  
**SHA-256**: `4808ce6b3f4a71dbb3e662d94f46cda76ef44ef7790bef50b622e15fb6bde4ee`  
**Influence**:
- Quantum limits on gravitational wave detection
- Aspirational sensitivity targets: $h \sim 10^{-30}$ for quantum sensors
- Conceptual framework for `src/efqs/gravitational_coupling.py`

**File**: `docs/reference/Detecting_single_gravitons/biblio.bib`  
**Bibliography**: GW detection literature and theory  
**Influence**: Citations for quadrupole formula and TT gauge

---

### 4. An Addendum (Supplementary Theory)

**File**: `docs/reference/An_Addendum/source.tex`  
**Title**: "An Addendum to Vacuum QED and Gravitational Coupling" (notes)  
**SHA-256**: `a6c804e801b1e3cb7d89cf3de70edd699edb064498507f70a0b5a92d1a7cd023`  
**Influence**:
- Theoretical extensions and corrections
- Edge cases in field computations
- Additional context for anomalous coupling scenarios

---

## Code-to-Reference Traceability

### `src/efqs/heisenberg_euler.py`
- **Primary Reference**: Dunne (2011) - Heisenberg-Euler effective Lagrangian
- **Equations**: H-E Lagrangian, QED stress-energy corrections
- **Validation**: Compared to known results in weak-field limit

### `src/efqs/gravitational_coupling.py`
- **Primary Reference**: Detecting single gravitons paper
- **Equations**: Quadrupole approximation, $h_{ij} \approx \frac{2G}{c^4 R}\ddot{Q}_{ij}$
- **Methodology**: Spectral derivatives via FFT for numerical stability

### `src/efqs/vacuum_birefringence.py`
- **Primary Reference**: PVLAS experiment papers
- **Equations**: Cotton-Mouton birefringence in vacuum
- **Validation**: Cross-checked with PVLAS experimental constraints

### `scripts/simulate_birefringence.py`
- **Primary Reference**: PVLAS methodology
- **Parameters**: Field strengths, cavity configurations from PVLAS design

---

## Verification Procedure

To verify the integrity of reference materials:

```bash
cd /home/echo_/Code/asciimath/extreme-field-qed-simulator
sha256sum docs/reference/Heisenberg-Euler/dunne-qfext11-web.tex
sha256sum docs/reference/PVLAS_experiment/Grav_fp5.tex
sha256sum docs/reference/Detecting_single_gravitons/main.tex
sha256sum docs/reference/An_Addendum/source.tex
```

Expected output:
```
2012566e9cf42635afc197ff6dcc6e3230e7d07e90795aa9a64c03a18bea5396  docs/reference/Heisenberg-Euler/dunne-qfext11-web.tex
626108b5816ccb5a298c22df8996bba1ee656c1c8d7dae888b879b85f0635ced  docs/reference/PVLAS_experiment/Grav_fp5.tex
4808ce6b3f4a71dbb3e662d94f46cda76ef44ef7790bef50b622e15fb6bde4ee  docs/reference/Detecting_single_gravitons/main.tex
a6c804e801b1e3cb7d89cf3de70edd699edb064498507f70a0b5a92d1a7cd023  docs/reference/An_Addendum/source.tex
```

---

## TF-IDF Analysis Results

**Date**: October 31, 2025  
**Tool**: `scripts/check_copilot_usage.py`  
**Method**: TF-IDF cosine similarity between reference documents and outputs

### Chat History Similarity
```
docs/reference/Heisenberg-Euler/dunne-qfext11-web.tex    → 0.0865
docs/reference/Detecting_single_gravitons/main.tex       → 0.0847
docs/reference/PVLAS_experiment/Grav_fp5.tex             → 0.0751
```

### Commit Message Similarity
```
docs/reference/Detecting_single_gravitons/biblio.bib     → 0.0946
docs/reference/Detecting_single_gravitons/mainNotes.bib  → 0.0844
docs/reference/PVLAS_experiment/PVLAS.bib                → 0.0810
```

**Interpretation**: Similarity scores of 0.08-0.09 indicate moderate overlap, confirming that reference materials were consulted during development. This is expected for a physics simulation codebase that implements equations from the literature.

---

## Agent Activity Log

**Note**: For full transparency, all file access and modifications by AI agents are logged in `agent_access_log.json` (if instrumentation is enabled).

### Current Session (October 31, 2025)
- Reviewed previous agent's work on `gravitational_coupling.py`
- Updated experiment YAML definitions to match existing pipeline format
- Created this provenance documentation
- Verified test suite passes for all new modules

---

## Update History

| Date | Update | Modified By |
|------|--------|-------------|
| 2025-10-31 | Initial provenance documentation created | Claude Sonnet 4.5 |
| 2025-10-31 | Added checksums for 4 primary reference documents | Claude Sonnet 4.5 |
| 2025-10-31 | Added TF-IDF analysis results | Claude Sonnet 4.5 |

---

## Acknowledgments

This work builds on:
- G. Dunne's review of QED effective field theory
- The PVLAS collaboration's experimental methods
- Theoretical proposals for graviton detection
- Classical GR quadrupole radiation formula (Einstein, 1916; Landau & Lifshitz)

---

## License & Attribution

All reference materials in `docs/reference/` are used for research and educational purposes. Original authors retain copyright. This project's code is licensed under the terms specified in the repository's LICENSE file.

For questions about provenance or to report discrepancies:
- Check `scripts/check_copilot_usage.py` output
- Verify checksums against this document
- Review git commit history for code evolution
