# FENICE — Future Energy traNsItioN multI-seCtor modEl

## Overview
**FENICE** (Future Energy traNsItioN multI-seCtor modEl) is a bottom-up energy system model, able to investigate long-term scenarios with a multi-annual span, solving the progressive configuration and operation of the system for the objective of minimum cost, under CO2 emission constraints as well as technological constraints. The tool is based on linear programming (LP), and it is implemented on the existing framework oemof-solph . The programming language is Python, the optimization problem is described via pyomo, and it can use both commercial and open-source solvers. Model results provide the optimal design and operation of the multi-sector integrated energy system for the multiple periods identified, which are devised to track the transition from a given system status to an end point featuring particularly relevant constraints.

## Features
- **Sector-coupling**: all the most energy demanding sector are represented (civil, industry and transport) and emissions are accounted for the others (agricolture and waste disposal)
- **Multi-vector**: electricity, hydrogen, biomass, liquid fuels, natural gas, biomethane.  
- **Temporal aggregation**: a clustering workflow that produces representative time slices to reduce computational cost while preserving key temporal patterns.  
- **Multi-period studies**: the model can be optimized on a wide horizon capable to analyse transition pathways.
- **Multi-node analysis**:
- **Reproducible inputs**: scenario and technology data are provided via structured Excel workbooks so analyses are easy to configure and share.  
- **Reproducible codebase**: implemented on top of a patched `oemof-solph` fork to include project-specific fixes required by the model.

---

## Cite FENICE
Please cite FENICE when used in research. Prefer citing an archived release (Zenodo DOI) if available. Example citation formats:

