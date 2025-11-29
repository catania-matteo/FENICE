# FENICE — Future Energy traNsItioN multI-seCtor modEl

## Overview
**FENICE** (Future Energy traNsItioN multI-seCtor modEl) is a bottom-up energy system model, able to investigate long-term scenarios with a multi-annual span, solving the progressive configuration and operation of the system for the objective of minimum cost, under CO2 emission constraints as well as technological constraints. The tool is based on linear programming (LP), and it is implemented on the existing framework oemof-solph . The programming language is Python, the optimization problem is described via pyomo, and it can use both commercial and open-source solvers. Model results provide the optimal design and operation of the multi-sector integrated energy system for the multiple periods identified, which are devised to track the transition from a given system status to an end point featuring particularly relevant constraints.

## Features
- **Sector-coupling**: all the most energy demanding sector are represented (civil, industry and transport) and emissions are accounted for the others (agricolture and waste disposal)
- **Multi-vector**: all the main energy carrier of current and future energy system represented such as electricity, hydrogen, biomass, liquid fuels, natural gas, biomethane.  
- **Temporal aggregation**: a clustering workflow that produces representative time slices to reduce computational cost while preserving key temporal patterns.  
- **Multi-period studies**: the model can be optimized on a wide horizon capable to analyse transition pathways.
- **Multi-node analysis**: the model is flexible in analysing mono or multi-node energy systems with the possibility to define several energy infrastrutures

---

## Cite FENICE
Please cite FENICE when used an refer it to this publication [https://dx.doi.org/10.2139/ssrn.5302804](https://doi.org/10.1016/j.enconman.2025.120663).

