# FENICE — Future Energy traNsItioN multI-seCtor modEl

## Presentation
**FENICE** (Future Energy traNsItioN multI-seCtor modEl) is a research-grade Python model to represent and optimize multi-sector energy systems.  
It is designed to model system detail and transition across years using a workflow that couples:

- **Multi-sector network representation**: electricity, heat, hydrogen, storage and flexible demand modeled as components and commodities in a network.  
- **Temporal aggregation**: a clustering workflow that produces representative time slices to reduce computational cost while preserving key temporal patterns.  
- **Multi-period studies**: the same model runs for multiple scenario years and configurations to analyse transition pathways.  
- **Reproducible inputs**: scenario and technology data are provided via structured Excel workbooks so analyses are easy to configure and share.  
- **Reproducible codebase**: implemented on top of a patched `oemof-solph` fork to include project-specific fixes required by the model.

> Typical use cases: long-term decarbonization scenarios, technology sensitivity studies, hydrogen integration analysis, storage and flexibility assessments.

---

## Cite FENICE
Please cite FENICE when used in research. Prefer citing an archived release (Zenodo DOI) if available. Example citation formats:

