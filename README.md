# th4din — Thermal-Hydraulic Network Modeling Framework for Direct and Inverse Problems

**th4din** is an open-source Python framework for steady-state thermal–hydraulic
network modeling with unified support for **direct** and **inverse** problem
formulations.

The framework implements the methodology presented in the accompanying paper
and enables the simulation, analysis, and set-point-driven optimization of
coupled thermal–hydraulic systems using a hybrid **causal–acausal** modeling
approach combined with a **tearing-based system reduction**.

---

## Key Features

- Component-based modeling of steady-state thermal–hydraulic systems  
- Hybrid **causal / acausal** formulation  
- Automatic system assembly using junction balance equations  
- Graph-based tearing algorithm to reduce nonlinear system size  
- Unified treatment of:
  - **Direct problems** (prescribed inputs, solve for system state)
  - **Inverse problems** (simultaneous analysis and design, SAND)
- Explicit handling of:
  - boundary conditions,
  - closed and open fluid loops,
  - additional constraint equations
-  Simulation using root-finding solver
- Constrained optimization using Sequential Least Squares Quadratic Programming (SLSQP)
- Fully implemented in **Python**
- Reproducible workflows using **Jupyter notebooks**

---

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MircoGanz/th4din/main?filepath=main.ipynb)


