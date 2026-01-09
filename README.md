# th4din — Thermal–Hydraulic Network Modeling Framework for Direct and Inverse Problems

**th4din** is an open-source Python module for steady-state thermal–hydraulic
network modeling with unified support for **direct** and **inverse** problem
formulations.

The module is a python implementation of the methodology presented in the accompanying paper:

**Mirco Ganz, Frank Tillenkamp, Christian Ghiaus**  
*Methodology for solving direct and inverse steady-State thermal–Hydraulic network problems*

and enables simulation, analysis, and set-point-driven optimization of coupled
thermal–hydraulic systems using a hybrid **causal–acausal** modeling approach
combined with **graph-based tearing for system reduction**.

A fully documented **Jupyter Notebook** reproducing the test case presented in
the paper can be executed directly via Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MircoGanz/th4din/main?filepath=main.ipynb)

**Author:** Mirco Ganz

---

## Key Features

- Component-based modeling of steady-state thermal–hydraulic systems  
- Hybrid **causal / acausal** formulation combining input–output component models
  with implicit system-level balance equations  
- Automatic system assembly using junction-level mass, energy and pressure balance equations
- Graph-based **tearing algorithm** to reduce the dimensionality of the nonlinear
  equation system  
- Unified treatment of:
  - **Direct problems**  
    (prescribed inputs, solution of steady-state system variables and outputs)
  - **Inverse problems**  
    (prescribed target outputs, simultaneous solution of system variables and
    unknown inputs using a SAND formulation)
  - **Optimization problems**  
    (design and operational optimization with variable parameters)
- Explicit handling of:
  - boundary conditions,
  - open and closed fluid loops,
  - additional equality and inequality constraint equations
- Steady-state simulation using nonlinear root-finding methods  
- Constrained optimization using **Sequential Least Squares Quadratic Programming
  (SLSQP)**  
- Fully implemented in **Python**  
- Reproducible and transparent workflows based on **Jupyter notebooks**

