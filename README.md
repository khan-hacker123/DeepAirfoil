# FastFlow-AI: Deep Learning Surrogate Model for Aerodynamics

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Aerodynamics](https://img.shields.io/badge/Domain-Fluid%20Dynamics-lightgrey.svg)

## Overview
**FastFlow-AI** is a data-driven surrogate model that predicts aerodynamic performance using Deep Learning. By mapping airfoil geometry and flow conditions directly to aerodynamic coefficients, this project bypasses the computationally expensive Navier-Stokes equations and panel method solvers (like XFOIL or OpenFOAM), delivering predictions in milliseconds.

This repository serves as a foundational step toward true Physics-Informed Machine Learning (PiML) in aerospace engineering.

---

## The Science: From Basics to Deep Learning

This project bridges two complex domains: Aerodynamics and Artificial Intelligence. Here is how they connect in this codebase.

### 1. The Aerodynamics (The Inputs & Outputs)
To understand how an aircraft wing performs, we evaluate specific dimensionless numbers. 
* **The Inputs (Flow & Geometry):** We feed the model 64 distinct parameters.
  * **Geometry (62 features):** The physical shape of the airfoil, represented by 31 upper surface and 31 lower surface coordinate coefficients (UIUC format).
  * **Angle of Attack ($\alpha$):** The angle between the oncoming air and the chord line of the airfoil.
  * **Reynolds Number (Re):** A massive number that dictates the flow regime (laminar vs. turbulent) based on the fluid's velocity and viscosity.
* **The Outputs (Performance):**
  * **Lift Coefficient (C_l):** How much upward force the shape generates.
  * **Drag Coefficient (C_d):** How much air resistance the shape creates.

### 2. The Neural Network (The Engine)
Traditional Computational Fluid Dynamics (CFD) calculates $C_l$ and $C_d$ by solving millions of tiny equations across a meshed grid. FastFlow-AI uses a **Multi-Layer Perceptron (MLP)** built in PyTorch to "learn" the mathematical relationship between the geometry and the forces without doing the physical math.
* **Data Normalization:** Because a Reynolds number can be 1,000,000 and a geometric coefficient can be 0.001, we use **Z-score scaling**. This prevents the neural network gradients from exploding during training.
* **Architecture:** The network scales down from 64 inputs -> 128 hidden neurons -> 64 neurons -> 32 neurons -> 2 final outputs ($C_l$, $C_d$). It uses ReLU (Rectified Linear Unit) activation functions to capture the highly non-linear nature of fluid dynamics.

---

## Dataset
This model is trained on the **Airfoil Performance and Geometry Dataset**, which contains thousands of UIUC airfoils evaluated across various flow conditions.
* Ensure you download the data as `airfoil_data.csv` and place it in the root directory.
* *Note: The dataset itself is not included in this repository to save space.*

---

## Getting Started

### Prerequisites
You will need Python 3.8+ and the following libraries:
```bash
pip install torch pandas numpy
