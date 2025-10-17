# Numerical-Integration-and-Drop-Dynamics-Model
This project extends the analysis performed in Dynamic Contact Angle Analysis of Droplet Impact on Solid Surfaces, focusing on the quantitative characterization and numerical modeling of droplet behavior.

It is divided into two main parts:

Geometric Analysis via Numerical Integration

Calculation of droplet volume and surface area from experimental contours.

Comparison between Spline interpolation and Polynomial fitting.

Application of Trapezoidal and Simpson’s integration rules.

Error estimation and convergence validation.

Dynamic Modeling of the Droplet Motion

Development of a damped oscillator model describing the droplet’s vertical oscillation after impact.

Resolution using Taylor (3rd order), Runge–Kutta 4–5, and Adams–Bashforth (4-step) methods.

Optimization of the stiffness (k) and damping (c) parameters via the Nelder–Mead algorithm to fit experimental data.

This work demonstrates how the selection of numerical integration and ODE-solving methods influences both precision and computational efficiency when modeling complex physical systems.

Technologies & Tools

Python (NumPy, SciPy, Matplotlib, OpenCV, Pandas)

Curve fitting and numerical integration

Differential equation solvers and optimization routines
