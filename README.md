# Non-Linear Optimization: Gauss-Newton vs. Newton Method

This repository contains a Python implementation comparing the **Gauss-Newton** and **Newton's** methods for non-linear least squares optimization. 

The algorithm is applied to fit experimental data to the **Michaelis-Menten** equation, commonly used in biochemistry to model enzyme kinetics.

## The Model

The goal is to find the optimal parameters $\alpha$ (maximum velocity, $v_{max}$) and $\beta$ (Michaelis constant, $K_M$) for the following non-linear function:

$$f(x) = \frac{\alpha x}{\beta + x}$$

We minimize the Sum of Squared Residuals (SSR) cost function:

$$S(\alpha, \beta) = \sum_{i=1}^{n} \left( y_i - \frac{\alpha x_i}{\beta + x_i} \right)^2$$

## Features
* **Custom Implementations:** Both optimization algorithms are written from scratch using `NumPy`, demonstrating a deep understanding of Jacobians and Hessians.
* **Gauss-Newton Method:** Approximates the Hessian matrix using the Jacobian to iteratively update parameters.
* **Exact Newton Method:** Calculates the exact Hessian matrix (including the second-order derivative correction matrix) for comparison.
* **Visualization:** Uses `Matplotlib` to plot the convergence rates of the cost function $S(\alpha, \beta)$ for both methods on a logarithmic scale.

## Requirements
* Python 3.x
* NumPy
* Matplotlib

## How to Run
Simply execute the script in your terminal to view the convergence plots and the final parameter estimations:
`python tp2gaussnewton.py`
