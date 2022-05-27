![](https://img.shields.io/badge/status-in_development-orange?style=flat-square)
![](https://img.shields.io/badge/licence-MIT-green?style=flat-square)
![](https://img.shields.io/badge/language-Python-blue?style=flat-square)
![](https://img.shields.io/badge/requirement-FEniCS-blue?style=flat-square)

# Minimal Rational Interpolation for Time-Harmonic Maxwell's Equations
Semester project for the computational science and engineering master's program at EPFL.

## Introduction
A wide class of problems in physics and engineering concerns itself with the study of the dependence of a model on one of its parameters. Of interest is usually a characteristic quantity that covaries with said parameter. Unless the system allows for an analytical solution, one may usually only find numerical solutions to the system for discrete values of the parameter by e.g. employing the finite element method (FEM). Minimal rational interpolation (MRI) offers a way to locally approximate the continuous dependence of a model on one of its parameters.

## Implementations
Built on `numpy` and `scipy`, this repository offers a complete MRI suite for time-harmonic problems in electromagnetism. The engine for obtaining solutions to the partial differential equations governing these problems is `fenics`. 

## Quick start

    git clone https://github.com/FMatti/Maxwell-MRI.git
    cd Maxwell-MRI
    touch quickstart.py

Open the file `quickstart.py` in your preferred python editor, and run the following lines:

    import fenics as fen
    import numpy as np
    
    # Import utilities
    from src.time_harmonic_maxwell_problem import TimeHarmonicMaxwellProblem
    from src.minimal_rational_interpolation import MinimalRationalInterpolation
    from src.vector_space import VectorSpaceL2

    # Define mesh
    mesh = fen.UnitSquareMesh(100, 100)
    V = fen.FunctionSpace(mesh, 'P', 1)

    # Define position of inlet to be at x=0
    class B_N(fen.SubDomain):
        def inside(self_, x, on_boundary):
            return on_boundary and fen.near(x[0], 0.0)

    # Define all other boundaries to be perfectly conducting
    class B_D(fen.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and not B_N().inside(x, 'on_boundary')

    # Set physical constants
    one = fen.Expression('1.0', degree=2)
    eps = fen.Expression('1.0', degree=2)
    j = fen.Expression('0.0', degree=2)
    
    # Set the boundary conditions for perfect conductor and the inlet
    u_D = fen.Expression('0.0', degree=2)
    g_N = fen.Expression('1.0', degree=2)

    # Set up the time-harmonic Maxwell problem
    THMP = TimeHarmonicMaxwellProblem(V, mu, eps, j, B_D(), u_D, B_N(), g_N)
    THMP.setup()

    # Create the corresponding vector space
    VS = VectorSpaceL2(THMP)

    # Perform the greedy MRI algorithm to find resonant frequencies in [3, 4]
    MRI = MinimalRationalInterpolation(VS)
    MRI.compute_surrogate(THMP, greedy=True, omegas=np.linspace(3, 4, 1000))
    print(MRI.get_interpolatory_eigenfrequencies(only_real=True))
       
which will indeed give the first resonant frequency (3.51230205) of the quadratic unit cavity with one edge acting as an inlet and trivial physical constants.

## Example applications
Central to this project are time-harmonic Maxwell problems (THMP), whose parameter is the (angular) frequency. These problems are governed by the time-harmonic Maxwell's equations. Choosing the quantity of interest to be a vector potential, these equations reduce to a single curl-curl equation.

Three examples are studied:

- The resonant modes of a two-dimensional resonant cavity
- The two-dimensional cavity with an imperfectly conducting boundary
- The scattering coefficients of a dual-mode waveguide filter (DMCWF)

## File structure
```
Maxwell-MRI
│   README.md
|   LICENSE
|
└───examples
│   └───circular_waveguide_filter
|   |   └───model
|   |   |   |   DMCWF.geo                       (Gmsh work file)
|   |   |   |   DMCWF.step                      (3D CAD model)
|   |   |   |   DMCWF.xml                       (final mesh)
|   |   |
|   |   |   circular_waveguide_filter.ipynb     (visualization and demonstration notebook)
|   |   |   circular_waveguide_filter.py        (engine for simulating the DMCWF)
|   |
│   └───imperfect_conductor
|   |   |   imperfect_conductor.ipynb           (visualization and demonstration notebook)
|   |   |   imperfect_conductor.py              (rectangular cavity with an imperfect boundary)
|   |
│   └───resonant_cavity
|   |   |   resonant_cavity.ipynb               (visualization and demonstration notebook)
|   |   |   rectangular_cavity.py               (rectangular cavities)
|   |   |   two_dimensional_cavity.py           (generic 2D cavities)
|
└───report
|   |   ...
|
└───src
|   |   helpers.py                              (helper script for visualization)
|   |   minimal_rational_interpolation.py       (handles the MRI of a THMP)
|   |   rational_function.py                    (rational polynomials in barycentric coordinates)
|   |   snapshot_matrix.py                      (save/load FEM solutions)
|   |   time_harmonic_maxwell_problem.py        (set up and solve THMP)
|   |   vector_space.py                         (provides a norm and inner product)
|
└───test
|   |   test_minimal_rational_interpolation.ipynb
|   |   test_rational_function.ipynb
|   |   test_vector_space.ipynb
```
