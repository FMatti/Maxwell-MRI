![](https://img.shields.io/badge/status-finished-green?style=flat-square)
![](https://img.shields.io/badge/licence-MIT-green?style=flat-square)
![](https://img.shields.io/badge/language-Python-blue?style=flat-square)
![](https://img.shields.io/badge/requirement-FEniCS-blue?style=flat-square)

# Minimal Rational Interpolation for Time-Harmonic Maxwell's Equations
First semester project for the computational science and engineering master's program at EPFL.

## Introduction
A wide class of problems in physics and engineering concerns itself with studying the dependence of a model on one of its parameters. Of interest is usually a characteristic quantity depending on this parameter. Unless the system allows for an analytical solution, one may usually only find numerical solutions to the system for discrete values of the parameter. Minimal Rational Interpolation (MRI) offers a way to locally approximate vector fields which exhibit a meromorphic dependence on one of their parameters.

## Implementations
Built on `numpy` and `scipy`, this repository offers a complete MRI suite for time-harmonic problems in electromagnetism (so-called Maxwell problems). The finite element engine for obtaining solutions to the partial differential equations governing these problems is [`fenics`](https://fenicsproject.org/).

## Example applications
Central to this project are time-harmonic Maxwell problems (THMP), whose parameter is the (angular) frequency $\omega$.
These problems are governed by the time-harmonic Maxwell's equations. Choosing the quantity of interest to be a vector potential $\mathbf{u}$,
these equations reduce to a single curl-curl equation:

$$ \nabla \times (\mu^{-1} \nabla \times \mathbf{u}) - \epsilon \omega^2 \mathbf{u} = \mathbf{j} $$

Three examples for contained in this family of problems are studied:

- The resonant modes of a two-dimensional resonant cavity
- The two-dimensional cavity with an imperfectly conducting boundary
- The scattering coefficients of a dual-mode waveguide filter (DMCWF)

## Quick start
To demonstrate the usage of the MinimalRationalInterpolation class, I hereafter show the reader a simple and straight forward application: Finding resonant frequencies of the cubic unit cavity in a vacuum with trivial physical units.

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
    mu = fen.Expression('1.0', degree=2)
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
       
which will indeed give the first resonant frequency (3.51230205) of the cubic unit cavity with one edge acting as an inlet and trivial physical constants.

## File structure
The core of my implementations are located in the `src/` directory. Illustrative jupyter notebooks for each of the example applications mentioned above can be found in the `examples/` directory. 

```
Maxwell-MRI
│   README.md
|   LICENSE
|
└───examples
|   |
│   └───1_resonant_cavity
|   |   |   resonant_cavity.ipynb               (visualization and demonstration notebook)
|   |   |   rectangular_cavity.py               (rectangular cavities)
|   |   |   two_dimensional_cavity.py           (generic 2D cavities)
|   |
│   └───2_imperfect_conductor
|   |   |   imperfect_conductor.ipynb           (visualization and demonstration notebook)
|   |   |   imperfect_conductor.py              (rectangular cavity with an imperfect boundary)
|   |
│   └───3_circular_waveguide_filter
|   |   └───model
|   |   |   |   DMCWF.geo                       (Gmsh work file)
|   |   |   |   DMCWF.step                      (3D CAD model)
|   |   |   |   DMCWF.xml                       (final mesh)
|   |   |   circular_waveguide_filter.ipynb     (visualization and demonstration notebook)
|   |   |   circular_waveguide_filter.py        (engine for simulating the DMCWF)
|
└───presentation
|   |   ...
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
