# -*- coding: utf-8 -*-

import numpy as np
from .minimal_rational_interpolation import MinimalRationalInterpolation
from .snapshot_matrix import SnapshotMatrix

def plot_solution_norms(ax, THMP, VS, **kwargs):
    """Plot the norm of the solution for a set of frequencies"""
    if not isinstance(THMP, SnapshotMatrix) and isinstance(THMP.g_N, list):
        n = len(THMP.g_N)
    else:
        n = 1
    omegas = THMP.get_frequency()
    norms = np.empty(len(omegas))
    u = THMP.get_solution(trace=VS.get_trace())
    norms = [VS.norm(a) for a in u]
    for i in range(n):
        ax.plot(omegas[i::n], norms[i::n], **kwargs)
    ax.set_yscale('log')
    return norms

def plot_surrogate_norms(ax, MRI, a=None, b=None, N=1000, **kwargs):
    """Plot vectorspace norm of surrogate at N uniform points in [a, b]"""
    if a is None:
        a = np.min(MRI.u_ring.get_nodes())
    if b is None:
        b = np.max(MRI.u_ring.get_nodes())
    linspace = np.linspace(a, b, N)
    norms = [np.linalg.norm(MRI.R.dot(x)) for x in MRI.u_ring(linspace).T]
    ax.plot(linspace, norms, **kwargs)
    ax.set_yscale('log')
    return norms

def plot_surrogate_error_norms(ax, THMP, MRI, VS, u=None, omegas=None, RI=None, **kwargs):
    """Plot relative error norm of surrogate"""
    if omegas is None:
        omegas = THMP.get_frequency()
    if u is None:
        u = THMP.get_solution(trace=VS.get_trace())
        if RI is None:
            RI = MRI.get_surrogate(u)
    elif RI is None:
        RI = MRI.get_surrogate(THMP.get_solution(trace=VS.get_trace()))
    err = [VS.norm(u[i] - RI(x)) / VS.norm(u[i]) for i, x in enumerate(omegas)]
    ax.plot(omegas, err, **kwargs)
    ax.set_yscale('log')
    return err
    
def plot_lines(ax, values, **kwargs):
    """Plot vertical lines at given values"""
    ax.vlines(values, ymin=0, ymax=1, **kwargs)
    ax.set_yticks([])
    ax.set_ylim(0, 1)

def complex_scatter(ax, values, **kwargs):
    """Create scatter plot of complex values"""
    real = np.real(values)
    imag = np.imag(values)
    ax.grid()
    ax.scatter(real, imag, **kwargs)

def get_interpolatory_solutions(omega, CWF_L, CWF_R, VS, tol=1e-2):
    """Compute gMRI solution at points 'omega' when forced from left and
    right hand side. Solutions for left and right hand side are computed
    and stored in U_MRI alternatingly for each element in 'omega'."""
    omega_test = np.linspace(np.min(omega), np.max(omega), 1000)
    MRI = MinimalRationalInterpolation(VS)
    MRI_L_hist = MRI.compute_surrogate(CWF_L, omegas=omega_test, greedy=True, tol=tol, solver='fenics', return_history=True)
    U_MRI_L = MRI.evaluate_surrogate(CWF_L.get_solution(), omega).T
    MRI_R_hist = MRI.compute_surrogate(CWF_R, omegas=omega_test, greedy=True, tol=tol, solver='fenics', return_history=True)
    U_MRI_R = MRI.evaluate_surrogate(CWF_R.get_solution(), omega).T
    U_MRI = np.empty((2*len(omega), U_MRI_R.shape[1]), dtype=complex)
    U_MRI[::2] = U_MRI_L
    U_MRI[1::2] = U_MRI_R
    return U_MRI, MRI_L_hist, MRI_R_hist
    
def compute_scattering_matrices(F, U, freq, freq_c, freq_0, mu, eps):
    """Computation of the scattering matrices for each pair of solutions
    (forced from left- and right-hand side) corresponding to a frequency"""
    N = len(freq) // 2
    S = [np.empty((2, 2), dtype=complex)] * N
    for n in range(N):
        S[n][0, 0] = np.inner(F[0], U[2*n])
        S[n][0, 1] = np.inner(F[1], U[2*n])
        S[n][1, 0] = np.inner(F[0], U[2*n+1])
        S[n][1, 1] = np.inner(F[1], U[2*n+1])
        S[n] *= np.sqrt(mu / eps)*1j*freq[n]/(2*np.pi)*np.sqrt((1 - (freq_c/freq_0)**2) / (1 - (freq_c / freq[n])**2))
        S[n] = np.eye(2) - 2 * np.linalg.inv(np.eye(2) + S[n])
    return S

def get_scattering_coefficients(S):
    """Convert scattering matrices to scattering coefficients"""
    S11 = np.abs(S)[:, 0, 0]
    S21 = np.abs(S)[:, 1, 0]
    S12 = np.abs(S)[:, 0, 1]
    S22 = np.abs(S)[:, 1, 1]
    return S11, S21, S12, S22

def convert_to_dezibel(x):
    """Convert list to decibel scale"""
    return 20*np.log(x / np.max(x))
