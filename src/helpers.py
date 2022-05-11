# -*- coding: utf-8 -*-

import numpy as np

def plot_solution_norms(ax, THMP, VS, **kwargs):
    """Plot the norm of the solution for a set of frequencies"""
    omegas = THMP.get_frequency()
    norms = np.empty(len(omegas))
    u = THMP.get_solution(tonumpy=True, trace=VS.get_trace())
    norms = [VS.norm(a) for a in u]
    ax.plot(omegas, norms, **kwargs)
    ax.set_yscale('log')

def plot_surrogate_norms(ax, MRI, a=None, b=None, N=1000, **kwargs):
    """Plot vectorspace norm of surrogate at N uniform points in [a, b]"""
    if a is None:
        a = np.min(MRI.u_ring.get_nodes())
    if b is None:
        b = np.max(MRI.u_ring.get_nodes())
    linspace = np.linspace(a, b, N)
    ax.plot(linspace, [np.linalg.norm(MRI.R.dot(x)) for x in MRI.u_ring(linspace).T], **kwargs)
    ax.set_yscale('log')

def plot_surrogate_error_norms(ax, THMP, MRI, VS, u=None, omegas=None, RI=None, **kwargs):
    """Plot relative error norm of surrogate"""
    if omegas is None:
        omegas = THMP.get_frequency()
    if u is None:
        u = THMP.get_solution(tonumpy=True, trace=VS.get_trace())
        if RI is None:
            RI = MRI.get_surrogate(u)
    elif RI is None:
        RI = MRI.get_surrogate(THMP.get_solution(tonumpy=True, trace=VS.get_trace()))
    err = [VS.norm(u[i] - RI(x)) / VS.norm(u[i]) for i, x in enumerate(omegas)]
    ax.plot(omegas, err, **kwargs)
    ax.set_yscale('log')
    
def plot_lines(ax, values, **kwargs):
    """Plot vertical lines at given values"""
    ax.vlines(values, ymin=0, ymax=1, **kwargs)
    ax.set_yticks([])
    ax.set_ylim(0, 1)

def complex_scatter(ax, values, **kwargs):
    real = np.real(values)
    imag = np.imag(values)
    ax.scatter(real, imag, marker='x', color='k', **kwargs)
