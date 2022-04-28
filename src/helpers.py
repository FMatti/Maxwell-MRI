# -*- coding: utf-8 -*-

import numpy as np

def plot_solution_norms(ax, THMP, VS, **kwargs):
    """Plot the norm of the solution for a set of frequencies"""
    omegas = THMP.get_frequency()
    norms = np.empty(len(omegas))
    A = THMP.get_solution(tonumpy=True, trace=VS.get_trace())
    norms = [VS.norm(a) for a in A]
    ax.plot(omegas, norms, **kwargs)
    ax.set_yscale('log')

def plot_surrogate_norms(ax, MRI, VS, a=None, b=None, N=1000, **kwargs):
    """Plot vectorspace norm of surrogate at N uniform points in [a, b]"""
    if a is None:
        a = np.min(MRI.RI.get_nodes())
    if b is None:
        b = np.max(MRI.RI.get_nodes())
    linspace = np.linspace(a, b, N)
    ax.plot(linspace, [VS.norm(MRI.RI(x)) for x in linspace], **kwargs)
    ax.set_yscale('log')

def plot_surrogate_error_norms(ax, THMP, MRI, VS, **kwargs):
    """Plot relative error norm of surrogate"""
    omegas = THMP.get_frequency()
    A = THMP.get_solution(tonumpy=True, trace=VS.get_trace())
    RI = MRI.get_surrogate()
    err = [VS.norm(A[i] - RI(x)) / VS.norm(A[i]) for i, x in enumerate(omegas)]
    ax.plot(omegas, err, **kwargs)
    ax.set_yscale('log')
    
def plot_lines(ax, values, **kwargs):
    """Plot vertical lines at given values"""
    ax.vlines(values, ymin=0, ymax=1, **kwargs)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
   