# -*- coding: utf-8 -*-

class SnapshotMatrix(object):
    """
    Class for storing the snapshot matrix of a problem.

    Members
    -------
    snapshots : np.ndarray
        Snapshot matrix of the solutions to a THMP.
    omegas : np.ndarray
        Frequencies to which the solutions corresponds.

    Methods
    -------
    get_solution() : None -> np.ndarray
        Return the snapshot matrix.
    get_frequency() : None -> np.ndarray
        Return requencies to which the snapshots corresponds.

    """
    def __init__(self, solution, omega):
        self.solution = solution
        self.omega = omega

    def get_solution(self, trace=None):
        """Return stored solution (trace argument for backward compatibility)"""
        return self.solution

    def get_frequency(self):
        """Return frequencies corresponding to stored solutions"""
        return self.omega
