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
    get_snapshots() : None -> np.ndarray
        Return the snapshot matrix.
    get_frequencies() : None -> np.ndarray
        Return requencies to which the snapshots corresponds.

    """
    def __init__(self, snapshots, omegas):
        self.snapshots = snapshots
        self.omegas = omegas

    def get_snapshots(self):
        return self.snapshots

    def get_frequencies(self):
        return self.omegas
