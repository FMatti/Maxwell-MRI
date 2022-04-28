# -*- coding: utf-8 -*-

import numpy as np

class SnapshotMatrix(object):
    """
    Class holding the snapshot matrix of a problem.

    Members
    -------

    Methods
    -------

    """
    def __init__(self, snapshots, omegas, V=None):
        self.snapshots = snapshots
        self.omegas = omegas
        self.V = V

    def get_snapshots(self):
        return self.snapshots

    def get_frequencies(self):
        return self.omegas

    def get_functionspace(self):
        return self.V
