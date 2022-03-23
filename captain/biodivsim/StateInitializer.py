import numpy as np
import pickle
import glob, os
import random

from .CellClass import *


def load_CellClass(pklfile):
    try:
        with open(pklfile, "rb") as pkl:
            list_cells_list = pickle.load(pkl)
    except (ValueError):
        import pickle5 as pkl5

        with open(pklfile, "rb") as pkl:
            list_cells_list = pkl5.load(pkl)

    list_cells = []
    for c in list_cells_list:
        list_cells.append(CellClass(c[0], c[1], c[2], c[3], c[4], c[5], c[6]))

    return list_cells


class StateInitializer(object):
    def getInitialState(self, K, num_species, length):
        pass


class PickleInitializer(StateInitializer):
    def __init__(self, pklfile=""):
        self._pklfile = pklfile

    def getInitialState(self, K=None, num_species=None, length=None):
        list_cells = load_CellClass(self._pklfile)

        n_species = len(list_cells[0].species_hist)
        n_cells = np.int(np.sqrt(len(list_cells)))

        sp_hist_3D = np.zeros((n_species, n_cells, n_cells)).astype(int)
        for i in range(len(list_cells)):
            h = list_cells[i].species_hist
            c = list_cells[i].coord
            sp_hist_3D[:, c[0], c[1]] = h

        # update grid object/settings?
        # self._length = n_cells
        # self._num_species = n_species
        # self._K_max = np.sum(sp_hist_3D,axis=0)  # initial (max) carrying capacity

        return sp_hist_3D


class PickleInitializerBatch(StateInitializer):
    def __init__(self, pklfolder, verbose=False, pklfile_i=0):
        self._pklfolder = pklfolder
        self._verbose = verbose
        self._pkl_files = [
            f
            for f in glob.glob(
                os.path.join(self._pklfolder, "init_cell*"), recursive=False
            )
        ]
        self._pklfile_ind = pklfile_i

    def getInitialState(self, K=None, num_species=None, length=None):
        pklfile = self._pkl_files[self._pklfile_ind]
        if self._verbose:
            print(f"selected pickle env: {pklfile}")

        list_cells = load_CellClass(pklfile)

        n_species = len(list_cells[0].species_hist)
        n_cells = np.int(np.sqrt(len(list_cells)))

        sp_hist_3D = np.zeros((n_species, n_cells, n_cells)).astype(int)
        for i in range(len(list_cells)):
            h = list_cells[i].species_hist
            c = list_cells[i].coord
            sp_hist_3D[:, c[0], c[1]] = h

        return sp_hist_3D


class PickleInitializerSequential(StateInitializer):
    def __init__(self, pklfolder, verbose=False, pklfile_i=0):
        self._pklfolder = pklfolder
        self._verbose = verbose
        self._pkl_files = [
            f
            for f in glob.glob(
                os.path.join(self._pklfolder, "init_cell*"), recursive=False
            )
        ]
        self._pklfile_ind = pklfile_i

    def getInitialState(self, K=None, num_species=None, length=None):
        if self._pklfile_ind >= len(self._pkl_files):
            self._pklfile_ind = 0  # reset count
        pklfile = self._pkl_files[self._pklfile_ind]
        if self._verbose:
            print(f"selected pickle env: {pklfile}")

        list_cells = load_CellClass(pklfile)

        n_species = len(list_cells[0].species_hist)
        n_cells = np.int(np.sqrt(len(list_cells)))

        sp_hist_3D = np.zeros((n_species, n_cells, n_cells)).astype(int)
        for i in range(len(list_cells)):
            h = list_cells[i].species_hist
            c = list_cells[i].coord
            sp_hist_3D[:, c[0], c[1]] = h

        self._pklfile_ind += 1

        return sp_hist_3D


class RandomPickleInitializer(StateInitializer):
    def __init__(self, pklfolder, verbose=False):
        self._pklfolder = pklfolder
        self._verbose = verbose
        self._pkl_files = [
            f
            for f in glob.glob(
                os.path.join(self._pklfolder, "init_cell*"), recursive=False
            )
        ]
        print("Found %s files" % len(self._pkl_files))
        # print(self._pkl_files)

    def getInitialState(self, K=None, num_species=None, length=None):
        # randomly select pkl from folder
        pklfile_i = random.choice(np.arange(len(self._pkl_files)))
        pklfile = self._pkl_files[pklfile_i]

        if self._verbose:
            print(f"selected pickle env: {pklfile}")

        list_cells = load_CellClass(pklfile)

        n_species = len(list_cells[0].species_hist)
        n_cells = np.int(np.sqrt(len(list_cells)))

        sp_hist_3D = np.zeros((n_species, n_cells, n_cells)).astype(int)
        for i in range(len(list_cells)):
            h = list_cells[i].species_hist
            c = list_cells[i].coord
            sp_hist_3D[:, c[0], c[1]] = h

        return sp_hist_3D


class RandomStateInitializer(StateInitializer):
    def __init__(self, seed=1):
        self._seed = seed

    def getInitialState(self, K, num_species, length):
        h = np.random.randint(0, K, (num_species, length, length))
        normH = np.einsum("sij->ij", h)
        return np.rint(h / normH * K)


class TwoSpeciesTriangleStateInitializer(StateInitializer):
    def __init__(self, minorityRatio=1):
        self._minorityRatio = minorityRatio

    def getInitialState(self, K, num_species, length):
        h = np.zeros((num_species, length, length))

        for s in (0, 1):
            hist_s = h[s]
            for i in range(0, length):
                for j in range(0, length):
                    if s == 0:
                        if i >= j:
                            # majority class
                            hist_s[i, j] = K
                        else:
                            hist_s[i, j] = K * (1 - self._minorityRatio)
                    else:
                        if i < j:
                            # minority class
                            hist_s[i, j] = K * self._minorityRatio

        normH = np.einsum("sij->ij", h)
        return np.rint(h / normH * K)


class RandomGaussianStateInitializer(StateInitializer):
    def __init__(self, minorityRatio=None, gaussScale=None, truncateToInt=False):
        self._minorityRatio = minorityRatio
        self._gaussScale = gaussScale
        self._truncateToInt = truncateToInt

    def getInitialState(self, K, num_species, length):
        population = int(K * length ** 2 / num_species)
        h = np.ones((num_species, length, length))
        # get random sources
        sources = np.random.randint(0, length - 1, (num_species, 2))
        for s, source in enumerate(sources):
            hist_s = h[s]
            mean = source
            cov = [[1, 0], [0, 1]]
            x = np.random.multivariate_normal(mean, cov, population)
            for nx in x:
                hist_s[
                    min(max(int(nx[0]), 0), length - 1),
                    min(max(int(nx[1]), 0), length - 1),
                ] += 1

        normH = np.einsum("sij->ij", h)
        if self._truncateToInt:
            return np.rint(h / normH * K)
        else:
            return h / normH * K
