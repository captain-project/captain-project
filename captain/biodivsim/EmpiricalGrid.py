import csv
import os
import sys

from ..biodivsim.BioDivEnv import BioDivEnv, Action, ActionType, RunMode
from ..agents.state_monitor import *

# extract_features, get_feature_indx, get_thresholds, get_thresholds_reverse, get_quadrant_coord_species_clean
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)  # prints floats, no scientific notation
np.set_printoptions(precision=3)  # rounds all array elements to 3rd digit
from ..biodivsim.StateInitializer import *
from ..algorithms.reinforce import RichProtectActionAdaptor, RichStateAdaptor
from ..agents.policy import PolicyNN, get_NN_model_prm

# from .env_setup import *
from ..algorithms.marxan_setup import *
from ..algorithms.env_setup import *
from ..biodivsim.BioDivEnv import *

"""
quadrant_resolution
current_protection_matrix
grid_obj.length <- n_cells expected to be length^2
grid_obj.h
grid_obj.individualsPerSpecies()
grid_obj.geoRangePerSpecies()
grid_obj._climate_layer <- can default to []
grid_obj._climate_as_disturbance <- can default to 0
"""


class EmpiricalGrid:
    def __init__(self, protection_matrix=None, species_sensitivities=None):
        self._counter = 0
        self._climate_layer = []
        self._climate_as_disturbance = 0
        self._protection_matrix = protection_matrix
        self._disturbance_matrix = None
        self._species_threshold = 1
        self._species_sensitivities = species_sensitivities
        self.list_species_values = None

    def initGrid(
        self,
        puvsp_file=None,
        hist_file=None,
        pu_id_file=None,
        sp_id_file=None,
        pu_info_file=None,
        hist_out_file=None,
        pu_id_out_file=None,
        sp_id_out_file=None,
        output=""

    ):
        self._counter = 0
        if hist_file is None:
            print("Reading file...")
            occs = np.loadtxt(puvsp_file, skiprows=1, delimiter=",")
            "species,pu,amount"
            self._species_id = np.unique(occs[:, 0]).astype(int)
            self._n_species = len(self._species_id)
            self._species_id_indx = np.arange(self._n_species)
            self._pus_id = np.unique(occs[:, 1]).astype(int)
            self._n_pus = len(np.unique(occs[:, 1]))
            self.length = np.sqrt(self._n_pus)  # <- n_cells expected to be length^2
            self._pus_id_ind = np.arange(self._n_pus)
            # init 3D sp histogram
            init_h = np.zeros((self._n_species, self._n_pus, 1))

            print("Parsing species occurrence data...")
            for s in self._species_id_indx:
                if s % 100 == 0:
                    print_update("Species %s-%s" % (s, s + 99))
                sp = occs[occs[:, 0] == self._species_id[s]]
                indx = []
                abundance = []
                for u in range(sp.shape[0]):
                    indx.append(self._pus_id_ind[self._pus_id == sp[u, 1]])
                    abundance.append(sp[u, 2])

                init_h[s, np.array(indx)[:, 0], 0] = np.array(abundance)

            self._h = init_h

            wd = os.path.dirname(os.path.abspath(puvsp_file))

            if hist_out_file is None:
                hist_out_file = output + "hist.npy"
                pu_id_out_file = output + "pu_id.npy"
                sp_id_out_file = output + "sp_id.npy"

            np.save(os.path.join(wd, hist_out_file), self._h)
            np.save(os.path.join(wd, pu_id_out_file), self._pus_id.astype(int))
            np.save(os.path.join(wd, sp_id_out_file), self._species_id.astype(int))
            print(
                "\nFiles:\n",
                os.path.join(wd, hist_out_file),
                "\n",
                os.path.join(wd, pu_id_out_file),
                "\n",
                os.path.join(wd, sp_id_out_file),
                "\nwere created for fast loading the data in the future using the",
                "\nhist_file and puid_file arguments"
            )

        else:
            self._h = np.load(hist_file)
            self._species_id = np.load(sp_id_file)
            self._n_species = len(self._species_id)
            self._species_id_indx = np.arange(self._n_species)
            if pu_id_file:
                self._pus_id = np.load(pu_id_file)
            else:
                self._pus_id = np.arange(self._h.shape[1])
            self._n_pus = len(self._pus_id)
            self.length = np.sqrt(self._n_pus)  # <- n_cells expected to be length^2
            self._pus_id_ind = np.arange(self._n_pus)
        if pu_info_file:
            self.coords = pd.read_csv(pu_info_file)
        else:
            self.coords = None

        if self._protection_matrix is None:
            self._protection_matrix = np.zeros((self._n_pus, 1))
        self._init_protection_matrix = self._protection_matrix + 0
        # keeps a record of init protection matrix (in case PUs already set before starting policy)
        self._h_initial = self._h * 1
        if not self.list_species_values:
            self.list_species_values = np.ones(self._n_species)

    def reset(self):
        self._protection_matrix = self._init_protection_matrix + 0
        self._counter = 0

    def randomize_grid(self):
        rnd_sequence = np.random.choice(
            range(self._n_pus), size=self._n_pus, replace=False
        )
        self._pus_id = self._pus_id[rnd_sequence]
        self._h = self._h[:, rnd_sequence, :]
        self._protection_matrix = self._protection_matrix[rnd_sequence, :]

    def set_disturbance_matrix(self, disturbance):
        self._disturbance_matrix = disturbance

    def individualsPerSpecies(self):
        return np.einsum("sij->s", self._h)

    def individualsPerCell(self):
        return np.einsum("sij->ij", self._h)

    def protectedIndPerSpecies(self):
        "protected individuals per species"
        tmp = np.einsum("sij,ij->sij", self._h, self._protection_matrix)
        return np.einsum("sij->s", tmp)

    def geoRangePerSpecies(self):
        temp = self._h + 0
        temp[temp > 1] = 1
        temp[temp < 1] = 0
        return np.einsum("sij->s", temp)

    def speciesPerCell(self):
        presence_absence = self._h + 0
        presence_absence[
            presence_absence < 1
        ] = 0  # species_threshold is only used for total pop size
        presence_absence[presence_absence > 1] = 1  # not within each cell
        return np.einsum("sij->ij", presence_absence)

    def numberOfSpecies(self):
        return np.sum(np.einsum("sij->s", self._h) > self._species_threshold)

    def extinctSpeciesID(self):
        return self._species_id[np.einsum("sij->s", self._h) < self._species_threshold]

    def extinctSpeciesIndexID(self):
        return self._species_id_indx[np.einsum("sij->s", self._h) < self._species_threshold]

    def step(self):
        # TODO: add binomial draw here based on disturbance matrix?
        self._counter += 1

    def update_protection_matrix(self, protection_matrix=None, indx=None, reset_matrix=False):
        if protection_matrix:
            self._protection_matrix = protection_matrix
        if indx:
            if reset_matrix:
                self._protection_matrix *= 0
            self._protection_matrix[indx] = 1

    def subsample_sp_h(self, disturbance_matrix, seed=0):
        if seed:
            np.random.seed(seed)
        if self._species_sensitivities is not None:
            species_sensitivity = self._species_sensitivities
        else:
            species_sensitivity = np.random.random(self._n_species)
        disturbance_effect_sp = np.repeat(disturbance_matrix, self._h.shape[0]).reshape(
            (self._h.shape[1], self._h.shape[0])
        )
        p = 1 - disturbance_effect_sp * species_sensitivity  # prob of sampling
        # print(p)
        x = np.random.binomial(1, p.T)
        a = self._h_initial[:, :, 0] * x
        a = a.reshape(self._h_initial.shape)
        self._h = a

    @property
    def h(self):
        return self._h

    @property
    def protection_matrix(self):
        return self._protection_matrix

    @property
    def disturbance_matrix(self):
        return self._disturbance_matrix
