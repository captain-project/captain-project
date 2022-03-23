import sys

import numpy as np

from numba import jit
import random
import scipy.stats

small_number = 1e-10


@jit(nopython=True)
def dispersalDistances(length, lambda_0):
    # print("calculating distances...")
    dumping_dist = np.zeros((length, length, length, length))
    for i in range(0, length):
        for j in range(0, length):
            for n in range(0, length):
                for m in range(0, length):
                    exp_rate = 1.0 / lambda_0
                    # relative dispersal probability: always 1 at distance = 0
                    # the actual number of offspring is modulated by growth_rate
                    dumping_dist[i, j, n, m] = np.exp(
                        -exp_rate * np.sqrt((i - n) ** 2 + (j - m) ** 2)
                    )
    return dumping_dist


def add_random_diffusion_mortality(
    length, sig=5, peak_disturbance=0.3, min_death_prob=0.01
):
    indx = np.meshgrid(np.arange(length), np.arange(length))
    locsxy = np.random.uniform(0, length, (2, 1))
    min_sig2 = 1
    sig_tmp = np.random.uniform(min_sig2, sig, 2)
    # print("\n\nlocsxy:", locsxy, "\n")
    disturbance_matrix_tmp = scipy.stats.norm.pdf(
        indx[0], loc=locsxy[0, 0], scale=sig_tmp[0]
    ) * scipy.stats.norm.pdf(indx[1], loc=locsxy[1, 0], scale=sig_tmp[1])

    disturbance_matrix_tmp = (
        disturbance_matrix_tmp / np.max(disturbance_matrix_tmp)
    ) * peak_disturbance + min_death_prob
    disturbance_matrix_tmp[disturbance_matrix_tmp > 0.99] = 1 - small_number
    return disturbance_matrix_tmp


def add_random_error(probs, sig=0.1):
    rates = -np.log(1 - probs)
    log_rates = np.log(rates)
    tmp_log_rates = np.random.normal(0, sig * log_rates, probs.shape)
    rnd_log_rates = log_rates + tmp_log_rates
    probs = 1 - np.exp(-np.exp(rnd_log_rates))
    probs = np.maximum(probs, np.zeros(rates.shape) + small_number)
    probs = np.minimum(probs, np.ones(rates.shape) - small_number)
    return probs


def add_random_error_per_species(probs, sig=0.1):
    rates = -np.log(1 - probs)
    log_mean_rates = np.abs(np.log(np.mean(rates)))
    tmp_rates = np.exp(np.random.normal(0, sig * log_mean_rates, probs.shape[0]))
    rnd_rates = np.einsum("sij,s-> sij", rates, tmp_rates)
    probs = 1 - np.exp(-rnd_rates)
    probs = np.maximum(probs, np.zeros(rates.shape) + small_number)
    probs = np.minimum(probs, np.ones(rates.shape) - small_number)
    return probs


def get_alpha_K(probs, N_K_ratio):
    rates = -np.log(1 - probs)
    new_rates = np.maximum((rates * (N_K_ratio - 1)), np.zeros(rates.shape))
    probs = 1 - np.exp(-new_rates)
    # print(np.max(probs),np.max(N_K_ratio),np.min(probs),np.min(N_K_ratio))
    return probs


def get_alpha_K_(probs, N_K_ratio):
    prob_death = np.zeros(probs.shape)
    # print( N_K_ratio.shape,prob_death.shape, N_K_ratio )
    prob_death[:, N_K_ratio > 1] = probs[:, N_K_ratio > 1]
    return prob_death

class empty_tree:
    def length(self):
        return 0

def extract_tree_with_taxa_labels(tree, labels):
    try:
        subtree = tree.extract_tree_with_taxa_labels(labels=labels)
    except:
        subtree = empty_tree()
    return subtree


class SimGrid(object):
    def __init__(
        self,
        length: int,
        num_species: int,
        alpha: float,
        K_max: float,
        lambda_0: float,
        disturbanceInitializer: object,
        disturbance_sensitivity: object,
        selectivedisturbanceInitializer: object = 0,
        selective_sensitivity: object = [],
        immediate_capacity: object = False,
        truncateToInt: object = False,
        species_threshold: object = 1,
        rnd_alpha: object = 0,
        K_disturbance_coeff: object = 1,
        actions: object = [],
        dispersal_before_death: object = 0,
        rnd_alpha_species: object = 0,
        climateModel: object = 0,
        growth_rate: object = np.ones(1),
        phyloGenerator: object = 0,
        climate_sensitivity: object = [],
        climate_as_disturbance=1,
        disturbance_dep_dispersal=1,
    ):
        self._length = length
        self._n_species = num_species
        self._species_id = np.arange(num_species)
        self._alpha = alpha  # fraction killed (1 number)
        self._K_max = K_max  # initial (max) carrying capacity
        self._lambda_0 = (
            lambda_0  # relative dispersal probability: always 1 at distance = 0
        )
        if len(growth_rate) < num_species:
            self._growth_rate = np.ones(num_species) * growth_rate
        else:
            self._growth_rate = growth_rate  # potential number of offspring per individual per year at distance = 0
        self._disturbanceInitializer = disturbanceInitializer
        self._disturbance_matrix = np.zeros((self._length, self._length))
        self._K_cells = (1 - self._disturbance_matrix) * self._K_max
        self._K_disturbance_coeff = (
            K_disturbance_coeff  # if set to 0.5, K is 0.5*(1-disturbance)
        )
        self._counter = 0
        self._species_threshold = species_threshold
        self._dispersal_before_death = dispersal_before_death  # set to 1/0 to get dispersing pool before/after death

        self._disturbance_sensitivity = (
            disturbance_sensitivity  # vector of sensitivity per species
        )
        self._alpha_histogram = self.alphaHistogram(
            self._disturbance_sensitivity, self._disturbance_matrix
        )
        self._rnd_alpha = rnd_alpha
        self._rnd_alpha_species = rnd_alpha_species
        self._immediate_capacity = immediate_capacity
        self._truncateToInt = truncateToInt

        if len(actions) == 0:
            self._selective_disturbance_matrix = np.zeros((self._length, self._length))
            self._protection_matrix = np.zeros((self._length, self._length))
        else:
            self._selective_disturbance_matrix = actions[0]
            self._protection_matrix = actions[1]
        self._selectivedisturbanceInitializer = selectivedisturbanceInitializer

        self._selective_sensitivity = (
            selective_sensitivity  # vector of selective sensitivity per species
        )
        self._selective_alpha_histogram = self.alphaHistogram(
            self._selective_sensitivity, self._selective_disturbance_matrix
        )
        # self.updateSelectiveAlphaHistogram() #TODO check if you can use this instead
        # TODO do we need to do this also? self._h = self._h * (1 - self._selective_alpha_histogram)
        # self._alpha_by_cell = np.ones((length,length))
        self._climate_sensitivity = climate_sensitivity
        self._climate_as_disturbance = climate_as_disturbance
        self._disturbance_dep_dispersal = disturbance_dep_dispersal
        self._disturbance_matrix_diff = 0

        if climateModel == 0:
            self._climateModel = 0
            self._climate_layer = np.zeros((self._length, self._length))
        else:
            self._climateModel = climateModel
            self._climateModel.reset_counter()
            if self._climate_as_disturbance:
                self._climate_layer = self._climateModel.updateClimate(
                    np.zeros((self._length, self._length))
                )
            else:
                self._climate_layer = self._climateModel.updateClimate(
                    np.ones((self._length, self._length))
                )

        # TODO: remove dependency on phylo data?
        if phyloGenerator == 0:
            from ..biodivinit.PhyloGenerator import ReadRandomPhylo as phyloGenerator

            try:
                self._phyloGenerator = phyloGenerator(
                    phylofolder="data_dependencies/phylo/"
                )
                (
                    self._phylo_tree,
                    self._all_tip_labels,
                    self._phylo_ed,
                    self._phylo_file_name,
                ) = self._phyloGenerator.getPhylo()
            except:
                from ..biodivinit.PhyloGenerator import SimRandomPhylo as phyloGenerator

                self._phyloGenerator = phyloGenerator(n_species=self._n_species)
                (
                    self._phylo_tree,
                    self._all_tip_labels,
                    self._phylo_ed,
                    self._phylo_file_name,
                ) = self._phyloGenerator.getPhylo()
        else:
            try:
                (
                    self._phylo_tree,
                    self._all_tip_labels,
                    self._phylo_ed,
                    self._phylo_file_name,
                ) = phyloGenerator.getPhylo()
            except:
                from ..biodivinit.PhyloGenerator import SimRandomPhylo as phyloGenerator

                self._phyloGenerator = phyloGenerator(n_species=self._n_species)
                (
                    self._phylo_tree,
                    self._all_tip_labels,
                    self._phylo_ed,
                    self._phylo_file_name,
                ) = self._phyloGenerator.getPhylo()

    def get_sp_pd_contribution(self):
        totalpd = self._phylo_tree.length()
        phylo_ed = np.zeros(self._n_species)
        # print(len(self._all_tip_labels))
        c = 0
        for i in self._all_tip_labels:
            subtree = extract_tree_with_taxa_labels(self._phylo_tree,
                                                    labels=self._all_tip_labels[self._all_tip_labels != i])
            # subtree = self._phylo_tree.extract_tree_with_taxa_labels(
            #     labels=self._all_tip_labels[self._all_tip_labels != i]
            # )
            phylo_ed[c] = totalpd - subtree.length()
            c += 1
        self._phylo_ed = phylo_ed / np.sum(phylo_ed) * self._n_species

    def alphaHistogram(self, disturbanceSensitivity, disturbanceMatrix):
        "when alphaHistogram==0: nobody dies, when==1: all die"
        return np.einsum("s,ij->sij", disturbanceSensitivity, disturbanceMatrix)

    def setProtectionMatrix(self, protection_matrix):
        self._protection_matrix = protection_matrix

    def setSelectiveDisturbanceMatrix(self, selective_disturbance_matrix):
        self._selective_disturbance_matrix = selective_disturbance_matrix

    def updateAlphaHistogram(self):
        new_dist = self._disturbanceInitializer.updateDisturbance(
            self._disturbance_matrix
        )
        self._disturbance_matrix_diff = np.mean(new_dist - self._disturbance_matrix)
        self._disturbance_matrix = new_dist
        self._alpha_histogram = self.alphaHistogram(
            self._disturbance_sensitivity,
            self._disturbance_matrix * (1 - self._protection_matrix),
        )
        if self._rnd_alpha > 0:
            self._alpha_histogram = add_random_error(
                self._alpha_histogram, sig=self._rnd_alpha
            )
        if self._rnd_alpha_species > 0:
            self._alpha_histogram = add_random_error_per_species(
                self._alpha_histogram, sig=self._rnd_alpha_species
            )

    def get_species_mid_coordinate(self):
        med_lat, med_lon = [], []
        for sp_i in range(self._n_species):
            tmp = np.sum(self._h[sp_i, :, :], axis=1)
            lat_range = np.where(tmp > 0)
            tmp = np.sum(self._h[sp_i, :, :], axis=0)
            lon_range = np.where(tmp > 0)
            med_lat.append(np.median(lat_range))
            med_lon.append(np.median(lon_range))
            # print(i, median_latitude, median_longitude)
        return np.array([med_lat, med_lon])

    def getClimateTolerance(self):
        temp = self._h + 0
        temp[temp > 1] = 1
        temp[temp < 1] = 0
        max_T = np.array(
            [
                np.max(self._climate_layer[temp[sp_i, :, :] == 1])
                for sp_i in range(self._n_species)
            ]
        )
        min_T = np.array(
            [
                np.min(self._climate_layer[temp[sp_i, :, :] == 1])
                for sp_i in range(self._n_species)
            ]
        )
        climate_tolerance_range = np.array(
            [min_T, max_T]
        ).T  # 2D array: species x 2 (min/max)
        # for species reaching the boundaries make up tolerance ranges wider than the full grid
        rnd_ranges = np.random.uniform(
            0,
            np.max(self._climate_layer) - np.min(self._climate_layer),
            climate_tolerance_range.shape,
        )
        climate_tolerance_range[
            climate_tolerance_range[:, 1] == np.max(self._climate_layer), 1
        ] += rnd_ranges[climate_tolerance_range[:, 1] == np.max(self._climate_layer), 1]
        climate_tolerance_range[
            climate_tolerance_range[:, 0] == np.min(self._climate_layer), 0
        ] -= rnd_ranges[climate_tolerance_range[:, 0] == np.min(self._climate_layer), 0]
        # mid-point, half-range
        climate_tolerance = np.array(
            [
                np.mean(climate_tolerance_range, axis=1),
                np.diff(climate_tolerance_range, axis=1)[:, 0] / 2.0,
            ]
        ).T
        n = np.ones(self._climate_layer.shape)
        # 3D: species x long x lat (values repeated for each species across all cells)
        climate_opt_sp_3D = np.einsum("s,ij -> sij", climate_tolerance[:, 0], n)
        climate_range_sp_3D = np.einsum("s,ij -> sij", climate_tolerance[:, 1], n)
        return climate_opt_sp_3D, climate_range_sp_3D

    def updateSelectiveAlphaHistogram(self):
        if self._selectivedisturbanceInitializer != 0:
            self._selective_disturbance_matrix = (
                self._selectivedisturbanceInitializer.updateDisturbance(
                    self._selective_disturbance_matrix
                )
            )
        else:  # in this case selective_disturbance = disturbance
            self._selective_disturbance_matrix = self._disturbance_matrix
        self._selective_alpha_histogram = self.alphaHistogram(
            self._selective_sensitivity,
            self._selective_disturbance_matrix * (1 - self._protection_matrix),
        )

    # def getAlphaByCell(self):
    # 	self._alpha_by_cell = self._alpha_by_cell * self._alpha
    # 	self._alpha_by_cell = add_random_error(self._alpha_by_cell,sig=self._rnd_alpha)

    def updateKcells(self):
        # carrying capacity should change with disturbance
        self._K_cells = (
            (1 - (self._disturbance_matrix * (1 - self._protection_matrix)))
            * self._K_disturbance_coeff
        ) * self._K_max

    def totalCapacity(self):
        return np.einsum("ij->", self._K_cells)

    def initGrid(self, stateInitializer):
        # random histogram
        self._h = stateInitializer.getInitialState(
            self._K_max, self._n_species, self._length
        )
        # init dumping factors
        self._dumping_dist = dispersalDistances(self._length, self._lambda_0)
        self.updateAlphaHistogram()
        self._climate_opt_sp_3D, self._climate_range_sp_3D = self.getClimateTolerance()
        if self._disturbance_dep_dispersal:
            sys.exit("Disturbance-dependent dispersal not implemented")
            # self._diag_list = getDiag.get_diagonals_from_pickle("../scripts/diagonals50.pkl")

    def individualsPerSpecies(self):
        return np.einsum("sij->s", self._h)

    def protectedIndPerSpecies(self):
        tmp = np.einsum("sij,ij->sij", self._h, self._protection_matrix)
        return np.einsum("sij->s", tmp)

    def individualsPerCell(self):
        return np.einsum("sij->ij", self._h)

    def speciesPerCell(self):
        presence_absence = self._h + 0
        presence_absence[
            presence_absence < 1
        ] = 0  # species_threshold is only used for total pop size
        presence_absence[presence_absence > 1] = 1  # not within each cell
        return np.einsum("sij->ij", presence_absence)

    def pdPerCell(self):  # calculate phylogenetic diversity (superslow)
        pd_grid = np.zeros((self._length, self._length))
        for i in range(0, self._length):
            for j in range(0, self._length):
                tmp_sp = self._h[:, i, j]
                if len(tmp_sp[tmp_sp > 1]) >= 2:
                    labels = self._all_tip_labels[tmp_sp > 1]
                    tree_cell = extract_tree_with_taxa_labels(
                        self._phylo_tree,
                        labels=labels
                    )
                    pd_grid[i, j] = tree_cell.length()
                else:
                    pd_grid[i, j] = 0
        return pd_grid

    def edPerCell(self):  # calculate evolutionary distinctiveness
        presence_absence = self._h + 0
        presence_absence[
            presence_absence < 1
        ] = 0  # species_threshold is only used for total pop size
        presence_absence[presence_absence > 1] = 1  # not within each cell
        return np.einsum("sij,s->ij", presence_absence, self._phylo_ed)

    def edPerSpecies(self):
        return self._phylo_ed

    def numberOfSpecies(self):
        return np.sum(np.einsum("sij->s", self._h) > self._species_threshold)

    def extantSpeciesID(self):
        return self._species_id[np.einsum("sij->s", self._h) > self._species_threshold]

    def extinctSpeciesID(self):
        return self._species_id[np.einsum("sij->s", self._h) < self._species_threshold]

    def totalEDextantSpecies(self):
        return np.sum(
            self._phylo_ed[np.einsum("sij->s", self._h) > self._species_threshold]
        )

    def totalPDextantSpecies(self):
        if self._phylo_tree == 0:
            return 0
        labels = self._all_tip_labels[self.extantSpeciesID()]
        tree_extant = extract_tree_with_taxa_labels(self._phylo_tree, labels=labels)
        pd_extant = tree_extant.length()
        return pd_extant

    def numberOfIndividuals(self):
        return np.einsum("sij->", self._h)

    def geoRangePerSpecies(self):  # number of occupied cells
        # TODO clean up this: no need for temp, just return np.einsum('sij->s',self._h[ > 1]) ?
        temp = self._h + 0
        temp[temp > 1] = 1
        temp[temp < 1] = 0
        return np.einsum("sij->s", temp)

    def histogram(self):
        return self._h

    def update_dumping_dist(self, fast=False):
        # print(self._disturbance_matrix_diff, np.mean(self._dumping_dist))
        multiplier = 2
        self._dumping_dist = np.exp(
            np.log(self._dumping_dist + small_number)
            - (self._disturbance_matrix_diff * self._lambda_0) * multiplier
        )
        # print(self._disturbance_matrix_diff, np.mean(self._dumping_dist))

    def step(self, action=None, fast_dist=False):
        # if self._counter == 0:
        # 	print(self._disturbance_sensitivity[0:5])
        # evolve the grid one time step
        if self._dispersal_before_death == 1:
            NumCandidates = np.einsum("sij,ijnm->snm", self._h, self._dumping_dist)
            normCandidates = NumCandidates / np.einsum("sij->ij", NumCandidates)
        # update alpha hist (only 1st step for now)
        self.updateAlphaHistogram()

        # update carrying capacity
        self.updateKcells()
        # kill individuals based on new carrying capacity
        if self._immediate_capacity:
            self._h = self._h * (self._K_cells / self._K_max)
            # kill individuals based on natural death rate + disturbance
            self._h = self._h * (1 - (self._alpha + self._alpha_histogram))
        else:
            final_alpha_histogram = get_alpha_K(
                self._alpha_histogram, self.individualsPerCell() / self._K_cells
            )
            # final_alpha_histogram = self._alpha_histogram * (self.individualsPerCell()-self._K_cells)/self._K_cells
            self._h = self._h * (1 - final_alpha_histogram)

        if len(self._selective_sensitivity) > 0:
            # update selective alpha hist
            self.updateSelectiveAlphaHistogram()
            self._h = self._h * (1 - self._selective_alpha_histogram)

        # climate change effects
        if self._climateModel != 0:
            # update climate layer
            self._climate_layer = self._climateModel.updateClimate(self._climate_layer)

            if self._climate_as_disturbance:
                """
                climate model as a regional disturbance C ~ (0,1):
                - species sensitivities as tolerance thresholds t ~ (0,1)
                - if C > t: death_rate = C/t - 1
                death_rate = np.max(0, C/t - 1)
                death_prob = 1 - np.exp(-death_rate)
                """
                death_rate = (
                    np.einsum(
                        "s,ij->sij", 1 / self._climate_sensitivity, self._climate_layer
                    )
                    - 1
                )
                death_rate[death_rate < 0] = 0
                death_prob = 1 - np.exp(-death_rate)

                # print("\n\n\n")
                # print(self._climate_layer)
                # print(death_prob.shape)
                # print(death_prob)
                # print(death_rate)

                self._h = self._h * (1 - death_prob)
            else:
                # Gradual climate change with N-S gradient
                # # distance from optimal climate, then if > range: change death rate
                abs_dist_from_opt = np.abs(
                    self._climate_opt_sp_3D - self._climate_layer
                )
                # print("self._climate_layer", self._climate_layer)
                # (1 over 2*t_range) times distance from the range boundary
                # convert to probability
                tempAlpha = 1 - np.exp(
                    -(1 / (2 * self._climate_range_sp_3D))
                    * (abs_dist_from_opt - self._climate_range_sp_3D)
                )
                # # set to 0 death rate within range
                tempAlpha[abs_dist_from_opt <= self._climate_range_sp_3D] = 0
                # print(abs_dist_from_opt[96,:,0]-self._climate_range_sp_3D[96,:,0])
                # print((1/(2*self._climate_range_sp_3D[96,:,0]))*(abs_dist_from_opt[96,:,0]-self._climate_range_sp_3D[96,:,0]))
                # print( tempAlpha[96,:,0], self._climate_range_sp_3D[96,:,0] )
                # quit()
                self._h = self._h * (1 - tempAlpha)

        # kill anyway by natural death and replace those
        if self._rnd_alpha_species == 0:
            red_hist = self._h * (1 - self._alpha)
        else:
            by_species = 0
            if by_species:
                # add by-species randomness
                tmp_alpha = np.ones(self._n_species) - self._alpha
                tmp_alpha = add_random_error(tmp_alpha, sig=self._rnd_alpha_species)
                # print(tmp_alpha[0:5])
                red_hist = np.einsum("sij,s -> sij", self._h, tmp_alpha)
            else:
                rr = np.random.uniform(0.95, 1)
                tmp_alpha = add_random_diffusion_mortality(
                    self._length,
                    sig=self._rnd_alpha_species,
                    peak_disturbance=rr,
                    min_death_prob=self._alpha,
                )
                # print(tmp_alpha, self._alpha, rr, np.max(tmp_alpha))
                red_hist = np.einsum("sij,ij -> sij", self._h, 1 - tmp_alpha)

        self._h = red_hist

        # how many can be replaced: max( current population-new_carrying capacity, 0)
        available_space = np.maximum(
            self._K_cells - self.individualsPerCell(), np.zeros(self._K_cells.shape)
        )

        if self._disturbance_dep_dispersal:
            self.update_dumping_dist(fast_dist)

        if self._dispersal_before_death == 0:
            NumCandidates = np.einsum(
                "sij,ijnm->snm", self._h, self._dumping_dist
            )  # * self._growth_rate
            NumCandidates = np.einsum("sij,s->sij", NumCandidates, self._growth_rate)
            totCandidates = np.einsum("sij->ij", NumCandidates)
            normCandidates = NumCandidates / (totCandidates + small_number)
        # print(NumCandidates)
        # print(np.einsum('sij,s->sij', NumCandidates,np.random.random(300)))
        # quit()
        # print(normCandidates.shape)
        # print(np.sum(normCandidates,axis=0))
        # print(np.einsum('sij->ij', NumCandidates))
        # quit()

        # replace individuals
        # dNdt = self._growth_rate * self.individualsPerCell() * (available_space/self._K_cells)
        # self._h = self._h + dNdt * normCandidates
        # self._h = self._h + available_space * normCandidates
        tot_replaced = np.minimum(available_space, totCandidates)
        self._h = self._h + tot_replaced * normCandidates
        # print("available_space",available_space)
        if self._truncateToInt:
            self._h = np.rint(self._h)
        # print(self._counter)
        self._counter += 1

    def set_species_values(self, v):
        "This is only used for plotting, nothing else"
        self._species_value_reference = v + 0

    @property
    def length(self):
        return self._length

    @property
    def h(self):
        return self._h

    @property
    def protection_matrix(self):
        return self._protection_matrix

    def getSelectiveDisturbance(self):
        return self._selective_disturbance_matrix

    @property
    def disturbance_matrix(self):
        return self._disturbance_matrix

    @property
    def selective_disturbance_matrix(self):
        return self._selective_disturbance_matrix
