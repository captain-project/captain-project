import sys

import gym
from gym import spaces
import numpy as np
from enum import Enum
from ..biodivsim.SimGrid import SimGrid
import copy

np.set_printoptions(suppress=1, precision=3)
from ..agents.state_monitor import extract_features
import random
from .DisturbanceGenerator import get_disturbance as get_disturbance


class ActionType(Enum):
    Observe = 1
    Protect = 2
    Disturb = 3
    NoAction = 4
    NoObserve = 6  # only check which species are not protected based on old observation, no trends


class Action(object):

    PROTECT_COST = 0.2  # cost per cell
    OBSERVE_COST = 1  # currently not used
    NOOBSERVE_COST = 1  # currently not used

    def __init__(self, actionType: ActionType, value: int, value_quadrant: int):
        self.actionType = actionType
        self.value = value
        self.value_quadrant = value_quadrant


class smallGrid(object):
    def __init__(self, biodivgrid):
        self._length = biodivgrid._length
        self._n_species = biodivgrid._n_species
        self._species_id = biodivgrid._species_id
        self._alpha = biodivgrid._alpha  # fraction killed (1 number)
        self._K_max = biodivgrid._K_max  # initial (max) carrying capacity
        self._lambda_0 = (
            biodivgrid._lambda_0
        )  # relative dispersal probability: always 1 at distance = 0
        self._growth_rate = (
            biodivgrid._growth_rate
        )  # potential number of offspring per individual per year at distance = 0
        self._disturbanceInitializer = biodivgrid._disturbanceInitializer
        self._disturbance_matrix = copy.deepcopy(biodivgrid._disturbance_matrix)
        self._K_cells = (1 - self._disturbance_matrix) * self._K_max
        self._K_disturbance_coeff = (
            biodivgrid._K_disturbance_coeff
        )  # if set to 0.5, K is 0.5*(1-disturbance)
        self._counter = biodivgrid._counter
        self._species_threshold = biodivgrid._species_threshold
        self._disturbance_sensitivity = (
            biodivgrid._disturbance_sensitivity
        )  # vector of sensitivity per species
        self._alpha_histogram = copy.deepcopy(biodivgrid._alpha_histogram)
        self._rnd_alpha = biodivgrid._rnd_alpha
        self._rnd_alpha_species = biodivgrid._rnd_alpha_species
        self._immediate_capacity = biodivgrid._immediate_capacity
        self._truncateToInt = biodivgrid._truncateToInt
        self._selective_disturbance_matrix = copy.deepcopy(
            biodivgrid._selective_disturbance_matrix
        )
        self._protection_matrix = copy.deepcopy(biodivgrid._protection_matrix)

        self._selectivedisturbanceInitializer = (
            biodivgrid._selectivedisturbanceInitializer
        )

        self._selective_sensitivity = copy.deepcopy(biodivgrid._selective_sensitivity)
        self._selective_alpha_histogram = copy.deepcopy(
            biodivgrid._selective_alpha_histogram
        )
        self._climate_sensitivity = copy.deepcopy(biodivgrid._climate_sensitivity)
        self._climate_as_disturbance = copy.deepcopy(biodivgrid._climate_as_disturbance)
        self._disturbance_dep_dispersal = copy.deepcopy(
            biodivgrid._disturbance_dep_dispersal
        )
        self._disturbance_matrix_diff = biodivgrid._disturbance_matrix_diff
        self._h = copy.deepcopy(biodivgrid._h)
        self._climate_layer = copy.deepcopy(biodivgrid._climate_layer)

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

    def individualsPerSpecies(self):
        return np.einsum("sij->s", self._h)

    def individualsPerCell(self):
        return np.einsum("sij->ij", self._h)

    def speciesPerCell(self):
        presence_absence = self._h + 0
        presence_absence[
            presence_absence < 1
        ] = 0  # species_threshold is only used for total pop size
        presence_absence[presence_absence > 1] = 1  # not within each cell
        return np.einsum("sij->ij", presence_absence)

    def geoRangePerSpecies(self):  # number of occupied cells
        # TODO clean up this: no need for temp, just return np.einsum('sij->s',self._h[ > 1]) ?
        temp = self._h + 0
        temp[temp > 1] = 1
        temp[temp < 1] = 0
        return np.einsum("sij->s", temp)

    def histogram(self):
        return self._h

    def numberOfSpecies(self):
        return self._n_species


# TODO: expose path to tree files (trees are otherwise simulated if not found)
# can be passed through the tree_generator argument in BioDivEnv()
def get_phylo_generator(seed=0, n_species=None):
    from ..biodivinit.PhyloGenerator import SimRandomPhylo as phyloGenerator
    tree_generator = phyloGenerator(n_species=n_species)
    # from ..biodivinit.PhyloGenerator import ReadRandomPhylo as phyloGenerator
    # phylo_obj = phyloGenerator(phylofolder="data_dependencies/phylo/", seed=seed)
    return tree_generator


class RunMode(Enum):
    # TODO: rename to FULLMONITORING, ONETIMEMONITORING etc.
    ORACLE = "ORACLE"
    STANDARD = "STANDARD"
    NOUPDATEOBS = "NOUPDATEOBS"
    PROTECTATONCE = "PROTECTATONCE"


# used in reinforce.RichProtectActionAdaptor
class BiodivEnvUtils(object):
    @staticmethod
    def getQuadrandCoord(grid_size, resolution):
        resolution_grid_size = grid_size / resolution
        x_coord = np.arange(0, grid_size + 1, resolution[0])
        y_coord = np.arange(0, grid_size + 1, resolution[1])
        quadrant_coords_list = []

        for x_i in np.arange(0, int(resolution_grid_size[0])):
            for y_i in np.arange(0, int(resolution_grid_size[1])):
                Xs = np.arange(x_coord[x_i], x_coord[x_i + 1])
                Ys = np.arange(y_coord[y_i], y_coord[y_i + 1])
                quadrant_coords_list.append([Xs, Ys])
        return quadrant_coords_list

    @staticmethod
    def getRichAction(action, grid_size, resolution):
        if action == 0:
            return Action(ActionType.Observe, 0, -1)
        # elif action == 1:
        #    return Action(ActionType.Observe, 0)
        else:
            cellList = BiodivEnvUtils.getQuadrandCoord(grid_size, resolution)
            return Action(ActionType.Protect, cellList[action - 1], action)

    @staticmethod
    def getRichProtectAction(action, grid_size, resolution):
        cellList = BiodivEnvUtils.getQuadrandCoord(grid_size, resolution)
        # print(grid_size, resolution, len(cellList), action)
        return Action(ActionType.Protect, cellList[action], action)


class BioDivEnv(gym.Env):
    """BioDiv Environment that follows gym interface"""

    metadata = {"render.modes": ["human_print", "human_plot", "dict_csv_ready"]}

    def __init__(
        self,
        budget,
        gridInitializer,
        length=None,
        n_species=None,
        alpha=0.01,
        K_max=None,
        dispersal_rate=0.1,
        disturbanceGenerator=None,
        disturbance_sensitivity=None,
        selectivedisturbanceInitializer=0,
        selective_sensitivity=[],
        max_fraction_protected=1,
        immediate_capacity=False,
        truncateToInt=False,
        species_threshold=10,
        rnd_alpha=0,
        K_disturbance_coeff=1,
        actions=[],
        dispersal_before_death=0,
        rnd_alpha_species=0,
        climateModel=0,
        ignoreFirstObs=0,
        buffer_zone=1,
        iterations=100,
        verbose=True,
        resolution=np.array([5, 5]),
        numFeatures=10,
        runMode=RunMode.STANDARD,
        worker_id=0,
        observeMode=1,
        use_protection_cost=0,
        rnd_sensitivities=0,
        rnd_disturbance_init=-1,
        tree_generator=0,
        list_species_values=[],
        rewardMode=0,
        climate_sensitivity=[],
        climate_as_disturbance=1,
        disturbance_dep_dispersal=0,
        growth_rate=[1],
        start_protecting=3,
    ):
        super(BioDivEnv, self).__init__()

        if K_max is None or length is None or n_species is None:
            init_data = gridInitializer.getInitialState(1, 1, 1)
            K_max = np.einsum("xyz -> yz", init_data)[0][0]
            length = init_data.shape[1]
            n_species = init_data.shape[0]

        self.lastActionType = None
        self.climateModel = climateModel
        self.climate_sensitivity = climate_sensitivity
        self.climate_as_disturbance = climate_as_disturbance
        self.rnd_alpha_species = rnd_alpha_species
        self.disturbance_dep_dispersal = disturbance_dep_dispersal
        self.actions = actions
        self.K_disturbance_coeff = K_disturbance_coeff
        self.rnd_alpha = rnd_alpha
        self.species_threshold = species_threshold
        self.truncateToInt = truncateToInt
        self.immediate_capacity = immediate_capacity
        self.selective_sensitivity = selective_sensitivity
        self.selectivedisturbanceInitializer = selectivedisturbanceInitializer
        self.disturbance_sensitivity = disturbance_sensitivity
        self.disturbanceGenerator = disturbanceGenerator
        self.dispersal_rate = dispersal_rate
        self.K_max = K_max

        self.alpha = alpha
        self.timeSinceLastObserve = None
        self.timeOfLastProtect = 0
        self.length = length
        self.num_quadrants = int((length / resolution[0]) * (length / resolution[1]))
        self.n_discrete_actions = (
            self.num_quadrants + 1
        )  # num_quadrants to Protect plus 1 to Observe (plus 1 to do nothing-removed)
        self.buffer_zone = buffer_zone  # size of buffer zone within protected area (with lower protection)
        self.rnd_sensitivities = rnd_sensitivities
        self.rnd_disturbance_init = rnd_disturbance_init
        # calc absolute budget from a fraction
        if budget < 1:
            total_cost = Action.PROTECT_COST * (self.length ** 2)
            self._initialBudget = budget * total_cost
        else:
            self._initialBudget = budget
        self.action_space = spaces.Discrete(self.n_discrete_actions)
        self.num_features = numFeatures  # as we now have 7 features per quadrant TODO this needs to be in init
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(1, self.num_quadrants * self.num_features)
        )
        self.ignoreFirstObs = ignoreFirstObs

        self.iterations = iterations
        self.n_species = n_species
        self.resolution = resolution
        self.runMode = runMode
        self.observeMode = observeMode
        self.rewardMode = rewardMode
        self.bioDivGrid = SimGrid(
            length,
            n_species,
            alpha,
            K_max,
            dispersal_rate,
            disturbanceGenerator,
            disturbance_sensitivity,
            selectivedisturbanceInitializer=selectivedisturbanceInitializer,
            selective_sensitivity=selective_sensitivity,
            immediate_capacity=immediate_capacity,
            truncateToInt=truncateToInt,
            species_threshold=species_threshold,
            rnd_alpha=rnd_alpha,
            K_disturbance_coeff=K_disturbance_coeff,
            dispersal_before_death=dispersal_before_death,
            actions=actions,
            rnd_alpha_species=rnd_alpha_species,
            climateModel=climateModel,
            climate_sensitivity=climate_sensitivity,
            climate_as_disturbance=self.climate_as_disturbance,
            disturbance_dep_dispersal=self.disturbance_dep_dispersal,
            growth_rate=growth_rate,
        )

        self._gridInitializer = gridInitializer
        self._verbose = verbose
        if worker_id > 0:
            self._verbose = 0
        self._max_n_protected_cells = int(max_fraction_protected * self.length ** 2)
        self.protected_quadrants = []
        self.use_protection_cost = use_protection_cost
        self.cost_protected_quadrants = 0
        self.list_species_values = list_species_values
        self.tree_generator = tree_generator
        self._growth_rate = growth_rate
        self._baseline_cost = Action.PROTECT_COST * (
            self.resolution[0] * self.resolution[1]
        )
        self._cost_coeff = 0.4
        self._start_protecting = (
            start_protecting  # n. steps after which protection policy starts
        )
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self._default_action = Action(ActionType(4), 0, 0)
        self.reset()

    def _initEnv(self):
        self.budget = self._initialBudget
        self.currentIteration = 0
        self.n_extant = self.n_species
        if self._verbose:
            print("re-loading grid...")

        if self.rnd_sensitivities:
            rr = random.randint(1000, 9999)
            np.random.seed(rr)
            self.disturbance_sensitivity = np.zeros(self.n_species) + np.random.random(
                self.n_species
            )
            self.selective_sensitivity = np.random.beta(0.2, 0.7, self.n_species)
            self.climate_sensitivity = np.random.beta(2, 2, self.n_species)
            # print(rr, "Rnd sens", disturbance_sensitivity[0:5])
        if self._verbose:
            print("Rnd sens", self.disturbance_sensitivity[0:5])
            print("Rnd climate sens", self.climate_sensitivity[0:5])

        if self.rnd_disturbance_init != -1:
            rr = random.randint(1000, 9999)
            np.random.seed(rr)
            distb_obj, selectivedistb_obj = get_disturbance(self.rnd_disturbance_init)
            self.disturbanceGenerator = distb_obj
            self.selectivedisturbanceInitializer = selectivedistb_obj

        if self.tree_generator:
            pass
        else:
            # random tree sampler
            self.tree_generator = get_phylo_generator(n_species=self.n_species)

        self.bioDivGrid = SimGrid(
            self.length,
            self.n_species,
            self.alpha,
            self.K_max,
            self.dispersal_rate,
            self.disturbanceGenerator,
            self.disturbance_sensitivity,
            selectivedisturbanceInitializer=self.selectivedisturbanceInitializer,
            selective_sensitivity=self.selective_sensitivity,
            immediate_capacity=self.immediate_capacity,
            truncateToInt=self.truncateToInt,
            species_threshold=self.species_threshold,
            rnd_alpha=self.rnd_alpha,
            K_disturbance_coeff=self.K_disturbance_coeff,
            actions=self.actions,
            rnd_alpha_species=self.rnd_alpha_species,
            climateModel=self.climateModel,
            phyloGenerator=self.tree_generator,
            climate_sensitivity=self.climate_sensitivity,
            climate_as_disturbance=self.climate_as_disturbance,
            disturbance_dep_dispersal=self.disturbance_dep_dispersal,
            growth_rate=self._growth_rate,
        )
        self.bioDivGrid.initGrid(self._gridInitializer)
        if not self.disturbance_dep_dispersal:
            self.grid_obj_previous = copy.deepcopy(self.bioDivGrid)
            self.grid_obj_most_recent = copy.deepcopy(self.bioDivGrid)

        # ALTERNATIVE COPY
        # TODO: check if anternative (faster) copy of the object can be always used
        # (i.e. not only with self.disturbance_dep_dispersal)
        else:
            self.grid_obj_previous = smallGrid(self.bioDivGrid)
            self.grid_obj_most_recent = smallGrid(self.bioDivGrid)

        self.quadrant_coords_list = BiodivEnvUtils.getQuadrandCoord(
            self.length, self.resolution
        )

        sp_coord = self.bioDivGrid.get_species_mid_coordinate()
        max_value_coord = []
        if len(self.list_species_values) == 0 or self.rnd_sensitivities:
            randval = np.random.choice([0.1, 10], self.n_species, p=[0.8, 0.2])
            self.list_species_values = randval / np.sum(randval) * self.n_species
            max_value_coord = np.random.choice(range(self.length), 2)

        elif len(self.list_species_values) == 2:
            max_value_coord = self.list_species_values[0]
            self.list_species_values = self.list_species_values[1]

        if len(max_value_coord):
            sp_dist_from_opt_lat = abs(max_value_coord[0] - sp_coord[0, :])
            sp_dist_from_opt_lon = abs(max_value_coord[1] - sp_coord[1, :])
            sp_dist_from_opt = 0.5 + np.sqrt(
                sp_dist_from_opt_lat ** 2 + sp_dist_from_opt_lon ** 2
            )
            geo_values = []
            for i in range(len(self.list_species_values)):
                geo_values.append(
                    self.list_species_values[i] / (sp_dist_from_opt[i] ** 2)
                )

            geo_values = np.array(geo_values) / np.sum(geo_values) * self.n_species
            self.list_species_values = geo_values + 0

        if self._verbose:
            print("Rnd sp values: ", self.list_species_values[0:5])

            # for i in range(len(self.list_species_values)):
            #     print(i, sp_dist_from_opt[i], np.round(self.list_species_values[i], 4), np.round(geo_values[i], 4))

        # set species value in biodivGrid for reference
        self.bioDivGrid.set_species_values(self.list_species_values)

        self.value_extant_sp = np.sum(self.list_species_values)
        self.pd_extant_sp = self.bioDivGrid.totalPDextantSpecies()
        self.n_protected_cells = np.sum(self.bioDivGrid.protection_matrix > 0.0)

        self.n_extant_init = np.copy(self.n_extant)
        self.value_extant_sp_init = np.copy(self.value_extant_sp)
        self.pd_extant_sp_init = np.copy(self.pd_extant_sp)
        self.done_protect_all_steps = 0
        self.history = [[1, 1, 1]]

    def _protectCellList(self, cellList):
        """from an action in the action space to an actual protection matrix and selective disturbance matrix
        currently not updating the selective disturbance
        """
        # TODO: do we need this np.copy?
        protectionMatrix = np.copy(self.bioDivGrid.protection_matrix)
        pcellList = []
        for i in cellList[0]:
            for j in cellList[1]:
                if self.buffer_zone > 0:
                    if (
                        i in cellList[0][: self.buffer_zone]
                        or i in cellList[0][-self.buffer_zone :]
                        or j in cellList[1][: self.buffer_zone]
                        or j in cellList[1][-self.buffer_zone :]
                    ):
                        protectionMatrix[i][j] = 0.5
                    else:
                        protectionMatrix[i][j] = 1.0
                else:
                    protectionMatrix[i][j] = 1
                pcellList.append((i, j))
        # print(f'Protecting Cells: {pcellList}')
        self.bioDivGrid.setProtectionMatrix(protectionMatrix)

    def _canProtect(self):
        canProtect = (
            self.bioDivGrid.protection_matrix > 0
        ).sum() < self._max_n_protected_cells
        return canProtect

    def observe(self, timeSinceLastObserve=0.0):
        self.timeSinceLastObserve = timeSinceLastObserve
        if not self.disturbance_dep_dispersal:
            self.grid_obj_previous = copy.deepcopy(self.grid_obj_most_recent)
            self.grid_obj_most_recent = copy.deepcopy(self.bioDivGrid)
        else:
            # ALTERNATIVE COPY
            self.grid_obj_previous = smallGrid(self.grid_obj_most_recent)
            self.grid_obj_most_recent = smallGrid(self.bioDivGrid)

        # return extract_features(self.grid_obj_most_recent, self.grid_obj_previous,
        #                         quadrant_resolution = self.resolution,
        #                         current_protection_matrix = self.bioDivGrid.protection_matrix,
        #                         mode=self.observeMode,
        #                         cost_quadrant = self._baseline_cost + self.getProtectCostQuadrant(),
        #                         budget=self.budget,
        #                         sp_values=self.list_species_values)

    def update_protected_quadrants_in_lastObs(self):
        sys.exit("update_protected_quadrants_in_lastObs not implemented")
        # Updates features to account for the latest protected quadrant
        # return extract_features(self.grid_obj_most_recent, self.grid_obj_previous,
        #                         quadrant_resolution = self.resolution,
        #                         current_protection_matrix = self.bioDivGrid.protection_matrix,
        #                         mode=self.observeMode,
        #                         cost_quadrant = self._baseline_cost + self.getProtectCostQuadrant(),
        #                         budget=self.budget,
        #                         sp_values=self.list_species_values)

    def getProtectCostQuadrant(self, coordinates=[], fun=np.sum):
        # with disturbance = 1, protection quadrants = 5x5, coeff_cost = 0.2, baseline_cost = 5
        # price doubles = 10
        # with disturbance = 1, protection quadrants = 5x5, coeff_cost = 0.4, baseline_cost = 5
        # price triples = 15 = 5 + (5*5)*0.4
        # set cost of already protected areas to 0
        dist_tmp = self.bioDivGrid.disturbance_matrix * (
            1 - self.bioDivGrid.protection_matrix
        )
        if len(coordinates) == 0:
            if self.use_protection_cost:
                # calculate for all quadrants
                quadrantCost = list()
                for coor in self.quadrant_coords_list:
                    quadrant_coords = np.meshgrid(coor[0], coor[1])
                    quadrantCost.append(
                        self._cost_coeff * fun(dist_tmp[tuple(quadrant_coords)])
                    )
                return quadrantCost
            else:
                return []
        else:
            if self.use_protection_cost:
                # calculate for one quadrant
                quadrant_coords = np.meshgrid(coordinates[0], coordinates[1])
                return self._cost_coeff * fun(dist_tmp[tuple(quadrant_coords)])
            else:
                return 0

    # def getProtectCostCells(self,  coeff_cost=0.4):
    #     # with disturbance = 1, protection quadrants = 5x5, coeff_cost = 0.2, baseline_cost = 5
    #     # price doubles = 10
    #     # with disturbance = 1, protection quadrants = 5x5, coeff_cost = 0.4, baseline_cost = 5
    #     # price triples = 15 = 5 + (5*5)*0.4
    #     return coeff_cost * fun(self.bioDivGrid.disturbance_matrix)

    def step(self, action: Action = None):
        if action == None:
            action = self._default_action
        # TODO: check/fix TIMETOPROTECT
        TIMETOPROTECT = self._start_protecting
        did_protect = 0
        # if self._verbose:
        #     print(self.lastObs.stats_quadrant[0:5,:])
        # this returns an observation, the reward, a flag to indicate the end of the experiment and additions info in a dict
        # execute action and pay cost pay cost of the action
        self.lastActionType = action.actionType

        if self.runMode == RunMode.ORACLE:
            # only allowed action is protect and do observe at no cost
            if action.actionType != ActionType.Protect:
                raise Exception("only allowed action is protect in ORACLE mode")
            if (
                self.currentIteration < TIMETOPROTECT
            ):  # before step 3 some features are not available
                action.actionType = ActionType.NoAction
            self.observe()
            # self.lastObs = self.observe()

        if self.runMode == RunMode.PROTECTATONCE:
            # only allowed action is protect and do observe at no cost
            if action.actionType != ActionType.Protect:
                raise Exception("only allowed action is protect in PROTECTATONCE mode")
            if (
                self.currentIteration < TIMETOPROTECT
            ):  # before step 3 some features are not available
                action.actionType = ActionType.NoAction
                # self.lastObs = self.observe() # only update biodiv if a step was made

        if self.runMode == RunMode.NOUPDATEOBS:
            # only allowed action is protect and do observe at no cost
            if action.actionType != ActionType.Protect:
                raise Exception("only allowed action is protect in NOUPDATEOBS mode")
            if (
                self.currentIteration < TIMETOPROTECT
            ):  # before step 3 some features are not available
                action.actionType = ActionType.NoAction

        if action.actionType == ActionType.Observe:
            if self.budget >= Action.OBSERVE_COST:
                self.timeSinceLastObserve = 0.0
                self.observe()
                self.budget -= Action.OBSERVE_COST

        elif action.actionType == ActionType.Protect:
            added_protection_cost = self.getProtectCostQuadrant(
                coordinates=action.value
            )
            # print(action.value_quadrant, Action.PROTECT_COST + added_protection_cost, self.budget)
            # print(self.protected_quadrants)
            if self.budget >= (self._baseline_cost + added_protection_cost):
                if self._canProtect():
                    # do not observe the state, keep knowledge as last step, update protection matrix
                    self._protectCellList(action.value)
                    self.protected_quadrants.append(action.value_quadrant)
                    self.budget -= self._baseline_cost + added_protection_cost
                    # self.lastObs = self.update_protected_quadrants_in_lastObs()
                    self.timeOfLastProtect = self.currentIteration + 0
                    # if self.cost_protected_quadrants == 0:
                    self.cost_protected_quadrants = (
                        self._baseline_cost + added_protection_cost
                    )
                    # else:
                    #     self.cost_protected_quadrants = (self.cost_protected_quadrants + (self._baseline_cost + added_protection_cost))/2
                    did_protect = 1

        elif action.actionType == ActionType.NoObserve:
            if self.budget >= Action.NOOBSERVE_COST:
                # self.lastObs = extract_features(self.grid_obj_most_recent, self.grid_obj_previous,
                #                                 quadrant_resolution = self.resolution,
                #                                 current_protection_matrix = self.bioDivGrid.protection_matrix,
                #                                 cost_quadrant = self._baseline_cost + self.getProtectCostQuadrant(),
                #                                 budget=self.budget)
                self.budget -= Action.NOOBSERVE_COST
            # print(self.grid_obj_previous.geoRangePerSpecies())

            # do nothing
            pass
        elif action.actionType == ActionType.NoAction:
            # do nothing
            pass

        else:
            raise NotImplemented("not yet implemented!!")
        # Execute one time step within the environment
        if self._verbose:
            if self.currentIteration == 0:
                self.tmp = 0
            d1 = np.round(np.mean(self.bioDivGrid._disturbance_matrix), 2)
            d2 = np.round(np.mean(self.bioDivGrid._selective_disturbance_matrix), 2)
            # TODO: improve screen output
            s = f"Step: {1 + self.currentIteration}"
            if self.runMode == RunMode.PROTECTATONCE:
                if did_protect == 1:
                    self.tmp += 1
                    s = f"  PU: {1 + self.currentIteration - self._start_protecting}"
                elif 1 + self.currentIteration > self._start_protecting:
                    s = f"Step: {1 + self.currentIteration - self.tmp}"
            print(
                s,
                f"N. protected cells: {np.sum(self.bioDivGrid.protection_matrix > 0.)}",
                f"Mean disturbance: {d1}, {d2}",
                f"Budget: {np.round(self.budget,2)}",
                f"N. species: {self.bioDivGrid.numberOfSpecies()}",
                f"Protection cost: {np.round(self.cost_protected_quadrants,2)}",
            )

        if self.runMode != RunMode.PROTECTATONCE:
            self.bioDivGrid.step()
        elif did_protect == 0:  # finished budget, continue with simulation
            self.observe()  # only update biodiv if a step was made
            if self.done_protect_all_steps == 1:
                self.bioDivGrid.step(fast_dist=False)
                self.done_protect_all_steps = 0
            else:
                self.bioDivGrid.step(fast_dist=True)
        else:
            self.done_protect_all_steps = 1

        # build output by stacking obs and time till last obs
        richObs = self._enrichObs()

        # update counters, compute reward and done flag
        self.currentIteration += 1
        self.timeSinceLastObserve += 1

        if self.rewardMode == 0:  # use species loss
            reward = self.bioDivGrid.numberOfSpecies() - self.n_extant
        elif self.rewardMode == 1:  # use sp value
            reward = (
                np.sum(self.list_species_values[self.bioDivGrid.extantSpeciesID()])
                - self.value_extant_sp
            )
        elif self.rewardMode == 2:  # amount of protected area
            reward = (
                np.sum(self.bioDivGrid.protection_matrix > 0.0) - self.n_protected_cells
            ) / (self.resolution[0] * self.resolution[1])
        elif self.rewardMode == 3:  # use PD loss
            reward = self.bioDivGrid.totalPDextantSpecies() - self.pd_extant_sp
        else:
            sys.exit("\nrewardMode not defined!\n")

        self.n_extant = self.bioDivGrid.numberOfSpecies()
        self.value_extant_sp = np.sum(
            self.list_species_values[self.bioDivGrid.extantSpeciesID()]
        )
        self.pd_extant_sp = self.bioDivGrid.totalPDextantSpecies()
        self.n_protected_cells = np.sum(self.bioDivGrid.protection_matrix > 0.0)
        # flag it done when it reaches the # of iterations
        # done = self.currentIteration ==  self.iterations
        done = self.bioDivGrid._counter == self.iterations
        info = self._getInfo()
        self.history.append(
            [info["ExtantSpecies"], info["ExtantSpeciesValue"], info["ExtantSpeciesPD"]]
        )

        # richObs = state in RL
        # print("""state['grid_obj_most_recent']""", np.mean(richObs['grid_obj_most_recent'].h))
        return richObs, reward, done, info

    def _enrichObs(self):
        state = {"budget_left": self.budget}
        state["full_grid"] = self.bioDivGrid.h
        state["disturbance_matrix"] = self.bioDivGrid.disturbance_matrix
        state["selective_disturbance"] = self.bioDivGrid.selective_disturbance_matrix
        state["grid_obj_most_recent"] = self.grid_obj_most_recent
        state["grid_obj_previous"] = self.grid_obj_previous
        state["resolution"] = self.resolution
        state["protection_matrix"] = self.bioDivGrid.protection_matrix
        state["protection_cost"] = self._baseline_cost + self.getProtectCostQuadrant()
        state["time_since_last_obs"] = self.timeSinceLastObserve
        if self.rewardMode == 3:  # (use PD loss) sp value -> PD contribution
            state["sp_values"] = self.bioDivGrid.get_sp_pd_contribution()
        state["sp_values"] = self.list_species_values
        state["min_pop_requirement"] = None
        state["met_prot_target"] = None
        return state

    def reset(self, initTimeSinceLastObserve=5, fullInfo=False):
        self._initEnv()
        if self.ignoreFirstObs:
            self.observe(initTimeSinceLastObserve)
            # tmp.stats_quadrant *= 0
            # #TODO you may want to consider this not to kill the nn
            # #tmp.stats_quadrant = np.random.random(tmp.stats_quadrant.shape)
            # self.lastObs = tmp
        else:
            self.observe()
        # build output by stacking obs and time till last obs
        richObs = self._enrichObs()
        self.protected_quadrants = []

        if not fullInfo:
            return richObs
        else:
            info = self._getInfo()
            return richObs, 0, False, info

    def _getInfo(self):
        info = {
            "budget_not_done": self.budget > 0.0,
            "can_protect": self._canProtect(),
            "NumberOfProtectedCells": np.sum(self.bioDivGrid.protection_matrix > 0.0),
            "budget_left": self.budget,
            "time_last_protect": self.timeOfLastProtect,
            "CostOfProtection": self.cost_protected_quadrants,
            "ExtantSpecies": self.n_extant / self.n_extant_init,
            "ExtantSpeciesValue": self.value_extant_sp / self.value_extant_sp_init,
            "ExtantSpeciesPD": self.pd_extant_sp / self.pd_extant_sp_init,
        }
        return info

    def render(self, mode="human_print", close=False):
        if mode == "human_print" and self._verbose:
            print(
                f"Iteration: {self.currentIteration}; Budget: {self.budget};"
                f" NumSpecies: {self.bioDivGrid.numberOfSpecies()}; NumIndividuals: {np.sum(self.bioDivGrid.h)}"
            )
        elif mode == "dict_csv_ready":
            return {
                "iteration": self.currentIteration,
                "budget": self.budget,
                "num_species": self.bioDivGrid.numberOfSpecies(),
                "num_individuals": np.sum(self.bioDivGrid.individualsPerSpecies()),
                "mean_disturbance": np.mean(self.bioDivGrid.disturbance_matrix),
                "mean_selective_disturbance": np.mean(
                    self.bioDivGrid.selective_disturbance_matrix
                ),
                "num_protected_cells": np.sum(self.bioDivGrid.protection_matrix > 0),
                "time_since_last_observe": self.timeSinceLastObserve,
                "last_action_type": self.lastActionType,
            }

        else:
            raise (NotImplementedError(f"mode {mode} not implemented!"))

    def _cellCoordinateFromIndex(self, actionValue):
        length = self.bioDivGrid.length
        col = actionValue % length
        row = int(actionValue / length)
        return row, col

    def reset_RunMode(self, mode):
        self.runMode = mode
