import csv
import os
import sys
from ..agents.state_monitor import *
import numpy as np
from ..agents.state_monitor import *
from ..algorithms.reinforce import RichStateAdaptor
from ..agents.policy import PolicyNN, get_NN_model_prm
from ..algorithms.marxan_setup import *
from ..algorithms.env_setup import *
from ..biodivsim.StateInitializer import print_update
np.set_printoptions(suppress=True, precision=3)

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


class DynamicConservationTarget:
    def __init__(self, step_size=0.01):
        self._step_size = step_size
    
    def update_target(self, target_pop):
        new_target = target_pop * (1 + self._step_size)
        return new_target


class BioDivEnvEmpirical:
    def __init__(
            self,
            emp,
            budget,
            runMode,
            rewardMode=0,
            start_protecting=0,
            starting_protection_matrix=None,
            init_sp_quadrant_list_only_once=False,
            iterations=None,
            stop_at_end_budget=True,  # overwrites n. iterations
            stop_at_target_met=False,  #
            cost_pu=None,
            h_seed=0,  # used to subsample h
            protect_fraction=0.1,
            lastObs=None,
            verbose=1,
            drop_unaffordable=False,
            dynamic_target=DynamicConservationTarget(),  # set to None for fixed target
    ):
        self.bioDivGrid = emp
        self.iterations = iterations
        self.runMode = runMode
        self.lastObs = lastObs
        self.rewardMode = rewardMode
        self.verbose = verbose  # 1: verbose, 2: more verbose
        self._stop_at_end_budget = stop_at_end_budget
        self._stop_at_target_met = stop_at_target_met
        self.protection_cost = cost_pu
        self._min_protection_cost = np.min(self.protection_cost)
        self.protection_sequence = []
        # if max prob is smaller than twice the uniform probability then stop updating features
        self.max_prob_threshold = (1 / (emp._n_pus)) * 2
        self.min_pop_requirement = emp.individualsPerSpecies() * protect_fraction
        self.protect_fraction = protect_fraction
        self.sp_quadrant_list = None
        self._init_sp_quadrant_list_only_once = init_sp_quadrant_list_only_once
        self.drop_unaffordable = (
            drop_unaffordable  # sp histogram set to zero in PUs that cant be
        )
        # protected with current budget
        # only works with: RunMode.ORACLE
        self.avg_min_pop_requirement = np.mean(self.min_pop_requirement)
        self._start_protecting = start_protecting
        self._h_seed = h_seed
        self._initialBudget = budget
        self._default_action = Action(ActionType(4), None, None)
        self.resolution = np.array([1, 1])
        self.static_system = True  # no changes in pop sizes, climate, disturbance -> no recurrent observe() needed
        self.starting_protection_matrix = starting_protection_matrix
        self._dynamic_target = dynamic_target
        self.print_freq = 10 # print frequency
        
        self.reset()
    
    def _initEnv(self):
        self.currentIteration = 0
        self.budget = self._initialBudget
        self.previous_action = None
        # randomize species presence/absence based on disturbance
        self.bioDivGrid.subsample_sp_h(
            self.bioDivGrid.disturbance_matrix, seed=self._h_seed
        )
        # reset protection matrix
        self.bioDivGrid.reset()
        self.set_conservation_target(self.protect_fraction)
        self._sp_target_met = self.get_species_met_target()
        
        if self.verbose:
            print(self.bioDivGrid.disturbance_matrix)
            print(np.einsum("sxy -> s", self.bioDivGrid._h))
            print(np.einsum("sxy -> s", self.bioDivGrid._h_initial))
        
        self.grid_obj_previous = copy.deepcopy(self.bioDivGrid)
        self.grid_obj_most_recent = copy.deepcopy(self.bioDivGrid)
        
        # init species pres/absence in each PU only once
        if self._init_sp_quadrant_list_only_once:
            self.sp_quadrant_list = self.get_sp_list_per_PU()
    
    def getProtectCostQuadrant(self):
        return self._cost_protection
    
    def _getInfo(self):
        info = {
            "NumberOfProtectedCells": np.sum(self.bioDivGrid.protection_matrix > 0.0),
            "budget_left": self.budget,
            # 'ExtantSpecies': self.n_extant/self.n_extant_init,
            # 'ExtantSpeciesValue': self.value_extant_sp/self.value_extant_sp_init,
            # 'ExtantSpeciesPD': self.pd_extant_sp/self.pd_extant_sp_init
        }
        return info
    
    def _enrichObs(self):
        state = {"budget_left": self.budget}
        state["full_grid"] = self.bioDivGrid.h
        state["disturbance_matrix"] = self.bioDivGrid.disturbance_matrix
        if self.static_system:
            state["grid_obj_most_recent"] = self.bioDivGrid
        else:
            state["grid_obj_most_recent"] = self.grid_obj_most_recent
        state["grid_obj_previous"] = self.grid_obj_previous
        state["protection_matrix"] = self.bioDivGrid.protection_matrix
        state["resolution"] = self.resolution
        state["min_pop_requirement"] = self.min_pop_requirement
        state["met_prot_target"] = self._sp_target_met
        state["protection_cost"] = self.protection_cost
        state["sp_values"] = self.bioDivGrid.list_species_values
        return state
    
    def observe(self):
        self.grid_obj_previous = copy.deepcopy(self.grid_obj_most_recent)
        self.grid_obj_most_recent = copy.deepcopy(self.bioDivGrid)
    
    def reset(self):
        self._initEnv()
        self.observe()
        richObs = self._enrichObs()
        info = self._getInfo()
        done = False
        reward = 0
        return richObs, reward, done, info
    
    def step(self, action=None):
        if action == None:
            action = self._default_action
        self.bioDivGrid.step()
        
        if self.currentIteration < self._start_protecting:
            action.actionType = ActionType.NoAction
        
        if action.actionType == ActionType.NoAction:
            pass
        
        elif action.actionType == ActionType.Protect:
            # compare action to previous
            if action == self.previous_action:
                print("action == previous_action")
                print(self.protection_cost[action.value], self.budget)
                if self.deterministic:
                    self.bioDivGrid._counter = self.iterations
            # protect unit
            if self.protection_cost[action.value] <= self.budget:
                # print("PROTECT!", action.value)
                self.budget -= self.protection_cost[action.value]
                self.bioDivGrid.update_protection_matrix(indx=action.value)
                
                if self.drop_unaffordable:
                    self.drop_unaffordable_cells()
                
                if self.lastObs:  # self.runMode == RunMode.NOUPDATEOBS:
                    # approximate update of lastObs
                    prot_indx = np.where(self.bioDivGrid.protection_matrix[:, -1] > 0)[
                        0
                    ]
                    # print("update!", prot_indx)
                    # print(self.lastObs.stats_quadrant[prot_indx, -1], self.bioDivGrid.protection_matrix[prot_indx, 0])
                    self.lastObs.stats_quadrant[
                        prot_indx, -1
                    ] = self.bioDivGrid.protection_matrix[prot_indx, 0]
                    # self.lastObs.stats_quadrant[prot_indx, :] = 1 - self.bioDivGrid.protection_matrix[prot_indx, 0]
                self.protection_sequence.append(action.value)
            self.previous_action = action
        else:
            raise Exception("only allowed action are NoAction/Protect")
        
        if self.verbose or self.bioDivGrid._counter % self.print_freq == 0:
            action_cost = 0
            if action.actionType == ActionType.Protect:
                action_cost = self.protection_cost[action.value]
            
            print_update(
                "%s – selected PU: %s " % (self.bioDivGrid._counter, action.value) +
                "n. PUs: %s " % (int(np.sum(self.bioDivGrid.protection_matrix))) +
                # "cost:", np.round(action_cost, 2),
                "budget (%):" + " %s " % (np.round(self.budget / self._initialBudget * 100, 2)) +
                "Current target: %s, %s, %s,... "
                % tuple(np.round(self.min_pop_requirement[0:3], 1)) +
                "met in %s sp." % len(self._sp_target_met)
                # TODO: log this to output file
            )
        
        state = self._enrichObs()
        # flag it done when it reaches the # of iterations or end budget
        if self._stop_at_end_budget:
            done = self.budget < self._min_protection_cost
        elif self._stop_at_target_met:
            done = self.get_species_met_target() == self.bioDivGrid._n_species
            # this should turn-off target auto-increase
        else:
            done = self.bioDivGrid._counter >= self.iterations
        
        # raise target if baseline already met
        if self._dynamic_target:
            if len(self._sp_target_met) == self.bioDivGrid._n_species:
                self.min_pop_requirement = self._dynamic_target.update_target(
                    self.min_pop_requirement
                )
        
        info = self._getInfo()
        reward = None
        if self.runMode == RunMode.ORACLE:
            if not self.static_system:
                self.observe()  # not needed if no changes in disturbance, pop sizes etc
        return state, reward, done, info
    
    def get_sp_list_per_PU(self):
        res = get_quadrant_coord_species_clean(
            self.bioDivGrid.length,
            self.bioDivGrid._h,
            resolution=self.resolution,
            protection_matrix=self.bioDivGrid.protection_matrix,
            sp_threshold=self.sp_threshold_feature_extraction,
            error=self.observe_error,
            climate_layer=self.bioDivGrid._climate_layer,
            climate_disturbance=self.bioDivGrid._climate_as_disturbance,
            flattened=True,
            sp_quadrant_list_arg=None,
        )
        return res[1]
    
    def get_species_met_target(self):
        target_met = self.bioDivGrid.protectedIndPerSpecies() > self.min_pop_requirement
        return np.nonzero(target_met)[0]
    
    def drop_unaffordable_cells(self):
        i = np.where(self.protection_cost > self.budget)[0]
        self.bioDivGrid._h[:, i, :] *= 0
        self.grid_obj_most_recent.h[:, i, :] *= 0
        self.grid_obj_previous.h[:, i, :] *= 0
    
    def set_lastObs(self, lastObs):
        self.lastObs = lastObs
        if self._dynamic_target:
            self._sp_target_met = self.get_species_met_target()
    
    def set_budget(self, budget, relative_budget=True):
        if relative_budget:
            budget = budget * (np.min(self.protection_cost) * self.bioDivGrid._n_pus)
        self.budget = budget
        self._initialBudget = budget
    
    def set_stopping_mode(self, n_steps, stop_at_end_budget, stop_at_target_met):
        self.iterations = n_steps
        self._stop_at_end_budget = stop_at_end_budget
        self._stop_at_target_met = stop_at_target_met
    
    def set_sp_quadrant_list_only_once(self):
        self._init_sp_quadrant_list_only_once = True
    
    def set_runMode(self, runMode):
        self.runMode = runMode
    
    def set_conservation_target(self, protect_fraction):
        tmp = (
                self.bioDivGrid.individualsPerSpecies() * protect_fraction
        )
        self.min_pop_requirement = np.ceil(tmp)
        self.protect_fraction = protect_fraction
    
    def reset_w_seed(self, seed):
        self._h_seed = seed
        self.reset()

    def set_print_freq(self, f):
        self.print_freq = f