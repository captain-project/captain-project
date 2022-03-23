import csv
import os
import sys
import pickle
import numpy as np
import collections
from ..biodivsim.BioDivEnv import *
from ..biodivsim.StateInitializer import PickleInitializer
from ..biodivsim.ClimateGenerator import get_climate

np.set_printoptions(suppress=1, precision=3)

# GLOBAL
# TODO: expose sp_threshold (used in BioDivEnv)
sp_threshold = 10


def SaveObject(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


EnvInput = collections.namedtuple(
    "EnvInput",
    (
        "budget",
        "gridInitializer",
        "n_cells",
        "n_species",
        "alpha",
        "K_max",
        "dispersal_rate",
        "distb_obj",
        "disturbance_sensitivity",
        "selectivedistb_obj",
        "selective_sensitivity",
        "climate_obj",
        "climate_sensitivity",
        "timeSteps",
        "runMode",
        "worker_id",
        "obsMode",
        "use_protection_cost",
        "random_training",
        "rnd_disturbance_init",
        "rewardMode",
        "list_species_values",
        "resolution",
        "climate_as_disturbance",
        "rnd_alpha_species",
        "disturbance_dep_dispersal",
        "max_fraction_protected",
        "edge_effect",
        "growth_rates",
        "start_protecting",
    ),
)

RunnerInput = collections.namedtuple("RunnerInput", ("env", "policy", "runner"))
EvolutionRunnerInput = collections.namedtuple(
    "EvolutionRunnerInput", ("env", "policy", "runner", "noise")
)


def init_sp_values(n_species, seed=1234, grid_size=50, use_BioDivEnv_geo_function=True):
    np.random.seed(seed)
    randval = np.random.choice([0.1, 10], n_species, p=[0.8, 0.2])
    if use_BioDivEnv_geo_function:
        # get coordinates with highest value
        max_value_coord = np.random.choice(range(grid_size), 2)
        return [max_value_coord, randval]
    return randval / np.sum(randval) * n_species


def init_sp_sensitivities(n_species, seed=1234):
    np.random.seed(seed)
    disturbance_sensitivity = np.zeros(n_species) + np.random.random(n_species)
    selective_sensitivity = np.random.beta(0.2, 0.7, n_species)
    climate_sensitivity = np.random.beta(2, 2, n_species)
    return disturbance_sensitivity, selective_sensitivity, climate_sensitivity


def buildEnv(runnerInput):
    print("runnerInput.runMode", runnerInput.runMode)
    env = BioDivEnv(
        runnerInput.budget,
        runnerInput.gridInitializer,
        runnerInput.n_cells,
        runnerInput.n_species,
        runnerInput.alpha,
        runnerInput.K_max,
        runnerInput.dispersal_rate,
        disturbanceGenerator=runnerInput.distb_obj,
        disturbance_sensitivity=runnerInput.disturbance_sensitivity,
        selectivedisturbanceInitializer=runnerInput.selectivedistb_obj,
        selective_sensitivity=runnerInput.selective_sensitivity,
        max_fraction_protected=runnerInput.max_fraction_protected,
        species_threshold=sp_threshold,
        rnd_alpha_species=runnerInput.rnd_alpha_species,
        climateModel=runnerInput.climate_obj,
        ignoreFirstObs=0,
        buffer_zone=runnerInput.edge_effect,
        iterations=runnerInput.timeSteps,
        resolution=runnerInput.resolution,
        numFeatures=10,
        runMode=runnerInput.runMode,
        worker_id=runnerInput.worker_id,
        observeMode=runnerInput.obsMode,
        use_protection_cost=runnerInput.use_protection_cost,
        rnd_sensitivities=runnerInput.random_training,
        rnd_disturbance_init=runnerInput.rnd_disturbance_init,
        list_species_values=runnerInput.list_species_values,
        rewardMode=runnerInput.rewardMode,
        climate_sensitivity=runnerInput.climate_sensitivity,
        climate_as_disturbance=runnerInput.climate_as_disturbance,
        disturbance_dep_dispersal=runnerInput.disturbance_dep_dispersal,
        growth_rate=runnerInput.growth_rates,
        start_protecting=runnerInput.start_protecting,
    )
    return env


def simulate_biodiv_env(
    sim_file,
    budget=0.1,
    dispersal_rate=0.1,
    alpha=0.01,
    time_steps=25,
    observePolicy=0,  # 0: NO-OBSERVE-UPDATE 1: ORACLE 2: PROTECTATONCE
    obsMode=0,  # 0: random, 1: full monitor, 2: citizen-science, 3: one-time, 4: value, 5: area
    use_protection_cost=1,
    random_sim=1,  # "0: fixed simulations; 1: random"
    disturbance_mode=4,
    rewardMode=0,  # 0 <- 'species loss' 1 <- 'value loss' 2 <- 'area loss' 3 <- 'PD loss (untested)
    rnd_alpha_species=0,
    resolution=np.array([5, 5]),
    climate_mode=3,
    disturbance_dep_dispersal=0,
    list_species_values=[],
    max_fraction_protected=1,
    edge_effect=1,
    growth_rates=None,
    start_protecting=0,
    seed=0,  # seed for disturbance (0: random)
):
    gridInitializer = PickleInitializer(sim_file)
    h = gridInitializer.getInitialState()
    n_cells = h.shape[1]
    n_species = h.shape[0]
    K_max = np.einsum("xyz -> yz", h)[0][0]
    if growth_rates == None:
        growth_rates = np.ones(n_species)

    distb_obj, selectivedistb_obj = get_disturbance(disturbance_mode, seed=seed)
    (
        disturbance_sensitivity,
        selective_sensitivity,
        climate_sensitivity,
    ) = init_sp_sensitivities(n_species)
    climate_obj, climate_as_disturbance = get_climate(climate_mode)
    #### build environment based on env_setup
    envInput = EnvInput(
        budget,
        gridInitializer,
        n_cells,
        n_species,
        alpha,
        K_max,
        dispersal_rate,
        distb_obj,
        disturbance_sensitivity,
        selectivedistb_obj,
        selective_sensitivity,
        climate_obj,
        climate_sensitivity,
        time_steps,
        observePolicy,
        0,
        obsMode,
        use_protection_cost,
        random_sim,
        disturbance_mode,
        rewardMode,
        list_species_values,
        resolution,
        climate_as_disturbance,
        rnd_alpha_species,
        disturbance_dep_dispersal,
        max_fraction_protected,
        edge_effect,
        growth_rates,
        start_protecting,
    )

    env = buildEnv(envInput)
    return env
