import csv
import os
import sys

# TODO fix imports
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
from ..biodivsim.BioDivEnv import BioDivEnv, Action, ActionType, RunMode
from ..agents.state_monitor import (
    extract_features,
    get_feature_indx,
    get_thresholds,
    get_thresholds_reverse,
)
import numpy as np

np.set_printoptions(suppress=True)  # prints floats, no scientific notation
np.set_printoptions(precision=3)  # rounds all array elements to 3rd digit
from ..biodivsim.StateInitializer import *
from .reinforce import RichProtectActionAdaptor, RichStateAdaptor
from ..agents.policy import PolicyNN, get_NN_model_prm
from .env_setup import *
from .marxan_setup import *
from ..plot.plot_env import plot_env_state, plot_env_state_generator
from ..biodivsim.BioDivEnv import *


class EVOLUTIONBoltzmannPredict:
    def __init__(
        self,
        state_adaptor,
        action_adaptor,
        outfile,
        rnd_policy=0,
        save_pkls=0,
        log_file_steps="",
        deterministic_policy=0,
        marxan_policy=0,
        plot_sim=0,
        plot_species=[],
    ):
        self._state_adaptor = state_adaptor
        self._action_adaptor = action_adaptor
        self._rewards = []
        self._outfile = outfile.split(".log")[0]
        self._count_rep = 0
        self._random_policy = rnd_policy
        self._marxan_policy = marxan_policy
        self._save_pkls = save_pkls
        self._log_file_steps = log_file_steps
        if self._log_file_steps != "":
            with open(self._log_file_steps, "w") as f:
                writer = csv.writer(f, delimiter="\t")
                l = [
                    "simulation",
                    "step",
                    "protected_cells",
                    "species",
                    "value",
                    "pd",
                    "disturbance",
                ]
                writer.writerow(l)
        self._deterministic_policy = deterministic_policy
        self._plot_sim = plot_sim
        self._plot_species = plot_species
        self._sim_count = 0

    def select_action(self, state, info, policy):
        state = self._state_adaptor.adapt(state, info)
        probs = policy.probs(state)
        if self._deterministic_policy:
            action = np.argmax(probs)
        else:
            action = np.random.choice(policy.num_output, 1, p=probs)
        return self._action_adaptor.adapt(action.item())

    def save_grid(self, env, step=0):
        # TODO: check this (fix imports or remove )
        filename = self._outfile + "_%s_step%s_.pkl" % (self._count_rep, step)
        print("Saving pickle file:", filename)
        SaveObject(env, filename)

    def log_steps(self, env, step=0):
        with open(self._log_file_steps, "a") as f:
            writer = csv.writer(f, delimiter="\t")
            info = env._getInfo()
            l = [
                self._sim_count,
                step,
                info["NumberOfProtectedCells"],
                info["ExtantSpecies"],
                info["ExtantSpeciesValue"],
                info["ExtantSpeciesPD"],
                np.mean(env.bioDivGrid.disturbance_matrix),
            ]
            writer.writerow(l)

    def get_marxan_res(self, env):
        marxan_actions = get_marxan_solution(env, policy_type=self._marxan_policy)
        return marxan_actions

    def _init_run_episode(self, env):
        self._sim_count += 1
        random_actions = None
        
        if self._random_policy == 1:
            random_actions = np.random.choice(
                range(len(env.quadrant_coords_list)),
                len(env.quadrant_coords_list),
                replace=False,
            )
            print(
                "random_actions:",
                random_actions[: np.min([10, len(random_actions)])],
                "...",
            )

        del self._rewards[:]

        state, ep_reward, done, info = env.reset(fullInfo=True)
        
        # if self._save_pkls:
        #     self.save_grid(env,0)

        #marxan_actions = np.array([0])
        t = 1
        p_count = 0

        time_series_stats = [
            [info["ExtantSpecies"], info["ExtantSpeciesValue"], info["ExtantSpeciesPD"]]
        ]

        return state, ep_reward, done, info, t, p_count, time_series_stats, random_actions
    
    def _get_action(self, env, policy, random_actions, p_count, state, info, t):
        if self._random_policy == 1:
            action = random_actions[np.min([p_count, len(random_actions) - 1])]
            return self._action_adaptor.adapt(action.item())
        
        elif self._marxan_policy:
            if t - 1 == env._start_protecting:
                marxan_actions = self.get_marxan_res(env)
                print("marxan_actions:", marxan_actions)
            action = marxan_actions[np.min([p_count, len(marxan_actions) - 1])]
            # print(t, p_count, env._start_protecting, action)
            return self._action_adaptor.adapt(action.item())

        else:
            return self.select_action(state, info, policy)

    def _step_episode(self, env, policy, random_actions, p_count, state, info, t, time_series_stats):
        action = self._get_action(env, policy, random_actions=random_actions, p_count=p_count, state=state, info=info, t=t)
            
        state, reward, done, info = env.step(action)

        time_series_stats.append(
            [
                info["ExtantSpecies"],
                info["ExtantSpeciesValue"],
                info["ExtantSpeciesPD"],
            ]
        )
        
        # state here is richObs in BioDivEnv which is lastObs.stats_quadrant + timeSince last observe
        self._rewards.append(reward)
        # ep_reward += reward # not used

        if self._log_file_steps != "":
            self.log_steps(env, t)

        if self._save_pkls:
            self.save_grid(env, step=t)

        if t > env._start_protecting:
            p_count += 1

        t += 1
        return state, done, p_count, t

    def run_episode(self, env, policy):
        state, ep_reward, done, info, t, p_count, time_series_stats, random_actions = self._init_run_episode(env)

        if self._save_pkls:
            self.save_grid(env, step=t)
        
        if self._plot_sim:
            self._plot_file = self._outfile + "_%s" % self._sim_count
            plot_env_state(
                env, outfile=self._plot_file, species_list=self._plot_species
            )

        while True:
            state, done, p_count, t = self._step_episode(env, policy, random_actions=random_actions, p_count=p_count, state=state, info=info, t=t, time_series_stats=time_series_stats)
            
            if self._plot_sim:
                plot_env_state(
                    env, outfile=self._plot_file, species_list=self._plot_species
                )
                
            if done:
                break

        self._count_rep += 1

        return info, self._rewards
    
    
    def run_episode_generator(self, env, policy, wd="."):
        state, ep_reward, done, info, t, p_count, time_series_stats, random_actions = self._init_run_episode(env)

        if self._save_pkls:
            self.save_grid(env, step=t)
        
        if self._plot_sim:
            self._plot_file = self._outfile + "_%s" % self._sim_count
            yield from plot_env_state_generator(
                env, outfile=self._plot_file, species_list=self._plot_species, wd=wd
            )

        while True:
            state, done, p_count, t = self._step_episode(env, policy, random_actions=random_actions, p_count=p_count, state=state, info=info, t=t, time_series_stats=time_series_stats)
            
            if self._plot_sim:
                yield from plot_env_state_generator(
                    env, outfile=self._plot_file, species_list=self._plot_species, wd=wd
                )
                
            if done:
                break

        self._count_rep += 1

        yield { "status": "done", "info": info, "rewards": self._rewards }


RunnerInput = collections.namedtuple("RunnerInput", ("env", "policy", "runner"))
EvolutionRunnerInput = collections.namedtuple(
    "EvolutionRunnerInput", ("env", "policy", "runner", "noise")
)


def _runOptimPolicyUpdateEnvWithFixedSimulations(
    epoch,
    env,
    n_species,
    n_cells,
    distb_obj,
    selectivedistb_obj,
    disturbance_mode=4,
    seed=0,
    climate_change=0,
    climate_obj=0,
    climate_model=1,
):
    # update with fixed seeds
    distb_obj, selectivedistb_obj = get_disturbance(disturbance_mode, seed + epoch)

    env.disturbanceGenerator = distb_obj
    env.selectivedisturbanceInitializer = selectivedistb_obj

    (
        disturbance_sensitivity,
        selective_sensitivity,
        climate_sensitivity,
    ) = init_sp_sensitivities(n_species, seed=seed + epoch)
    list_species_values = init_sp_values(
        n_species, seed=seed + epoch, grid_size=n_cells
    )
    env.list_species_values = list_species_values
    env.tree_generator = get_phylo_generator(n_species=env.n_species, seed=seed + epoch)
    env.disturbance_sensitivity = disturbance_sensitivity
    env.selective_sensitivity = selective_sensitivity

    # print('climate_obj',climate_obj)
    if climate_obj != 0:
        if climate_model == 1:
            from ..biodivsim.ClimateGenerator import (
                SimpleGradientClimateGenerator as ClimateGen,
            )

            climate_obj = ClimateGen(
                0, seed=seed + epoch, climate_change=climate_change
            )
        if climate_model == 2:
            from ..biodivsim.ClimateGenerator import (
                RegionalClimateGenerator as ClimateGen,
            )

            climate_obj = ClimateGen(0, seed=seed + epoch)
        if climate_model == 3:
            peak_anomaly = climate_obj._peak_anomaly
            from ..biodivsim.ClimateGenerator import (
                GradientClimateGeneratorRnd as ClimateGen,
            )

            climate_obj = ClimateGen(
                0,
                seed=seed + epoch,
                climate_change=climate_change,
                peak_anomaly=peak_anomaly,
            )
        env.climateModel = climate_obj
        # print(' env.climateModel = climate_obj',  env.climateModel , climate_model)


epoch_data_head = [
    "simulation",
    "reward",
    "protected_cells",
    "budget_left",
    "time_last_protect",
    "avg_cost",
    "extant_sp",
    "extant_sp_value",
    "extant_sp_pd",
    "species_loss",
    "value_loss",
    "pd_loss",
]

def _write_epoch_data(res, n_species, epoch, outfile):
    avg_reward = np.sum(res[1])
    avg_budget_left = res[0]["budget_left"]
    avg_time_last_protect = res[0]["time_last_protect"]
    avg_protected_cells = res[0]["NumberOfProtectedCells"]
    avg_cost = res[0]["CostOfProtection"]
    avg_extant_sp = res[0]["ExtantSpecies"]
    avg_extant_sp_value = res[0]["ExtantSpeciesValue"]
    avg_extant_sp_pd = res[0]["ExtantSpeciesPD"]
    species_loss = (1 - avg_extant_sp) * n_species
    relative_value_loss = (1 - avg_extant_sp_value) * 100
    relative_pd_loss = (1 - avg_extant_sp_pd) * 100

    epoch_data = [
        epoch,
        avg_reward,
        avg_protected_cells,
        avg_budget_left,
        avg_time_last_protect,
        avg_cost,
        avg_extant_sp,
        avg_extant_sp_value,
        avg_extant_sp_pd,
        species_loss,
        relative_value_loss,
        relative_pd_loss,
    ]

    with open(outfile, "a") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(epoch_data)
    
    return epoch_data

def _runOptimPolicyEpoch(
    epoch,
    env,
    n_species,
    n_cells,
    evolutionRunner,
    policy,
    distb_obj,
    selectivedistb_obj,
    outfile="",
    disturbance_mode=4,
    seed=0,
    random_sim=1,
    climate_change=0,
    climate_obj=0,
    climate_model=1,
):
    if random_sim == 0:
        _runOptimPolicyUpdateEnvWithFixedSimulations(
            epoch=epoch,
            env=env,
            n_species=n_species,
            n_cells=n_cells,
            distb_obj=distb_obj,
            selectivedistb_obj=selectivedistb_obj,
            disturbance_mode=disturbance_mode,
            seed=seed,
            climate_change=climate_change,
            climate_obj=climate_obj,
            climate_model=climate_model,
        )

    print("=======================================")
    print(f"running simulation: {epoch}")
    print("=======================================")

    res = evolutionRunner.run_episode(env, policy)

    print("=======================================")
    print(f"epoch {epoch} summary")
    print("=======================================")
    print(f"policy coeff: {policy.coeff}")
    print(f"avg reward: {np.sum(res[1])}")
    print("budget left", res[0]["budget_left"])
    print("time last protect", res[0]["time_last_protect"])
    print("n. protected cells", res[0]["NumberOfProtectedCells"])
    print("selected units:", env.protected_quadrants)
    
    epoch_data = _write_epoch_data(res, n_species, epoch, outfile)

    return dict(zip(epoch_data_head, epoch_data))

def _runOptimPolicyEpoch_generator(
    epoch,
    env,
    n_species,
    n_cells,
    evolutionRunner,
    policy,
    distb_obj,
    selectivedistb_obj,
    outfile="",
    disturbance_mode=4,
    seed=0,
    random_sim=1,
    climate_change=0,
    climate_obj=0,
    climate_model=1,
    plot_dir=".",
):
    if random_sim == 0:
        _runOptimPolicyUpdateEnvWithFixedSimulations(
            epoch=epoch,
            env=env,
            n_species=n_species,
            n_cells=n_cells,
            distb_obj=distb_obj,
            selectivedistb_obj=selectivedistb_obj,
            disturbance_mode=disturbance_mode,
            seed=seed,
            climate_change=climate_change,
            climate_obj=climate_obj,
            climate_model=climate_model,
        )

    print("=======================================")
    print(f"running simulation: {epoch}")
    print("=======================================")

    res = None
    for progress in evolutionRunner.run_episode_generator(env=env, policy=policy, wd=plot_dir):
        if "status" in progress and progress["status"] == "done":
            res = [progress["info"], progress["rewards"]]
        else:
            yield progress

    print("=======================================")
    print(f"epoch {epoch} summary")
    print("=======================================")
    print(f"policy coeff: {policy.coeff}")
    print(f"avg reward: {np.sum(res[1])}")
    print("budget left", res[0]["budget_left"])
    print("time last protect", res[0]["time_last_protect"])
    print("n. protected cells", res[0]["NumberOfProtectedCells"])
    print("selected units:", env.protected_quadrants)
    
    epoch_data = _write_epoch_data(res, n_species, epoch, outfile)

    data = dict(zip(epoch_data_head, epoch_data))
    yield { "type": "simulation", "status": "finished", "data": data }


def _runOptimPolicySetup(
    time_steps,
    budget,
    outfile="",
    disturbance_mode=4,
    seed=0,
    obsMode=1,
    runMode=RunMode.ORACLE,
    observe_error=0,
    use_protection_cost=0,
    wNN=None,
    n_NN_nodes=[4, 0],
    rewardMode=0,
    random_sim=1,
    pickle_num=0,
    save_pkls=0,
    resolution=np.array([1, 1]),
    climate_obj=0,
    dispersal_rate=0.1,
    climate_as_disturbance=0,
    rnd_alpha_species=0,
    disturbance_dep_dispersal=0,
    log_all_steps=0,
    max_fraction_protected=1,
    temperature=100,
    growth_rates=[0.1],
    edge_effect=1,
    deterministic_policy=1,
    marxan_policy=0,
    wd="",
    sim_file=None, # Specified sim file overriding wd
    sp_threshold_feature_extraction=1,
    start_protecting=3,
    plot_sim=0,
    plot_species=[],
):

    RESOLUTION = resolution
    if sim_file is not None:
        gridInitializer = PickleInitializer(sim_file)
        rnd_disturbance_init = disturbance_mode # TODO: This or -1?
    elif random_sim == 1:
        gridInitializer = RandomPickleInitializer(pklfolder=wd, verbose=True)
        rnd_disturbance_init = disturbance_mode
    elif random_sim == 0:
        # gridInitializer = PickleInitializerBatch(pklfolder=wd, verbose=True, pklfile_i=0)
        gridInitializer = PickleInitializerSequential(
            pklfolder=wd, verbose=True, pklfile_i=pickle_num
        )
        if disturbance_mode not in [3, 4, 5, 0, -1]:
            sys.exit(
                "Fixed seed not available on disturbance modes not in [3,4,5,0,-1]"
            )
        rnd_disturbance_init = -1
    elif random_sim == 2:
        gridInitializer = PickleInitializerSequential(pklfolder=wd, verbose=True)
        rnd_disturbance_init = -1
    init_data = gridInitializer.getInitialState(1, 1, 1)
    n_cells = init_data.shape[1]
    n_species = init_data.shape[0]
    alpha = 0.01
    K_max = np.einsum("xyz -> yz", init_data)[0][0]

    grid_size = n_cells
    OUTPUT = (grid_size ** 2) / (RESOLUTION[0] * RESOLUTION[1])
    if OUTPUT % np.int(OUTPUT) != 0:
        sys.exit("\n\nResolution not allowed!\n\n")
    else:
        OUTPUT = np.int(OUTPUT)
        print("Number of protection units: ", OUTPUT)

    distb_obj, selectivedistb_obj = get_disturbance(disturbance_mode, seed)
    disturbance_sensitivity = np.zeros(n_species) + np.random.random(n_species)
    selective_sensitivity = np.random.beta(0.2, 0.7, n_species)
    climate_sensitivity = np.random.beta(2, 2, n_species)
    if random_sim:
        list_species_values = []
    else:
        (
            disturbance_sensitivity,
            selective_sensitivity,
            climate_sensitivity,
        ) = init_sp_sensitivities(n_species, seed=seed)
        list_species_values = init_sp_values(n_species, seed=seed, grid_size=n_cells)

    # species loss is calculated in BioDivEnv: `reward = self.bioDivGrid.numberOfSpecies() - self.n_extant`
    if obsMode == 0:
        # the only feature is "check already protected", i.e. RANDOM POLICY
        rnd_policy = 1
    elif obsMode == 6:
        rnd_policy = 2
        # check deltaVC feature, i.e. HEURISTIC POLICY
    else:
        rnd_policy = 0
        if wNN is None:
            sys.exit("Please provide trained model or use obsMode [0,6]")
    num_features = len(get_feature_indx(mode=obsMode))
    [
        num_output,
        num_meta_features,
        nodes_layer_1,
        nodes_layer_2,
        nodes_layer_3,
        n_prms,
    ] = get_NN_model_prm(num_features, n_NN_nodes, OUTPUT)

    if wNN is None:
        coeff_features = np.ones(n_prms)
        coeff_meta_features = np.zeros(num_meta_features)
        if rnd_policy == 1:
            print("Running with random policy")
        else:
            print("Running with heuristic policy")
    else:
        coeff_features = wNN[:-num_meta_features]
        coeff_meta_features = wNN[-num_meta_features:]

    policy = PolicyNN(
        num_features,
        num_meta_features,
        num_output,
        coeff_features,
        coeff_meta_features,
        temperature=temperature,
        mode=obsMode,
        observe_error=observe_error,
        nodes_l1=nodes_layer_1,
        nodes_l2=nodes_layer_2,
        nodes_l3=nodes_layer_3,
        sp_threshold=sp_threshold_feature_extraction,
    )

    state_adaptor = RichStateAdaptor()
    action_adaptor = RichProtectActionAdaptor(grid_size, RESOLUTION)
    # init out file
    with open(outfile, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        epoch_data_head
        writer.writerow(epoch_data_head)
    # TODO end refactor
    if log_all_steps:
        log_file_steps = outfile + "_steps.log"
    else:
        log_file_steps = ""
    evolutionRunner = EVOLUTIONBoltzmannPredict(
        state_adaptor,
        action_adaptor,
        outfile,
        rnd_policy=rnd_policy,
        marxan_policy=marxan_policy,
        save_pkls=save_pkls,
        log_file_steps=log_file_steps,
        deterministic_policy=deterministic_policy,
        plot_sim=plot_sim,
        plot_species=plot_species,
    )

    # 'budget', 'gridInitializer', 'n_cells', 'n_species', 'alpha', 'K_max', 'dispersal_rate',
    # 'distb_obj', 'disturbance_sensitivity', 'selectivedistb_obj', 'selective_sensitivity', 'climate_obj', 'timeSteps',
    # 'runMode', 'worker_id', 'obsMode', 'use_protection_cost', 'random_training',
    # 'rnd_disturbance_init', 'rewardMode', 'list_species_values'

    # if random_sim == 1:
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
        runMode,
        0,
        obsMode,
        use_protection_cost,
        random_sim,
        rnd_disturbance_init,
        rewardMode,
        list_species_values,
        RESOLUTION,
        climate_as_disturbance,
        rnd_alpha_species,
        disturbance_dep_dispersal,
        max_fraction_protected,
        edge_effect,
        growth_rates,
        start_protecting,
    )
    # elif random_sim == 2:
    #     envInput = EnvInput(budget, gridInitializer, n_cells, n_species, alpha, K_max, dispersal_rate,
    #                         distb_obj, disturbance_sensitivity, selectivedistb_obj, selective_sensitivity,
    #                         climate_obj, climate_sensitivity, time_steps, runMode, 0,
    #                         obsMode, use_protection_cost, random_sim,
    #                         rnd_disturbance_init,
    #                         rewardMode, list_species_values, RESOLUTION,
    #                         climate_as_disturbance, rnd_alpha_species, disturbance_dep_dispersal,
    #                         max_fraction_protected, edge_effect, growth_rates,
    #                         start_protecting)
    # else: # random_sim == 0
    #
    #
    #     envInput = EnvInput(budget, gridInitializer, n_cells, n_species, alpha, K_max, dispersal_rate,
    #                         distb_obj, disturbance_sensitivity, selectivedistb_obj, selective_sensitivity,
    #                         climate_obj, climate_sensitivity, time_steps, runMode, 0,
    #                         obsMode, use_protection_cost, random_sim,
    #                         rnd_disturbance_init,  # BioDivEnv.rnd_disturbance_init = -1 : fixed disturbance
    #                         rewardMode, list_species_values, RESOLUTION,
    #                         climate_as_disturbance, rnd_alpha_species, disturbance_dep_dispersal,
    #                         max_fraction_protected, edge_effect, growth_rates,
    #                         start_protecting)

    return (
        envInput,
        n_species,
        n_cells,
        evolutionRunner,
        policy,
        distb_obj,
        selectivedistb_obj,
    )


def runOptimPolicy(
    simulations,
    time_steps,
    budget,
    outfile="",
    disturbance_mode=4,
    seed=0,
    obsMode=1,
    runMode=RunMode.ORACLE,
    observe_error=0,
    use_protection_cost=0,
    wNN=None,
    n_NN_nodes=[4, 0],
    rewardMode=0,
    random_sim=1,
    pickle_num=0,
    save_pkls=0,
    save_GIF=False,
    resolution=np.array([1, 1]),
    climate_change=0,
    climate_obj=0,
    climate_model=1,
    dispersal_rate=0.1,
    climate_as_disturbance=0,
    rnd_alpha_species=0,
    disturbance_dep_dispersal=0,
    log_all_steps=0,
    max_fraction_protected=1,
    temperature=100,
    growth_rates=[0.1],
    edge_effect=1,
    deterministic_policy=1,
    marxan_policy=0,
    wd="",
    sim_file=None,
    sp_threshold_feature_extraction=1,
    start_protecting=3,
    plot_sim=0,
    plot_species=[],
):
    (
        envInput,
        n_species,
        n_cells,
        evolutionRunner,
        policy,
        distb_obj,
        selectivedistb_obj,
    ) = _runOptimPolicySetup(
        time_steps=time_steps,
        budget=budget,
        outfile=outfile,
        disturbance_mode=disturbance_mode,
        seed=seed,
        obsMode=obsMode,
        runMode=runMode,
        observe_error=observe_error,
        use_protection_cost=use_protection_cost,
        wNN=wNN,
        n_NN_nodes=n_NN_nodes,
        rewardMode=rewardMode,
        random_sim=random_sim,
        pickle_num=pickle_num,
        save_pkls=save_pkls,
        resolution=resolution,
        climate_obj=climate_obj,
        dispersal_rate=dispersal_rate,
        climate_as_disturbance=climate_as_disturbance,
        rnd_alpha_species=rnd_alpha_species,
        disturbance_dep_dispersal=disturbance_dep_dispersal,
        log_all_steps=log_all_steps,
        max_fraction_protected=max_fraction_protected,
        temperature=temperature,
        growth_rates=growth_rates,
        edge_effect=edge_effect,
        deterministic_policy=deterministic_policy,
        marxan_policy=marxan_policy,
        wd=wd,
        sim_file=sim_file,
        sp_threshold_feature_extraction=sp_threshold_feature_extraction,
        start_protecting=start_protecting,
        plot_sim=plot_sim,
        plot_species=plot_species,
    )

    print("=======================================")
    print("setup done! Running simulations...")
    print("=======================================")
    env = buildEnv(envInput)
    for epoch in range(simulations):
        _runOptimPolicyEpoch(
            epoch=epoch,
            env=env,
            n_species=n_species,
            n_cells=n_cells,
            evolutionRunner=evolutionRunner,
            policy=policy,
            distb_obj=distb_obj,
            selectivedistb_obj=selectivedistb_obj,
            outfile=outfile,
            disturbance_mode=disturbance_mode,
            seed=seed,
            random_sim=random_sim,
            climate_change=climate_change,
            climate_obj=climate_obj,
            climate_model=climate_model,
        )

    if save_pkls == 2 and save_GIF:
        # TODO: fix or remove this
        print("Functionality not implemented.")
        # print("Creating GIF file...")
        # from scripts.plot_SimpleGrid_stats import get_GIF_animation
        # get_GIF_animation(os.path.dirname(outfile), out_tag=os.path.basename(outfile).split(".log")[0])
    return env


def runOptimPolicy_generator(
    simulations,
    time_steps,
    budget,
    outfile="",
    disturbance_mode=4,
    seed=0,
    obsMode=1,
    runMode=RunMode.ORACLE,
    observe_error=0,
    use_protection_cost=0,
    wNN=None,
    n_NN_nodes=[4, 0],
    rewardMode=0,
    random_sim=1,
    pickle_num=0,
    save_pkls=0,
    save_GIF=False,
    resolution=np.array([1, 1]),
    climate_change=0,
    climate_obj=0,
    climate_model=1,
    dispersal_rate=0.1,
    climate_as_disturbance=0,
    rnd_alpha_species=0,
    disturbance_dep_dispersal=0,
    log_all_steps=0,
    max_fraction_protected=1,
    temperature=100,
    growth_rates=[0.1],
    edge_effect=1,
    deterministic_policy=1,
    marxan_policy=0,
    wd="",
    sim_file=None,
    sp_threshold_feature_extraction=1,
    start_protecting=3,
    plot_sim=0,
    plot_species=[],
    plot_dir=".",
):
    (
        envInput,
        n_species,
        n_cells,
        evolutionRunner,
        policy,
        distb_obj,
        selectivedistb_obj,
    ) = _runOptimPolicySetup(
        time_steps=time_steps,
        budget=budget,
        outfile=outfile,
        disturbance_mode=disturbance_mode,
        seed=seed,
        obsMode=obsMode,
        runMode=runMode,
        observe_error=observe_error,
        use_protection_cost=use_protection_cost,
        wNN=wNN,
        n_NN_nodes=n_NN_nodes,
        rewardMode=rewardMode,
        random_sim=random_sim,
        pickle_num=pickle_num,
        save_pkls=save_pkls,
        resolution=resolution,
        climate_obj=climate_obj,
        dispersal_rate=dispersal_rate,
        climate_as_disturbance=climate_as_disturbance,
        rnd_alpha_species=rnd_alpha_species,
        disturbance_dep_dispersal=disturbance_dep_dispersal,
        log_all_steps=log_all_steps,
        max_fraction_protected=max_fraction_protected,
        temperature=temperature,
        growth_rates=growth_rates,
        edge_effect=edge_effect,
        deterministic_policy=deterministic_policy,
        marxan_policy=marxan_policy,
        wd=wd,
        sim_file=sim_file,
        sp_threshold_feature_extraction=sp_threshold_feature_extraction,
        start_protecting=start_protecting,
        plot_sim=plot_sim,
        plot_species=plot_species,
    )

    print("=======================================")
    print("setup done! Running simulations...")
    print("=======================================")
    env = buildEnv(envInput)

    simulations = 1
    for epoch in range(simulations):
        yield from _runOptimPolicyEpoch_generator(
            epoch=epoch,
            env=env,
            n_species=n_species,
            n_cells=n_cells,
            evolutionRunner=evolutionRunner,
            policy=policy,
            distb_obj=distb_obj,
            selectivedistb_obj=selectivedistb_obj,
            outfile=outfile,
            disturbance_mode=disturbance_mode,
            seed=seed,
            random_sim=random_sim,
            climate_change=climate_change,
            climate_obj=climate_obj,
            climate_model=climate_model,
            plot_dir=plot_dir,
        )


def _run_policy_init(
    obsMode=0,  # 0: random, 1: full monitor, 2: citizen-science, 3: one-time, 4: value, 5: area
    observePolicy=1,  #  0: NO-OBSERVE-UPDATE 1: ORACLE 2: PROTECTATONCE
    n_nodes=[4, 0],
    use_climate=3,  # "0: no climate change, 1: climate change, 2: climate disturbance,
    # 3: climate change + random variation"
    climate_change_magnitude=0.05,
    peak_anomaly=2,
    trained_model=None,  # if None: random policy
    # 1: deterministic policy (overrides temperature)
    burnin=0,  # skip the first n. epochs (generally not needed)
    load_best_epoch=0,  # 0: load last epoch; 1: load best epoch (post burnin)
):

    runMode = [RunMode.NOUPDATEOBS, RunMode.ORACLE, RunMode.PROTECTATONCE][
        observePolicy
    ]
    climate_disturbance = 0
    if use_climate == 1:
        climate_change = climate_change_magnitude
        from ..biodivsim.ClimateGenerator import (
            SimpleGradientClimateGenerator as ClimateGen,
        )

        CLIMATE_OBJ = ClimateGen(0, climate_change=climate_change)
    elif use_climate == 2:
        climate_disturbance = 1
        from ..biodivsim.ClimateGenerator import RegionalClimateGenerator as ClimateGen

        CLIMATE_OBJ = ClimateGen(0)
    elif use_climate == 3:  # global warming + random variation
        climate_change = climate_change_magnitude
        PEAK_ANOMALY = peak_anomaly
        from ..biodivsim.ClimateGenerator import (
            GradientClimateGeneratorRnd as ClimateGen,
        )

        CLIMATE_OBJ = ClimateGen(
            0, climate_change=climate_change, peak_anomaly=PEAK_ANOMALY
        )
    else:
        CLIMATE_OBJ = 0

    if trained_model is not None and trained_model != "heuristic":
        head = next(open(trained_model)).split()
        loaded_ws = np.loadtxt(trained_model, skiprows=np.max([1, burnin]))
        if load_best_epoch:
            selected_epoch = np.argmax(loaded_ws[:, head.index("reward")])
        else:
            selected_epoch = -1
        print(
            "Selected epoch",
            selected_epoch,
            loaded_ws[:, head.index("reward")][selected_epoch],
        )
        loadedW = loaded_ws[selected_epoch]

        num_features = len(get_feature_indx(mode=obsMode))
        [_, num_meta_features, _, _, _, _] = get_NN_model_prm(
            num_features, n_nodes, None
        )
        loadedW[-num_meta_features:] = get_thresholds_reverse(
            loadedW[-num_meta_features:]
        )
        ind = [head.index(s) for s in head if "coeff_" in s]
        wNN = loadedW[
            np.min(ind) :
        ]  # remove first 4 columns (reward, protected_cells, budget_left, time_last_protect, running_reward)
        running_reward_start = loadedW[head.index("running_reward")]

        print(wNN)
        print(running_reward_start)

    else:
        wNN = None
        if trained_model is None:
            n_nodes = [1, 0]
            obsMode = 0

    # TODO: Why call this?
    get_feature_indx(obsMode, print_obs_mode=True)

    return runMode, wNN, CLIMATE_OBJ, climate_disturbance


def run_policy(
    rnd_seed=1234,
    # TODO: fix obsMode options
    obsMode=0,  # 0: random, 1: full monitor, 2: citizen-science, 3: one-time, 4: value, 5: area
    steps=10,
    simulations=100,
    observePolicy=1,  #  0: NO-OBSERVE-UPDATE 1: ORACLE 2: PROTECTATONCE
    disturbance=4,
    edge_effect=1,
    protection_cost=1,
    n_nodes=[4, 0],
    random_sim=0,  # "0: fixed (replicable) simulations; 1: random; 2: fixed training, seq pickle"
    rewardMode=0,  # "0: species loss; 1: sp value; 2: protected area"; 3: PD loss (not yet tested)
    obs_error=0,  # "Amount of error in species counts (feature extraction)"
    resolution=np.array([5, 5]),
    grid_size=50,
    budget=0.11,
    max_fraction_protected=1,
    dispersal_rate=0.1,  # TODO: check if this can also be a vector per species
    growth_rates=[0.1],  # can be 1 values (list of 1 item) or or one value per species
    use_climate=3,  # "0: no climate change, 1: climate change, 2: climate disturbance,
    # 3: climate change + random variation"
    climate_change_magnitude=0.05,
    peak_anomaly=2,
    rnd_alpha=0,  # (st.dev of sp.-specific fluctuation in mortality (if 'by_species' ==1 in SimpleGrid)
    dist_dependent_dispersal=0,
    outfile="policy_output.log",
    log_all_steps=True,
    save_pkls=False,  # 0) no, 1) save pickle file at each step
    # model settings
    marxan_policy=0,  # if 1: run Marxan
    trained_model=None,  # if None: random policy
    temperature=100,
    deterministic_policy=1,  # 0: random policy (altered by temperature);
    # 1: deterministic policy (overrides temperature)
    wd="data_dependencies/pickles",
    burnin=0,  # skip the first n. epochs (generally not needed)
    load_best_epoch=0,  # 0: load last epoch; 1: load best epoch (post burnin)
    sp_threshold_feature_extraction=1,
    start_protecting=3,
    plot_sim=False,
    plot_species=[],
):
    runMode, wNN, CLIMATE_OBJ, climate_disturbance = _run_policy_init(
        obsMode=obsMode,
        observePolicy=observePolicy,
        n_nodes=n_nodes,
        use_climate=use_climate,
        # 3: climate change + random variation"
        climate_change_magnitude=climate_change_magnitude,
        peak_anomaly=peak_anomaly,
        trained_model=trained_model,
        # 1: deterministic policy (overrides temperature)
        burnin=burnin,
        load_best_epoch=load_best_epoch,
    )

    return runOptimPolicy(
        simulations=simulations,
        time_steps=steps,
        budget=budget,
        temperature=temperature,
        outfile=outfile,
        log_all_steps=log_all_steps,
        save_pkls=save_pkls,
        disturbance_mode=disturbance,
        seed=rnd_seed,
        obsMode=obsMode,
        runMode=runMode,
        observe_error=obs_error,
        use_protection_cost=protection_cost,
        rewardMode=rewardMode,
        wNN=wNN,
        n_NN_nodes=n_nodes,
        random_sim=random_sim,
        resolution=resolution,
        dispersal_rate=dispersal_rate,
        climate_obj=CLIMATE_OBJ,
        climate_as_disturbance=climate_disturbance,
        climate_model=use_climate,
        rnd_alpha_species=rnd_alpha,
        disturbance_dep_dispersal=dist_dependent_dispersal,
        max_fraction_protected=max_fraction_protected,
        growth_rates=growth_rates,
        edge_effect=edge_effect,
        deterministic_policy=deterministic_policy,
        marxan_policy=marxan_policy,
        wd=wd,
        sp_threshold_feature_extraction=sp_threshold_feature_extraction,
        start_protecting=start_protecting,
        plot_sim=plot_sim,
        plot_species=plot_species,
    )


def run_policy_generator(
    rnd_seed=1234,
    # TODO: fix obsMode options
    obsMode=0,  # 0: random, 1: full monitor, 2: citizen-science, 3: one-time, 4: value, 5: area
    time_steps=10,
    simulations=100,
    observePolicy=1,  #  0: NO-OBSERVE-UPDATE 1: ORACLE 2: PROTECTATONCE
    disturbance=4,
    edge_effect=1,
    protection_cost=1,
    n_nodes=[4, 0],
    random_sim=0,  # "0: fixed (replicable) simulations; 1: random; 2: fixed training, seq pickle"
    rewardMode=0,  # "0: species loss; 1: sp value; 2: protected area"; 3: PD loss (not yet tested)
    obs_error=0,  # "Amount of error in species counts (feature extraction)"
    resolution=np.array([5, 5]),
    grid_size=50,
    budget=0.11,
    max_fraction_protected=1,
    dispersal_rate=0.1,  # TODO: check if this can also be a vector per species
    growth_rates=[0.1],  # can be 1 values (list of 1 item) or or one value per species
    use_climate=3,  # "0: no climate change, 1: climate change, 2: climate disturbance,
    # 3: climate change + random variation"
    climate_change_magnitude=0.05,
    peak_anomaly=2,
    rnd_alpha=0,  # (st.dev of sp.-specific fluctuation in mortality (if 'by_species' ==1 in SimpleGrid)
    dist_dependent_dispersal=0,
    outfile="policy_output.log",
    log_all_steps=True,
    save_pkls=False,  # 0) no, 1) save pickle file at each step
    # model settings
    marxan_policy=0,  # if 1: run Marxan
    trained_model=None,  # if None: random policy
    temperature=100,
    deterministic_policy=1,  # 0: random policy (altered by temperature);
    # 1: deterministic policy (overrides temperature)
    wd="data_dependencies/pickles",
    sim_file=None,
    burnin=0,  # skip the first n. epochs (generally not needed)
    load_best_epoch=0,  # 0: load last epoch; 1: load best epoch (post burnin)
    sp_threshold_feature_extraction=1,
    start_protecting=3,
    plot_sim=False,
    plot_species=[],
    plot_dir=".",
):
    runMode, wNN, CLIMATE_OBJ, climate_disturbance = _run_policy_init(
        obsMode=obsMode,
        observePolicy=observePolicy,
        n_nodes=n_nodes,
        use_climate=use_climate,
        # 3: climate change + random variation"
        climate_change_magnitude=climate_change_magnitude,
        peak_anomaly=peak_anomaly,
        trained_model=trained_model,
        # 1: deterministic policy (overrides temperature)
        burnin=burnin,
        load_best_epoch=load_best_epoch,
    )

    yield from runOptimPolicy_generator(
        simulations=simulations,
        time_steps=time_steps,
        budget=budget,
        temperature=temperature,
        outfile=outfile,
        log_all_steps=log_all_steps,
        save_pkls=save_pkls,
        disturbance_mode=disturbance,
        seed=rnd_seed,
        obsMode=obsMode,
        runMode=runMode,
        observe_error=obs_error,
        use_protection_cost=protection_cost,
        rewardMode=rewardMode,
        wNN=wNN,
        n_NN_nodes=n_nodes,
        random_sim=random_sim,
        resolution=resolution,
        dispersal_rate=dispersal_rate,
        climate_obj=CLIMATE_OBJ,
        climate_as_disturbance=climate_disturbance,
        climate_model=use_climate,
        rnd_alpha_species=rnd_alpha,
        disturbance_dep_dispersal=dist_dependent_dispersal,
        max_fraction_protected=max_fraction_protected,
        growth_rates=growth_rates,
        edge_effect=edge_effect,
        deterministic_policy=deterministic_policy,
        marxan_policy=marxan_policy,
        wd=wd,
        sim_file=sim_file,
        sp_threshold_feature_extraction=sp_threshold_feature_extraction,
        start_protecting=start_protecting,
        plot_sim=plot_sim,
        plot_species=plot_species,
        plot_dir=plot_dir,
    )
