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
from ..algorithms.empirical_env_setup import *
from concurrent.futures import ProcessPoolExecutor
import multiprocessing.pool

import collections
np.set_printoptions(suppress=True, precision=3)

EmpRunnerInput = collections.namedtuple("EmpRunnerInput", ("env", "policy", "runner"))

def runEpisode(runnerInput):
    env = runnerInput.env
    policy = runnerInput.policy
    runner = runnerInput.runner
    runner.run_episode(env, policy)
    return env

class EvolutionPredictEmpirical:
    def __init__(
            self,
            state_adaptor=RichStateAdaptor(),
            outfile="output.log",
            rnd_policy=0,
            save_pkls=0,
            log_file_steps="",
            deterministic_policy=0,
            marxan_policy=0,
            plot_sim=0,
            plot_species=[],
            update_features=1,
            drop_unaffordable=True,
    ):
        self._state_adaptor = state_adaptor
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
        self._update_features = update_features
        self.drop_unaffordable = drop_unaffordable
    
    def select_action(self, state, info, policy, lastObs=None):
        adapted_state = self._state_adaptor.adapt(state, info)
        probs = policy.probs(adapted_state, lastObs=lastObs)
        # force remove already protected units
        probs += 1e-20
        probs[state["protection_matrix"][:, 0] == 1] = 0
        # force remove too expensive units
        if self.drop_unaffordable:
            probs[state["protection_cost"] > info["budget_left"]] = 0
            probs = probs / np.sum(probs)
        
        if self._deterministic_policy:
            action = np.argmax(probs)
        else:
            action = np.random.choice(policy.num_output, 1, p=probs)[0]
        # in this case cellList[action] == action (see getRichProtectAction)
        return Action(ActionType.Protect, action, action)
    
    def run_episode(self, env, policy):
        del self._rewards[:]
        state, ep_reward, done, info = env.reset()
        counter = 0
        while True:
            if counter % self._update_features == 0 and self._update_features > 1:
                # if == 1: lastObs is not used and full updates are done at each step
                # if env.print_freq < np.Inf:
                #     print("\nUpdating features...")
                updated_features = self.get_obsFeatures(env, policy)
                env.set_lastObs(updated_features)
            lastObs = env.lastObs
            action = self.select_action(state, info, policy, lastObs=lastObs)
            state, reward, done, info = env.step(action)
            counter += 1
            if done:
                break
    
    def get_obsFeatures(self, env, policy):
        env_tmp = copy.deepcopy(env)
        state, reward, done, info = env_tmp.step()
        state = self._state_adaptor.adapt(state, info)
        _, lastObs = policy.probs(state, return_lastObs=True)
        return lastObs

def launch_runner(env_list, policy, evolutionRunner, max_workers=None):
    if len(env_list) == 1:
        r = EmpRunnerInput(env_list[0], policy, evolutionRunner)
        runEpisode(r)
    else:
        runnerInputList = [EmpRunnerInput(env, policy, evolutionRunner) for env in env_list]
        if not max_workers:
            max_workers = len(runnerInputList)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            env_list = list(pool.map(runEpisode, runnerInputList))
    
    return env_list

def run_policy_empirical(
        env,
        trained_model,
        obsMode,
        observePolicy=2,
        n_nodes=[8, 0],
        budget=None,
        protection_target=0.1,
        n_steps=25,
        stop_at_end_budget=True,  # has priority over n_steps and stop_at_target_met
        stop_at_target_met=False,  # has priority over n_steps
        wd="",
        seed=0,
        verbose=1,
        update_features=10,
        result_file="out",
        init_sp_quadrant_list_only_once=False,
        temperature=1,
        deterministic_policy=0,
        replicates=1,
        max_workers=None
):
    # load systems
    env_list = []
    seed_list = []
    for rep in range(replicates):
        env_rep = copy.deepcopy(env)
        # update env settings based on run settings
        env_rep.set_budget(budget)
        env_rep.set_stopping_mode(n_steps, stop_at_end_budget, stop_at_target_met)
        runMode = [RunMode.NOUPDATEOBS, RunMode.ORACLE, RunMode.PROTECTATONCE][
            observePolicy
        ]
        env_rep.set_runMode(runMode)
        if init_sp_quadrant_list_only_once:
            # initialize sp list only once to save one for loop in
            # state_monitor.get_quadrant_coord_species_clean()
            env.set_sp_quadrant_list_only_once()
        # set conservation target
        env_rep.set_conservation_target(protection_target)
        if rep > 0:
            # print output only from 1st run
            env_rep.set_print_freq(np.Inf)
        
        # re-initialize species
        s = seed + rep
        env_rep.reset_w_seed(s)
        seed_list.append(s)
        env_list.append(env_rep)
    
    # load policy
    policy = load_policy_empirical(
        obsMode=obsMode,
        trained_model=trained_model,
        n_NN_nodes=n_nodes,
        temperature=temperature,
        num_output=env.bioDivGrid._n_pus,
    )
    
    evolutionRunner = EvolutionPredictEmpirical(update_features=update_features,
                                                deterministic_policy=deterministic_policy)
    
    env_list = launch_runner(env_list, policy, evolutionRunner, max_workers)
    out_files = []
    for i, env in enumerate(env_list):
        result_file_name = "%s_%s.pkl" % (result_file, seed_list[i])
        full_file_name = os.path.join(wd, result_file_name)
        SaveObject(env, full_file_name)
        print("Results saved as:", full_file_name)
        out_files.append(full_file_name)
    
    if rep == 1:
        env_list = env_list[0]
        out_files = out_files[0]
    return env_list, out_files
