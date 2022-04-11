import os
import numpy as np
import pandas as pd
import pickle
from ..biodivsim.EmpiricalBioDivEnv import *
from ..biodivsim.EmpiricalGrid import *
from ..biodivsim.StateInitializer import PickleInitializer
from ..biodivsim.ClimateGenerator import get_climate

np.set_printoptions(suppress=1, precision=3)

# Required (initial) files
"""
Inputs/puvsp.dat <- species,pu,amount
Inputs/pu.dat <- cost per unit and status (in case already protected)
Planning_Units.txt <- unit ID and coordinates

"""
from captain import *
import numpy as np
import pandas as pd


def build_empirical_env(
    wd="",
    puvsp_file=None,
    pu_file=None,
    pu_info_file=None,
    # fast loading files
    hist_file=None,
    puid_file=None,
    spid_file=None,
    budget=1,
    protect_fraction=0.1,
    max_disturbance=0.95,
    observePolicy=2,
    seed=1234,
    species_sensitivities=None,
    hist_out_file=None,
    pu_id_out_file=None,
    sp_id_out_file=None
):

    emp = EmpiricalGrid(species_sensitivities=species_sensitivities)
    if wd != "":
        f_list = [puvsp_file, hist_file, puid_file, pu_info_file, pu_file, spid_file]
        for i in range(len(f_list)):
            try:
                f_list[i] = os.path.join(wd, f_list[i])
            except:
                pass
        [puvsp_file, hist_file, puid_file, pu_info_file, pu_file, spid_file] = f_list

    emp.initGrid(
        puvsp_file=puvsp_file,
        hist_file=hist_file,
        pu_id_file=puid_file,
        pu_info_file=pu_info_file,
        sp_id_file=spid_file,
        hist_out_file=hist_out_file,
        pu_id_out_file=pu_id_out_file,
        sp_id_out_file=sp_id_out_file
    )

    cost_tbl = pd.read_csv(pu_file)
    # subset cost table to PUs included in the species histogram
    cost_array = np.array(cost_tbl["cost"])[cost_tbl["id"].isin(emp._pus_id)] #KD!##np.array(cost_tbl["cost"])[emp._pus_id - 1]

    # add a minimum cost per unit and rescale
    if np.min(cost_array) == 0:
        if np.max(cost_array) == 0:
            min_cost = 1
        else:
            min_cost = 0.01 * np.min(
                cost_array[cost_array > 0]
            )  # 1% of the cheapest cell with a cost
        cost_array[cost_array == 0] = min_cost

    # rescale cost
    cost_array = cost_array / np.mean(cost_array)
    total_cost = np.sum(cost_array)
    # set a budget sufficient to protect 10% of cheapest PUs
    budget = budget * (np.min(cost_array) * emp._n_pus)
    disturbance_matrix = max_disturbance * cost_array / np.max(cost_array)
    emp.set_disturbance_matrix(disturbance_matrix)
    runMode = [RunMode.NOUPDATEOBS, RunMode.ORACLE, RunMode.PROTECTATONCE][
        observePolicy
    ]

    env = BioDivEnvEmpirical(
        emp,
        budget,
        runMode=runMode,
        cost_pu=cost_array,
        stop_at_end_budget=True,
        verbose=0,
        iterations=None,
        protect_fraction=protect_fraction,
        h_seed=seed,
    )
    return env


# LOAD POLICY
def load_policy_empirical(
    obsMode=1,
    trained_model=None,
    n_NN_nodes=[8, 0],
    temperature=1,
    observe_error=0,
    sp_threshold_feature_extraction=1,
    num_output=None,
):

    # load trained model
    head = next(open(trained_model)).split()
    loaded_ws = np.loadtxt(trained_model, skiprows=1)
    selected_epoch = -1
    loadedW = loaded_ws[selected_epoch]

    num_features = len(get_feature_indx(mode=obsMode))
    [
        num_output,
        num_meta_features,
        nodes_layer_1,
        nodes_layer_2,
        nodes_layer_3,
        _,
    ] = get_NN_model_prm(num_features, n_NN_nodes, num_output)
    coeff_meta_features = get_thresholds_reverse(loadedW[-num_meta_features:])
    ind = [head.index(s) for s in head if "coeff_" in s]
    coeff_features = loadedW[np.min(ind) :]  # remove first columns

    num_features = len(get_feature_indx(mode=obsMode))
    # model_prm = [coeff_features, coeff_meta_features]

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
        flattened=True,
        verbose=0,
    )

    return policy
