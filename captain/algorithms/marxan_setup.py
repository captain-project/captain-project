import os, sys
import pickle
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=1)  # prints floats, no scientific notation
np.set_printoptions(precision=3)  # rounds all array elements to 3rd digit
np.random.seed(123)
plt.style.use("ggplot")
sys.path.insert(0, r"/Users/dsilvestro/Software/captain-dev/")
import pandas as pd

from ..agents.state_monitor import get_quadrant_coord_species_clean

# read pickle file
out_wd = "./captain/marxan"


def get_marxan_solution(BioDivGrid, policy_type=1, return_best=False):
    evolveGrid = BioDivGrid.bioDivGrid
    resolution = BioDivGrid.resolution

    (
        quadrant_coords_list,
        sp_quadrant_list,
        protected_list,
        protected_species_list,
        _,
        total_pop_size,
    ) = get_quadrant_coord_species_clean(
        evolveGrid.length,
        evolveGrid.h,
        resolution=resolution,
        protection_matrix=evolveGrid._protection_matrix,
        sp_threshold=1,
        error=0,
        pop_size_per_unit=True,
    )

    # species presence/absence per protection unit
    # in all analyses we used a sp. threshold = 10, i.e. at least 10 individuals for the species to be extant
    n_PUs = len(sp_quadrant_list)

    # MARXAN FILES

    # puvspr.dat
    "species,pu,amount"
    tbl = list()
    for i in range(n_PUs):
        for s in range(len(sp_quadrant_list[i])):
            local_pop_size = np.round(sp_quadrant_list[i][s]).astype(int)
            if local_pop_size > 1:
                tbl.append([s + 1, i + 1, local_pop_size])

    tbl = pd.DataFrame(tbl, columns=["species", "pu", "amount"])
    try:
        tbl.to_csv(os.path.join(out_wd, "input/puvspr.dat"), sep=",", index=False)
    except:
        print("could not write file:", os.path.join(out_wd, "input/puvspr.dat"))

    # pu.dat
    "id,cost,status"
    cost = BioDivGrid._baseline_cost
    additional_cost = np.array(BioDivGrid.getProtectCostQuadrant())
    if len(additional_cost):
        cost += additional_cost
    # status == 2: locked-in -> must be protected
    # status == 3: locked-out -> can't be protected
    status = np.array(protected_list)  # if already protected:
    status[status > 0] = 2
    tbl = np.ones((n_PUs, 3))  # .astype(int)
    tbl[:, 0] = range(1, n_PUs + 1)
    tbl[:, 1] = cost
    tbl[:, 2] = status
    tbl = pd.DataFrame(tbl, columns=["id", "cost", "status"])
    try:
        tbl.to_csv(os.path.join(out_wd, "input/pu.dat"), sep=",", index=False)
    except:
        print("could not write file:", os.path.join(out_wd, "input/pu.dat"))
    # spec.dat
    "id,target,spf"  # sp ID, min individuals, sp penalty factor
    sp_threshold = BioDivGrid.species_threshold
    spf = 10
    tbl = np.ones((BioDivGrid.bioDivGrid._n_species, 3)).astype(int)
    tbl[:, 0] = range(1, 1 + BioDivGrid.bioDivGrid._n_species)
    tbl[:, 1] *= sp_threshold
    tbl[:, 2] *= spf

    tbl = pd.DataFrame(tbl, columns=["id", "target", "spf"])
    try:
        tbl.to_csv(os.path.join(out_wd, "input/spec.dat"), sep=",", index=False)
    except:
        print("could not write file:", os.path.join(out_wd, "input/spec.dat"))
        return tbl

    # run MARXAN
    cmd = "cd %s && ./marxan > out.txt" % out_wd
    os.system(cmd)

    # read output
    res = pd.read_csv(os.path.join(out_wd, "output/output_best.csv"))
    indx_PUs = np.where(res["SOLUTION"] == 1)[0]
    return indx_PUs


if __name__ == "__main__":
    # pickle file generated using:
    """
    python3 reinforce_batch_GSNN.py -run 4 -pklfile
    /Users/dsilvestro/Software/BioDivForecast/pickles/init_cell_1000_c50_s500_d0.3_t0.25.pkl
    """
    wd = "/Users/dsilvestro/Documents/Projects/Ongoing/BioDivForecast_full_results/marxan_comparison/marxan_biodivfor/pkls"

    pklfile = "init_cell_1000_c50_s500_d0.3_t0.25.pkl_env.pkl"
    pklfile = "/Users/dsilvestro/Software/BioDivForecast/algorithms/output/prioritize_species.log_d5_opt_e0_plot_D0.1_A0.0_C3.log_5_0.pkl"
    pklfile = os.path.join(wd, pklfile)

    with open(pklfile, "rb") as pkl:
        BioDivGrid = pickle.load(pkl)

    x = get_marxan_solution(BioDivGrid, return_best=True)
    print(x)
