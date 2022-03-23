import numpy as np
import random
import matplotlib.pyplot as plt

# matplotlib.use('Agg')
import matplotlib.backends.backend_pdf

plt.rcParams.update({"figure.max_open_warning": 0})
import seaborn as sns

sns.set()
import pickle
import sys, os

from ..biodivsim.CellClass import *

np.set_printoptions(suppress=1, precision=3)  # prints floats, no scientific notation


def get_nature_emoji():
    nature = np.array(
        [
            u"\U0001F400",
            u"\U0001F439",
            u"\U0001F430",
            u"\U0001F407",
            u"\U0001F43F",
            u"\U0001F994",
            u"\U0001F40F",
            u"\U0001F411",
            u"\U0001F410",
            u"\U0001F42A",
            u"\U0001F42B",
            u"\U0001F999",
            u"\U0001F992",
            u"\U0001F418",
            u"\U0001F98F",
            u"\U0001F993",
            u"\U0001F98C",
            u"\U0001F42E",
            u"\U0001F402",
            u"\U0001F403",
            u"\U0001F404",
            u"\U0001F405",
            u"\U0001F406",
            u"\U0001F989",
            u"\U0001F99C",
            u"\U0001F40A",
            u"\U0001F422",
            u"\U0001F98E",
            u"\U0001F40D",
            u"\U0001F995",
            u"\U0001F996",
            u"\U0001F433",
            u"\U0001F40B",
            u"\U0001F42C",
            u"\U0001F41F",
            u"\U0001F420",
            u"\U0001F421",
            u"\U0001F988",
            u"\U0001F419",
            u"\U0001F41A",
            u"\U0001F40C",
            u"\U0001F98B",
            u"\U0001F41B",
            u"\U0001F41C",
            u"\U0001F41D",
            u"\U0001F41E",
            u"\U0001F997",
            u"\U0001F577",
            u"\U0001F982",
            u"\U0001F99F",
            u"\U0001F9A0",
            u"\U0001F331",
            u"\U0001F332",
            u"\U0001F333",
            u"\U0001F334",
            u"\U0001F335",
            u"\U0001F33E",
            u"\U0001F33F",
        ]
    )
    return np.random.choice(nature)


# SAVE CELL AND SPECIES OBJECTS
def save_object_as_list(obj, filename):
    CellList = []
    for c in obj:
        l = [
            c.coord,
            c.id,
            c.dist_matrix,
            c.species_hist,
            c.carrying_capacity,
            c.disturbance,
            c.protection,
        ]
        CellList.append(l)

    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(CellList, output, pickle.HIGHEST_PROTOCOL)


# TODO: add phylo simulation here too


def init_simulated_system(
    seed=0,
    disp_rate=0.3,
    grid_size=50,
    n_species=100,
    cell_capacity=100,
    out_dir="./sim_data",
    verbose=1,
):
    try:
        os.mkdir(out_dir)
    except (FileExistsError):
        pass
    try:
        os.mkdir(os.path.join(out_dir, "pickles"))
    except (FileExistsError):
        pass

    if seed == 0:
        rseed = np.random.randint(1000, 9999)
    else:
        rseed = seed
    np.random.seed(rseed)
    random.seed(rseed)

    out_tag = ""

    # init Area
    cell_carrying_capacity = cell_capacity  # init (max) individuals per cell
    total_cells = grid_size ** 2
    SP_ID = np.arange(n_species)
    death_at_climate_boundary = 0.25  # if set to 0.5 death probability is 50% at the
    # climatic boundaries (based on empirical init range)
    lat_steepness = 0.1  # 5 degrees difference

    out_tag += "_d%s_t%s" % (disp_rate, death_at_climate_boundary)

    # rel_rank_abundance_distribution = np.sort(np.random.weibull(0.75,n_species))[::-1] +(np.random.uniform(0.2,0.5,n_species))
    # TRUNCATED WEIBULL
    rel_rank_abundance_distribution = np.sort(
        np.random.weibull(0.75, n_species + int(0.10 * n_species))
    )[::-1]
    rel_rank_abundance_distribution = rel_rank_abundance_distribution[0:n_species]
    # fatter tail:
    # rel_rank_abundance_distribution = np.sort(np.random.weibull(0.5,n_species))[::-1]+np.sort(np.random.uniform(0.2,0.5,n_species))

    # get distances all vs all (3D array)
    coord = np.linspace(0.5, grid_size - 0.5, grid_size)
    d_matrix = get_all_to_all_dist_jit(coord, grid_size)
    # table with cell ID and x/y coordinantes
    cell_id_n_coord = get_coordinates_jit(coord, grid_size).astype(int)

    cell_file_pkl = "pickles/init_cell_%s_c%s_s%s%s.pkl" % (
        rseed,
        grid_size,
        n_species,
        out_tag,
    )
    cell_file_pkl = os.path.join(out_dir, cell_file_pkl)

    # init cell object
    list_cells = init_cell_objects(
        cell_id_n_coord, d_matrix, n_species, cell_carrying_capacity, lat_steepness
    )
    # climatic gradient
    temp_by_cell = np.array([c.temperature for c in list_cells]).reshape(
        grid_size, grid_size
    )

    # tot number of individuals per species (init)
    tot_init_individuals = cell_carrying_capacity * total_cells
    rel_freq_species = rel_rank_abundance_distribution / np.sum(
        rel_rank_abundance_distribution
    )
    sample_individuals_per_species = np.random.choice(
        SP_ID, size=tot_init_individuals, p=rel_freq_species, replace=True
    )
    n_individuals_per_species = np.unique(
        sample_individuals_per_species, return_counts=True
    )

    # init all species
    rnd_species_order = np.random.choice(SP_ID, len(SP_ID), replace=False)
    aval_space = np.array([c.room_for_one for c in list_cells])
    n_indviduals_per_cell_1D = np.array([c.n_individuals for c in list_cells])

    j = 1
    for species_id in rnd_species_order:
        max_n_ind = n_individuals_per_species[1][species_id]
        if verbose:
            print_update(
                "%s/%s init species %s (%s ind.) %s"
                % (j, n_species, species_id, max_n_ind, get_nature_emoji())
            )
        aval_space[n_indviduals_per_cell_1D == cell_carrying_capacity] = 0
        rnd_starting_cell = np.random.choice(
            cell_id_n_coord[:, 0], p=aval_space / np.sum(aval_space)
        )
        n_indviduals_per_cell_1D, n_indviduals_per_cell_sp_i = init_propagate_species(
            [rnd_starting_cell],
            max_n_ind,
            list_cells,
            cell_id_n_coord,
            cell_carrying_capacity,
            species_id,
            disp_rate,
        )

        j += 1
    save_object_as_list(list_cells, cell_file_pkl)
    if verbose:
        print("\nSystem saved in: ", cell_file_pkl)
    return os.path.abspath(cell_file_pkl)
