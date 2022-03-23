import sys
import numpy as np

from numba import jit
import pickle

def print_update(s):
    sys.stdout.write('\r')
    sys.stdout.write(s)
    sys.stdout.flush()

@jit(nopython=True)  # compiled by numba: 30X speedup
def get_all_to_all_dist_jit(coord, n_cells):
    print("calculating  distances...")
    linear_indx_cells = np.arange(n_cells ** 2)
    d = np.zeros((len(linear_indx_cells), len(coord), len(coord)))
    for i in linear_indx_cells:
        xa = coord[int(np.floor(i / n_cells))]
        ya = coord[np.mod(i, n_cells)]  # numpy modulo operation
        d_temp = np.zeros((len(coord), len(coord)))
        h = 0
        for x in coord:
            j = 0
            for y in coord:
                d_temp[h, j] = np.sqrt((xa - x) ** 2 + (ya - y) ** 2)
                j += 1
            h += 1
        d[i] = d_temp
    print("done.")
    return d


@jit(nopython=True)
def get_coordinates_jit(coord, n_cells):
    linear_indx_cells = np.arange(n_cells ** 2)
    d = np.zeros((len(linear_indx_cells), 3))
    for i in linear_indx_cells:
        xa = int(np.floor(i / n_cells))
        ya = np.mod(i, n_cells)
        d[i, :] = np.array([i, xa, ya])
    return d


def init_cell_objects(
    cell_id_n_coord, d_matrix, n_species, cell_carrying_capacity, lat_steep
):
    print("init cells...")
    list_cells = []
    for i in cell_id_n_coord[:, 0]:
        c_coord = cell_id_n_coord[i, 1:]
        c_id = cell_id_n_coord[i, 0]
        c_carrying_cap = cell_carrying_capacity  # for now assuming all equal
        c_species_hist = np.zeros(n_species, dtype=np.uint16)
        c_dist_matrix = d_matrix[i]
        c = CellClass(
            c_coord,
            c_id,
            c_dist_matrix,
            c_species_hist,
            c_carrying_cap,
            lat_steepness=lat_steep,
        )
        list_cells.append(c)
    print("done.")
    return list_cells


def init_propagate_species(
    curr_ind,
    max_n_ind,
    list_cells,
    cell_id_n_coord,
    cell_carrying_capacity,
    species_id,
    disp_rate=1.0,
):
    aval_space = np.array([c.room_for_one for c in list_cells])
    n_indviduals_per_cell = np.array([c.n_individuals for c in list_cells])
    n_indviduals_per_cell_sp_i = np.zeros(len(list_cells))

    while True:
        for cell_id in curr_ind:
            # when reached carrying capacity set respective aval_space to 0
            aval_space[n_indviduals_per_cell == cell_carrying_capacity] = 0

            c = list_cells[cell_id]
            disp = (
                1.0 / disp_rate
            )  # higher dispersal rate, smaller rate of the exp distribution
            disp_probability = disp * np.exp(
                -disp * c.dist_matrix
            )  # dispersal is exponentially distributed
            disp_vec = disp_probability.flatten()
            sampling_prob = disp_vec * aval_space

            selected_cell = np.random.choice(
                cell_id_n_coord[:, 0], p=sampling_prob / np.sum(sampling_prob)
            )
            # update species histogram in selected cell
            list_cells[selected_cell].species_hist[species_id] += 1

            # keep track of where individuals are being added
            n_indviduals_per_cell[selected_cell] += 1
            n_indviduals_per_cell_sp_i[selected_cell] += 1

            # append new individual to current population
            curr_ind.append(selected_cell)
            # print(curr_ind, np.max(sampling_prob)/np.sum(sampling_prob))
            if len(curr_ind) > max_n_ind:
                # n_indviduals_per_cell_sp_i = np.unique(curr_ind,return_counts=True),
                # but includes 0s
                return (
                    n_indviduals_per_cell,
                    n_indviduals_per_cell_sp_i,
                )  # np.array(curr_ind),


@jit(nopython=True)
def find_a_parabola_jit(target_x, target_y=0.05):
    x = np.linspace(0, 1000, 10000)
    a = 3.0
    while True:
        y = a * (x ** 2)
        delta_y = np.abs(y - target_y)
        indx = np.argmin(delta_y)
        delta_x = x[indx]
        a = a * 0.99
        if np.abs(delta_x - target_x) < 0.1:
            break
        if a < 0.01:
            break
    return a


class CellClass:
    def __init__(
        self,
        coord,
        cell_id,
        dist_matrix,
        species_hist,
        carrying_capacity,
        disturbance=0,
        protection=1,
        pathogen=0,
        lat_steepness=0.1,
    ):
        self.coord = coord  # array of two integers
        self.id = cell_id  # 1 integer
        self.carrying_capacity = carrying_capacity  # 1 integer
        self.species_hist = species_hist  # array of integers
        self.dist_matrix = dist_matrix  # 2D array
        self.disturbance = disturbance  # in [0,1]
        self.protection = protection  # in [0,1]
        self.pathogen = pathogen  # presence/absence
        self.temperature = (
            coord[0] * lat_steepness
        )  # deviation from present temperature

    # disp_temp = (1/(self.dist_matrix+1))
    # disp_temp = np.exp(-self.dist_matrix)
    # disp_temp = 1/(np.log((1+self.dist_matrix))+1)
    # self.dispersal_prob    = (disp_temp/np.sum(disp_temp))

    # read-only property decorator
    @property
    def n_individuals(self):
        return np.sum(self.species_hist)

    @property
    def available_space(self):
        return self.carrying_capacity - self.n_individuals

    @property
    def room_for_one(self):
        return np.min((self.available_space, 1))

    @property
    def n_species(self):
        return len(self.species_hist[self.species_hist > 0])


def load_pickle_file(pkl):
    with open(pkl, "rb") as pkl:
        list_cells = pickle.load(pkl)
        return list_cells
