import numpy as np
import glob
import random
import os, sys
import dendropy
from dendropy.simulate import treesim
import baltic as bt


def convert_to_bt_tree(tree):
    tree_string = tree.as_string(schema="newick")
    tree_string = tree_string.split("[&R] ")[1]
    tree_string = tree_string.split("\n")[0]
    ll = bt.make_tree(tree_string)
    ll.sortBranches()
    return ll


def get_ED(tree):
    tree = convert_to_bt_tree(tree)
    taxa_names = []
    eds = []
    for k in tree.getExternal():
        x1 = k.height
        x2 = k.parent.height
        siblings = tree.traverse_tree(
            k.parent, include_condition=lambda w: w.is_leaf() and w != k
        )
        x3 = np.min([s.height for s in siblings])
        ed = x1 + x3 - (x2 * 2)  ## patristic distance, computed as (x1-x2)+(x3-x2)
        # print(k.name, ed)
        taxa_names.append(k.name)
        eds.append(ed)

    numeric_tip_labels = np.array([s.strip("T") for s in taxa_names]).astype(int)
    # sorted_taxa = np.array(taxa_names)[np.argsort(numeric_tip_labels)]
    sorted_eds = np.array(eds)[np.argsort(numeric_tip_labels)]
    return sorted_eds


class ReadRandomPhylo:
    def __init__(self, phylofolder, verbose=False, seed=0, n_species=None):
        self._phylofolder = phylofolder
        self._verbose = verbose
        self._tree_files = [
            f for f in glob.glob(self._phylofolder + "*.tre", recursive=False)
        ]
        self._ED_files = [
            f for f in glob.glob(self._phylofolder + "*.txt", recursive=False)
        ]
        self._n_species = n_species
        if seed:
            rr = seed
        else:
            rr = random.randint(1000, 9999)
        self._rr = rr

    def getPhylo(self):
        np.random.seed(self._rr)
        # randomly select tree from folder
        phylofile_i = np.random.choice(np.arange(len(self._tree_files)))
        print("Picked tree n.", phylofile_i)
        tree_file = self._tree_files[phylofile_i]
        ED_file = self._ED_files[phylofile_i]
        phylo_tree = dendropy.Tree.get_from_path(tree_file, "nexus")
        all_tip_labels = np.array([taxon.label for taxon in phylo_tree.taxon_namespace])
        n_species = len(all_tip_labels)
        # read evolutionary distinctiveness table
        eds = np.loadtxt(ED_file, skiprows=1, usecols=1)
        eds = eds / np.sum(eds) * n_species
        return phylo_tree, all_tip_labels, eds, self._tree_files[phylofile_i]


#


class SimRandomPhylo:
    def __init__(self, n_species, verbose=True, seed=0):
        self._phylofolder = None
        self._verbose = verbose
        self._tree_files = None
        self._ED_files = None
        self._n_species = n_species
        if seed:
            rr = seed
        else:
            rr = random.randint(1000, 9999)
        self._rr = rr

    def getPhylo(self):
        np.random.seed(self._rr)
        b, d = 1, np.random.uniform(0, 0.9)
        birth_rate_sd = np.random.exponential(1)
        if self._verbose:
            print("Simulating %s species tree..." % self._n_species)
        rng = random.Random(self._rr)
        phylo_tree = treesim.birth_death_tree(
            b,
            d,
            birth_rate_sd=birth_rate_sd,  # rate variation across branches
            num_extant_tips=self._n_species,
            rng=rng,
        )
        # phylo_tree.print_plot(plot_metric='age')
        all_tip_labels = np.array([taxon.label for taxon in phylo_tree.taxon_namespace])
        # phylo_tree.write_to_path('sim_tree.tre', 'newick')
        eds = get_ED(phylo_tree)
        eds = eds / np.sum(eds) * self._n_species
        return phylo_tree, all_tip_labels, eds, "dendropy_sim"


# class SimPhyloInitializer():
#     sim_in_python = 0
#     if sim_in_python:
#         b, d = 1, np.random.uniform(0, 0.9)
#         phylo_tree = treesim.birth_death_tree(b, d, num_extant_tips=self._n_species)
#         self._phylo_tree = phylo_tree
#         self._all_tip_labels = np.array([taxon.label for taxon in phylo_tree.taxon_namespace])
#         # phylo_tree.print_plot(plot_metric='age')
#         #phylo_tree.write_to_path('sim_tree.tre', 'newick')
#         self.get_sp_pd_contribution()
#     else:
#         wd = os.path.join(sys.path[0],"biodiv")
#         rnd_id = random.randint(100,199)
#         cmd = "cd %s; Rscript SimulatorPhyloInit.R --f %s/phylo/ --n %s --i %s --t %s " \
#             % (wd, wd, n_species, rnd_id, phylo_tree)
#         os.system(cmd)
#         # (phylo_tree) path to output file
#         tree_file = "phylo/%s.tre" % rnd_id
#         ed_file = "phylo/%s_ED.txt" % rnd_id
#         phylo_tree = dendropy.Tree.get_from_path(os.path.join(wd,tree_file), 'nexus')
#         all_tip_labels = np.array([taxon.label for taxon in phylo_tree.taxon_namespace])
#         # read evolutionary distinctiveness table
#         eds = np.loadtxt(os.path.join(wd,ed_file),skiprows = 1, usecols=1)
#         phylo_ed = eds / np.sum(eds) * n_species
