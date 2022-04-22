import numpy as np

from ..algorithms.env_setup import *

np.set_printoptions(suppress=True, precision=3)


class ConservationTarget():

    def reset_species_target(self, species_target):
        self._species_target = species_target

    def init_sp_target(self, x):
        return x

class RangeConservationTarget(ConservationTarget):
    def __init__(self,
                 min_fr=0.1,
                 min_range=1,
                 max_range=np.inf,
                 loglinear=True,
                 step_size=0.01 # set to 0 for static target
                 ):
        self._step_size = step_size
        self._min_fr = min_fr
        self._min_range = min_range
        self._max_range = max_range
        self._loglinear = loglinear

    def init_sp_target(self, x):
        if self._loglinear:
            def f(i):
                return np.log10(i)
        else:
            def f(i):
                return i

        max_x = np.max(f(x))
        min_x = f(self._min_range)
        max_fr = 1
        slope = (max_fr - self._min_fr) / (min_x - max_x)
        intercept = slope * (-f(self._min_range)) + max_fr
        y = intercept + f(x) * slope
        y[y > 1] = 1
        count = y * x
        count[count > self._max_range] = self._max_range
        return count

    def update_target(self, target_pop):
        new_target = target_pop * (1 + self._step_size)
        return new_target


class FractionConservationTarget(ConservationTarget):
    def __init__(self,
                 step_size=0.01,
                 protect_fraction=0.1):
        self._step_size = step_size
        self._protect_fraction = protect_fraction

    def update_target(self, target_pop):
        new_target = target_pop * (1 + self._step_size)
        return new_target

    def init_sp_target(self, x):
        return np.ceil(self._protect_fraction * x)



def plot_target(protect_target : ConservationTarget, x=None):
    # test
    import matplotlib.pyplot as plt
    if x is None:
        x = np.arange(1, 1000)
    else:
        x = np.sort(x)
    y = protect_target.init_sp_target(x)
    fig = plt.figure(figsize=(15, 8))
    fig.add_subplot(121)
    plt.plot(x, y/x, '-', c='#045a8d')
    plt.xlabel("Species range size")
    plt.ylabel("Fraction of protected range (target)")
    fig.add_subplot(122)
    plt.plot(x, y, '-', c='r')
    plt.xlabel("Species range size")
    plt.ylabel("Protected range (target)")
    fig.show()