import gym
import numpy as np
import abc
from itertools import count
import collections
from ..biodivsim.BioDivEnv import BiodivEnvUtils


class StateAdaptor(object):
    def adapt(self, state, info):
        pass


class TrivialStateAdaptor(StateAdaptor):
    def adapt(self, state, info=None):
        return state


class SpatialOnlyStateAdaptor(StateAdaptor):
    def __init__(self, state_len):
        self.state_len = state_len

    def adapt(self, state, info=None):
        return state[:, :-1].reshape((self.state_len))


RichState = collections.namedtuple(
    "RichState",
    (
        "grid_obj_most_recent",
        "grid_obj_previous",
        "resolution",
        "protection_matrix",
        "protection_cost",
        "budget_left",
        "sp_values",
        "min_pop_requirement",
        "met_prot_target",
    ),
)


class RichStateAdaptor(StateAdaptor):
    """
    info['full_grid'] = self.bioDivGrid.h
    info['protection_matrix'] = self.bioDivGrid.protection_matrix
    info['disturbance_matrix'] = self.bioDivGrid.disturbance_matrix
    info['selective_disturbance'] = self.bioDivGrid.selective_disturbance_matrix

    rich_state.grid_obj_most_recent,
            rich_state.grid_obj_previous,
            rich_state.resolution,
            rich_state.current_protection_matrix,
    """

    def adapt(self, state, info):
        return RichState(
            state["grid_obj_most_recent"],
            state["grid_obj_previous"],
            state["resolution"],
            state["protection_matrix"],
            state["protection_cost"],
            state["budget_left"],
            state["sp_values"],
            state["min_pop_requirement"],
            state["met_prot_target"],
        )


class ConvolutionOnFeaturesAdaptor(StateAdaptor):
    def __init__(self, state_len):
        self.state_len = state_len

    def adapt(self, state, info=None):
        return state[:, :-1].reshape((self.state_len))


class ActionAdaptor(object):
    def adapt(self, action):
        pass


class RichProtectActionAdaptor(ActionAdaptor):
    def __init__(self, grid_size, resolution):
        self.grid_size = grid_size
        self.resolution = resolution

    def adapt(self, action):
        return BiodivEnvUtils.getRichProtectAction(
            action, self.grid_size, self.resolution
        )
