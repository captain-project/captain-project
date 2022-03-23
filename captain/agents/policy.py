import sys
import numpy as np
import scipy
from ..agents import state_monitor as state_monitor


class PolicyNN(object):
    def __init__(
        self,
        num_features,
        num_meta_features,
        num_output,
        coeff_features,
        coeff_meta_features,
        temperature,
        mode,
        observe_error,
        nodes_l1=16,
        nodes_l2=16,
        nodes_l3=16,
        sp_threshold=1,
        fully_connected=0,
        verbose=0,
        flattened=False,
    ):
        # check meta features matches the features generating function
        self._num_meta_features = num_meta_features
        self._num_features = num_features
        self._num_output = num_output
        # init coefficients
        self._coeff = np.append(coeff_features, coeff_meta_features)
        self._temperature = temperature
        self._mode = mode  # indices of features
        self._observe_error = observe_error
        self._nodes_l1 = nodes_l1
        self._nodes_l2 = nodes_l2
        self._nodes_l3 = nodes_l3
        self._sp_threshold = sp_threshold
        self._fully_connected = fully_connected
        self._verbose = verbose
        self._flattened = flattened

    @property
    def num_output(self):
        return self._num_output

    @property
    def temperature(self):
        return self._temperature

    @property
    def coeff(self):
        return self._coeff

    def perturbeParams(self, noise):
        self._coeff += noise
        # self._coeff[ : -self._num_meta_features ] += noise[ : -self._num_meta_features ]
        # print("Before", self._coeff[self._num_features:])
        # self._coeff[self._num_features:] = UpdateUniform(self._coeff[ self._num_features : ] + 0)
        # print("after",self._coeff[self._num_features:])

    def setCoeff(self, newCoeff):
        self._coeff = newCoeff

    def probs(
        self, rich_state, lastObs=None, sp_quadrant_list_arg=None, return_lastObs=False
    ):

        # print(np.array(rich_state.protection_cost)[0:10])
        # logistic function applied here: coeff_meta_features must be between 0 and 1
        coeff_meta_features = state_monitor.get_thresholds(
            self._coeff[-self._num_meta_features :]
        )
        if lastObs is None:
            lastObs = state_monitor.extract_features(
                rich_state.grid_obj_most_recent,
                rich_state.grid_obj_previous,
                rich_state.resolution,
                rich_state.protection_matrix,
                rare_sp_qntl=coeff_meta_features[0],
                smallrange_sp_qntl=coeff_meta_features[self._num_meta_features - 1],
                mode=self._mode,
                observe_error=self._observe_error,
                cost_quadrant=rich_state.protection_cost,
                budget=rich_state.budget_left,
                sp_threshold=self._sp_threshold,
                sp_values=rich_state.sp_values,
                flattened=self._flattened,
                met_prot_target=rich_state.met_prot_target,
                min_pop_requirement=rich_state.min_pop_requirement,
                sp_quadrant_list_arg=sp_quadrant_list_arg,
            )

        state = lastObs.stats_quadrant
        if self._verbose:
            print(state[:20, :])
            print("features protected cells:")
            print(state[np.where(state[:, -1] == 1)[0], :])
            print(np.mean(state, 0), np.min(state, 0), np.max(state, 0))
        # if self._verbose:
        # print(lastObs.stats_quadrant[0:5,:])
        # print(np.sum(rich_state.grid_obj_most_recent.individualsPerSpecies()))

        # remove metafeatures
        coeff_policy = self._coeff[: -self._num_meta_features]
        internal_state = state[:, :]

        if self._fully_connected > 0:
            sys.exit("NN setting not available")
        elif self._nodes_l1 == 1:
            # linear regression
            h2 = np.einsum("nf, f -> n", internal_state, coeff_policy)
        else:  # NN with parameter sharing
            if self._nodes_l2:  # only used if using additional hidden layer
                tmp = coeff_policy[: self._num_features * self._nodes_l1]
                weights_l1 = tmp + 0
                tmp_coeff = weights_l1.reshape(self._num_features, self._nodes_l1)

                weights_l2 = coeff_policy[len(tmp) : -self._nodes_l3]
                # print(tmp_coeff.shape, weights_l2.shape)
                tmp_coeff2 = weights_l2.reshape(self._nodes_l1, self._nodes_l2)

                weights_l3 = coeff_policy[-self._nodes_l3 :]
            else:
                tmp = coeff_policy[: self._num_features * self._nodes_l1]
                weights_l1 = tmp + 0
                tmp_coeff = weights_l1.reshape(self._num_features, self._nodes_l1)

                weights_l3 = coeff_policy[-(self._nodes_l3) :]

            z1 = np.einsum("nf, fi->ni", internal_state, tmp_coeff)
            z1[z1 < 0] = 0
            if self._nodes_l2:
                # print(tmp_coeff2.shape, z1.shape)
                h1 = np.einsum("ni,ic->nc", z1, tmp_coeff2)
                h1[h1 < 0] = 0
                z1 = h1 + 0
            h2 = np.einsum("f,nf->n", weights_l3, z1)

        if self._temperature != 1:
            h2 *= self._temperature
            # same as
            """
            probs = scipy.special.softmax(h2)
            if self._temperature != 1:
                probs = probs ** self._temperature
                probs /= np.sum(probs)
            """
        probs = scipy.special.softmax(h2)
        # set to 0 probs of already protected units
        probs[internal_state[:, -1] == 1] = 0
        if np.sum(probs) < 1e-20:
            # avoid overflows
            probs += 1e-20
        probs /= np.sum(probs)
        if return_lastObs:
            return probs, lastObs
        else:
            return probs

    def setTemperature(self, temp):
        self._temperature = temp


def get_NN_model_prm(num_features, n_NN_nodes, num_output):
    num_meta_features = 1  # TODO: check metafeatures
    nodes_layer_1 = n_NN_nodes[0]
    nodes_layer_2 = n_NN_nodes[1]  # set > 0 to add hidden layer
    if n_NN_nodes[0] == 1:
        nodes_layer_3 = 0
    elif n_NN_nodes[1] == 0:
        nodes_layer_3 = nodes_layer_1
    else:
        nodes_layer_3 = nodes_layer_2
    n_prms = (
        num_features * nodes_layer_1 + nodes_layer_1 * nodes_layer_2 + nodes_layer_3
    )
    # print("\n\nget_NN_model_prm")
    # print(num_features, nodes_layer_1, num_features * nodes_layer_1,nodes_layer_1 * nodes_layer_2, nodes_layer_3, n_prms)
    return [
        num_output,
        num_meta_features,
        nodes_layer_1,
        nodes_layer_2,
        nodes_layer_3,
        n_prms,
    ]
