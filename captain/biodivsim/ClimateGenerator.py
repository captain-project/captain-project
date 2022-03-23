import numpy as np

small_number = 1e-10
import scipy.stats


class ClimateGenerator(object):
    def updateClimate(self, climate_layer):
        pass


class SimpleGradientClimateGenerator(object):
    # random disturbance in each cell ~ U(0,1)
    def __init__(
        self, counter, seed=0, steepness=0.1, climate_change=0.025
    ):  # 4.9 degrees difference in a 50x50 grid
        self._counter = counter
        self._steepness = steepness
        self._climate_change = climate_change

    def reset_counter(self):
        self._counter = 0

    def updateClimate(self, climate_layer):
        if self._counter < 1:
            temp = (
                np.linspace(0, climate_layer.shape[0] - 1, climate_layer.shape[0])
                * self._steepness
            )
            climate_layer = np.einsum("ij,i->ij", climate_layer, temp)
            self._counter += 1
        else:
            climate_layer = self._climate_change + climate_layer
        return climate_layer


class RegionalClimateGenerator(object):
    def __init__(self, counter, seed=0, extent=750, expansion_rate=2.5):
        self._counter = counter
        from biodiv.DisturbanceGenerator import (
            DiffusionDisturbanceGenerator as climateInitializer,
        )

        self._climate_obj = climateInitializer(
            counter=0,
            extent=extent,
            n_init_events=1,
            dist_resolution=20,
            expansion_rate=expansion_rate,
            p_grow_disturbance=1,
            max_disturbance=0.5,
            seed=seed,
        )

    def reset_counter(self):
        self._counter = 0

    def updateClimate(self, climate_layer):
        climate_layer = self._climate_obj.updateDisturbance(climate_layer)
        return climate_layer


class GradientClimateGeneratorRnd(object):
    # random disturbance in each cell ~ U(0,1)
    def __init__(
        self,
        counter,
        seed=0,
        steepness=0.1,
        climate_change=0.025,
        sig=10,
        peak_anomaly=2,
    ):
        # 4.9 degrees difference in a 50x50 grid
        self._counter = counter
        self._steepness = steepness
        self._climate_change = climate_change
        self._sig = sig
        self._peak_anomaly = peak_anomaly
        self._n_peaks = 10

    def reset_counter(self):
        self._counter = 0

    def updateClimate(self, climate_layer):
        climate_layer = np.ones(climate_layer.shape)
        temp = (
            np.linspace(0, climate_layer.shape[0] - 1, climate_layer.shape[0])
            * self._steepness
        )
        climate_layer = np.einsum("ij,i->ij", climate_layer, temp)
        climate_layer = self._climate_change * self._counter + climate_layer

        length = climate_layer.shape[0]
        p_cold_anomaly = 0.5  # probability of a negative anomaly

        for i in range(self._n_peaks):
            indx = np.meshgrid(np.arange(length), np.arange(length))
            locsxy = np.random.uniform(0, length, (2, 1))
            sig_tmp = np.random.uniform(2, self._sig, 2)
            disturbance_matrix_tmp = scipy.stats.norm.pdf(
                indx[0], loc=locsxy[0, 0], scale=sig_tmp[0]
            ) * scipy.stats.norm.pdf(indx[1], loc=locsxy[1, 0], scale=sig_tmp[1])

            disturbance_matrix_tmp = (
                disturbance_matrix_tmp / np.max(disturbance_matrix_tmp)
            ) * (self._peak_anomaly * np.random.random())

            if np.random.random() > p_cold_anomaly:
                disturbance_matrix_tmp = -disturbance_matrix_tmp

            climate_layer += disturbance_matrix_tmp

        # print(np.sum(disturbance_matrix_tmp))
        self._counter += 1

        return climate_layer


def get_climate(mode=3, climate_change_magnitude=0.025, peak_anomaly=2.0):
    climate_disturbance = 0
    if mode == 1:
        climate_change = climate_change_magnitude
        CLIMATE_OBJ = SimpleGradientClimateGenerator(0, climate_change=climate_change)
    elif mode == 2:
        climate_disturbance = 1
        CLIMATE_OBJ = RegionalClimateGenerator(0)
    elif mode == 3:  # global warming + random variation
        climate_change = climate_change_magnitude
        PEAK_ANOMALY = peak_anomaly
        CLIMATE_OBJ = GradientClimateGeneratorRnd(
            0, climate_change=climate_change, peak_anomaly=PEAK_ANOMALY
        )
    else:
        CLIMATE_OBJ = 0
    return CLIMATE_OBJ, climate_disturbance
