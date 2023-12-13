import numpy as np
import pandas as pd
from _collections_abc import Iterable

def summarize_policy_empirical(env_list, use_ccordinates=True):
    if isinstance(env_list, Iterable):
        pass
    else:
        env_list = [env_list]
    comb_output = np.zeros(env_list[0].bioDivGrid._pus_id.shape)
    for i in range(len(env_list)):
        res = env_list[i]
        protected_indx = np.array(res.protection_sequence)
        protected_PUs = res.bioDivGrid._pus_id[protected_indx] # sorted by selection order
        comb_output += res.bioDivGrid.protection_matrix.flatten()
    
    comb_output /= len(env_list)
    
    # add coordinates to output, if available
    if env_list[0].bioDivGrid.coords is not None and use_ccordinates:
        puid = env_list[0].bioDivGrid.coords['PUID'].to_numpy()
        included_pus = [i for i in range(len(puid)) if puid[i] in res.bioDivGrid._pus_id]
        lon = env_list[0].bioDivGrid.coords['Coord_x'].to_numpy()[included_pus]
        lat = env_list[0].bioDivGrid.coords['Coord_y'].to_numpy()[included_pus]
        res_pd = pd.DataFrame(np.array([res.bioDivGrid._pus_id, lon, lat, comb_output]).T)
        res_pd.columns = ["PUID","Longitude","Latitude","Priority"]
    else:
        res_pd = pd.DataFrame(np.array([res.bioDivGrid._pus_id, comb_output]).T)
        res_pd.columns = ["PUID","Priority"]
    
    return res_pd
