from typing import Dict, List, Tuple, Any

import numpy as np
from scipy import stats


class RandomParamGenerator:
    def __init__(self, params_config: Dict[str, Any]):
        self.params_config = params_config

    # def _draw_dist_rand_number(self, distribution: stats.rv_continuous):
    #     p = np.random.uniform(0, 1, 1)
    #     return distribution.ppf(p)

    def _get_param_dict(self):
        params_dict = {}
        for cur_param_name, cur_param_space in self.params_config.items():
            # Parameter space is defined by a continuous distribution
            if isinstance(cur_param_space, stats._distn_infrastructure.rv_frozen):
                params_dict[cur_param_name] = cur_param_space.rvs(1)[0]
            # Parameter space is defined as a list of discrete values
            elif isinstance(cur_param_space, list) or isinstance(
                cur_param_space, np.ndarray
            ):
                if isinstance(cur_param_space[0], tuple):
                    param_values = [t[0] for t in cur_param_space]
                    weight_values = [t[1] for t in cur_param_space]
                    params_dict[cur_param_name] = np.random.choice(param_values, 1, replace=False, p=weight_values)[0]
                else:
                    params_dict[cur_param_name] = cur_param_space[
                        np.random.randint(0, len(cur_param_space), 1)[0]
                    ]
            else:
                raise ValueError(f"Parameter definition of type {type(cur_param_space)} "
                                 f"for {cur_param_name} is not supported")

        return params_dict

    def get_param_set(self):
        yield self._get_param_dict()
