from aggregators.aggregatorbase import AggregatorBase
from aggregators.krum import krum
import numpy as np
from aggregators import aggregator_registry


@aggregator_registry
class Bulyan(AggregatorBase):
    """[The Hidden Vulnerability of Distributed Learning in Byzantium](https://arxiv.org/abs/1802.07927)
    Bulyan first select a subset of updates via Krum or other norm-based aggregation rules and then computes the coordinate-wise robust aggregation of the remaining updates
    For coordinate-wise robust aggregation, original paper use coordinate-wise closest beta median, other coordinate-wise method, e.g., trimmed mean, can also be used
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        enable_check (bool): whether to enable the check of the number of Byzantine clients
        """
        self.default_defense_params = {"enable_check": False}
        self.update_and_set_attr()

        # with prior knowledge of the number of adversaries
        self.beta = self.args.num_clients - 2 * self.args.num_adv

    def aggregate(self, updates, **kwargs):
        """
        Bulyan condition check
        """
        if self.enable_check:
            if 4*self.args.num_adv + 3 > self.args.num_clients:
                raise ValueError(
                    f"num_adv should be meet 4f+3 <= n, got {4*self.args.num_adv+3} > {self.args.num_clients}.")

        # 1. get the selection set by krum
        set_size = self.args.num_clients - 2 * self.args.num_adv
        selected_idx = []
        while len(selected_idx) < set_size:
            try:
                krum_idx = krum(np.delete(
                    updates, selected_idx, axis=0), self.args.num_adv, return_index=True)
            except ValueError:
                # break, if Krum condition check don't meet anymore
                if len(selected_idx) > 0:
                    break
                else:
                    raise
            except Exception as e:
                # if get other exceptions
                raise e
            # Use extend to add multiple indices efficiently
            selected_idx.append(krum_idx)
        # Convert the list to a NumPy array once the loop is complete
        selected_idx = np.array(selected_idx, dtype=np.int64)

        # for the case of NoAttack, otherwise, argpartition will raise error
        if self.beta == self.args.num_clients or self.beta == len(selected_idx):
            bening_updates = updates[selected_idx]
        else:
            # 2. compute the robust aggregation via coordiante-wise method in selection set
            # return trimmed_mean(updates[selected_idx], self.args.num_adv)# if use trimmed mean as the coordinate-wise aggregation method
            median = np.median(updates[selected_idx], axis=0)
            abs_dist = np.abs(updates[selected_idx] - median)

            # get the smallest beta-closest-median number of elements in axis=0
            beta_idx = np.argpartition(
                abs_dist, self.beta, axis=0)[:self.beta]
            bening_updates = np.take_along_axis(
                updates[selected_idx], beta_idx, axis=0)
        return np.mean(bening_updates, axis=0)
