

class AggregatorBase():
    """Base class of aggregators.
    """

    def __init__(self, args, **kwargs):
        self.args = args

    def update_and_set_attr(self):
        """
        update the aggregator parameters if given, otherwise use the default aggregator parameters
        """
        new_defense_params = self.args.defense_params
        self.defense_params = self.default_defense_params
        # update default attack params with new defense_params
        if new_defense_params:
            self.defense_params.update(new_defense_params)
        # set the attack parameters as the class attributes
        for key, value in self.defense_params.items():
            setattr(self, key, value)

    def aggregate(self, updates, **kwargs):
        """Aggregate the inputs/clients' updates, and return the aggregated result.
        updates: 2d numpy array with the [rows, columns] as [clients, updates].
        """
        raise NotImplementedError


""" Template for creating a new aggregator:

# Path: aggregators/aggregatortemplate.py
class AggregatorTemplate(AggregatorBase):
    def __init__(self, args):
        super().__init__(args)

    def aggregate(self, updates, **kwargs):
        # do some aggregation about the updates
        # if you need some additional information, you can pass and get them through kwargs dictionary
        # return the aggregated result

"""
