from attackers.pbases.pbase import PBase


class MPBase(PBase):

    # The following two functions are for non-omniscient attacks and omniscient_attack separately, you can customize your non-omniscient attacks by inheriting this class and re-implementing the corresponding one.

    """
    model poisoning base class. When you inherit from this base class, you can have attack points,
    1. rewrite any function in client, normally local_training and step function for training process manipulation
    1. rewrite non_omniscient
    2. rewrite omniscient_attack
    """

    def non_omniscient(self):
        """
        non-omniscient model poisoning attacks mean attackers perform attacks alone, which happen at the end of each local epoch
        """
        raise NotImplementedError

    def omniscient(self, clients):
        """
        omniscient model poisoning attacks mean attackers perform attacks collusively or even do eavesdropping on other benign updates, which happen at the end of each global epoch
        """
        raise NotImplementedError
