from collections import OrderedDict
from .client import Client
from attackers import get_attacker_handler
from datapreprocessor.data_utils import subset_by_idx
from attackers import data_poisoning_attacks, hybrid_attacks


def init_clients(args, client_indices, train_dataset, test_dataset):
    clients = []
    for worker_id in range(args.num_clients):
        # for attacker, if the attack type is not model poisoning attack, use the default client class. For data poisoning attacks, it's already handled in the client class.
        # for benign clients, use the default client class
        if args.attack == "NoAttack":
            """
            For NoAttack scenario, use client class, and ignore args.num_adv
            """
            client_obj = Client
        else:
            if args.num_adv == 0:
                raise AssertionError(
                    "Attack {args.attack} specified, but attackers set to 0.")
            client_obj = Client if worker_id >= args.num_adv else get_attacker_handler(
                args.attack)
        local_dataset = subset_by_idx(
            args, train_dataset, client_indices[worker_id])
        tmp_client = client_obj(args, worker_id,
                                local_dataset, test_dataset)
        clients.append(tmp_client)
    return clients


def set_fl_algorithm(args, the_server, clients):
    """set the federated learning algorithm for the server and clients. If the algorithm type is not specified in arguments, use the default algorithm type of the server.

    Args:
        the_server (Server): server object
        clients (Client): a list of client objects
        algorithm (str): federated learning algorithm types

    Raises:
        ValueError: No specified or default algorithm type can be used
    """
    if args.algorithm:
        alg_type = args.algorithm
    elif hasattr(the_server, 'algorithm'):
        args.algorithm = the_server.algorithm
    else:
        raise ValueError(
            "No specified algorithm or default algorithm type of the server. Please specify an algorithm type, with `--algorithm`")

    the_server.set_algorithm(alg_type)
    for client in clients:
        client.set_algorithm(alg_type)


def evaluate(the_server, test_dataset, args, global_epoch):
    """
    Backdoor attacks evaluation requires inference-time attacks. However, since the server is unaware of the backdoor attacks, the client's `client_test` is used in the coordinator for ASR evaluation.
    """
    test_keys = ["Test Acc", "Test loss", "ASR", "ASR loss"]
    results = OrderedDict()

    # normal evaluation
    imbalanced_flag = True if 'imbalanced' in args.distribution else False
    if imbalanced_flag:
        test_keys.insert(1, 'Tail Acc')

    test_loader = the_server.get_dataloader(test_dataset, train_flag=False)
    clean_test = the_server.test(
        the_server.global_model, test_loader, imbalanced=imbalanced_flag)

    for idx in range(len(clean_test)):
        results[test_keys[idx]] = clean_test[idx]

    if args.attack in data_poisoning_attacks + hybrid_attacks:
        # index [0, f] is poisoning attacker
        results['ASR'], results['ASR loss'] = the_server.clients[0].client_test(
            the_server.global_model, test_dataset, poison_epochs=True)
    return results
