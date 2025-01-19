import gc
import logging
import time
from fl import coordinator
from global_args import benchmark_preprocess, read_args, override_args, single_preprocess
from global_utils import avg_value, print_filtered_args, setup_logger, setup_seed
from datapreprocessor.data_utils import load_data, split_dataset
from fl.server import Server
from plot_utils import plot_accuracy


def fl_run(args):
    """
    function to run federated learning logics
    """
    # setup logger
    args.logger = setup_logger(
        __name__, f'{args.output}', level=logging.INFO)
    print_filtered_args(args, args.logger)
    start_time = time.time()
    args.logger.info(
        f"Started on {time.asctime(time.localtime(start_time))}")
    # fix randomness
    setup_seed(args.seed)

    # 1. load dataset and split dataset indices for clients with i.i.d or non-i.i.d
    train_dataset, test_dataset = load_data(args)
    client_indices, test_dataset = split_dataset(
        args, train_dataset, test_dataset)
    args.logger.info("Data partitioned")

    # 2. initialize clients and server with seperate training data indices
    clients = coordinator.init_clients(
        args, client_indices, train_dataset, test_dataset)
    the_server = Server(args, clients, test_dataset, train_dataset)

    # 3. initialize the federated learning algorithm for clients and server
    coordinator.set_fl_algorithm(args, the_server, clients)
    args.logger.info("Clients and server are initialized")
    args.logger.info("Starting Training...")
    for global_epoch in range(args.epochs):
        epoch_msg = f"Epoch {global_epoch:<3}\t"
        # print(f"Global epoch {global_epoch} begin")
        # server dispatches numpy version global weights 1d vector to clients
        global_weights_vec = the_server.global_weights_vec

        # clients' local training
        avg_train_acc, avg_train_loss = [], []
        for client in clients:
            client.load_global_model(global_weights_vec)
            train_acc, train_loss = client.local_training()
            client.fetch_updates()
            avg_train_acc.append(train_acc)
            avg_train_loss.append(train_loss)

        avg_train_loss = avg_value(avg_train_loss)
        avg_train_acc = avg_value(avg_train_acc)
        epoch_msg += f"\tTrain Acc: {avg_train_acc:.4f}\tTrain loss: {avg_train_loss:.4f}\t"

        # perform post-training attacks, for omniscient model poisoning attack, pass all clients
        omniscient_attack(clients)

        # server collects weights from clients
        the_server.collect_updates(global_epoch)
        the_server.aggregation()
        the_server.update_global()

        # evalute the attack success rate (ASR) when a backdoor attack is launched
        test_stats = coordinator.evaluate(
            the_server, test_dataset, args, global_epoch)

        # print the training and testing results of the current global_epoch
        epoch_msg += "\t".join(
            [f"{key}: {value:.4f}" for key, value in test_stats.items()])
        args.logger.info(epoch_msg)
        # clear memory
        gc.collect()

    if args.record_time:
        report_time(clients, the_server)

    plot_accuracy(args.output)

    end_time = time.time()
    time_difference = end_time - start_time
    minutes, seconds = int(
        time_difference // 60), int(time_difference % 60)
    args.logger.info(
        f"Training finished on {time.asctime(time.localtime(end_time))} using {minutes} minutes and {seconds} seconds in total.")


def report_time(clients, the_server):
    [c.time_recorder.report(f"Client {idx}") for idx, c in enumerate(clients)]
    the_server.time_recorder.report("Server")


def omniscient_attack(clients):
    """
    Perform an omniscient attack, which involves eavesdropping or collusion
    between malicious clients to craft adversarial updates.
    """
    # Filter out all omniscient attackers from the client list
    omniscient_attackers = [
        client for client in clients
        if client.category == "attacker" and "omniscient" in client.attributes
    ]

    # If no omniscient attackers exist, exit early
    if not omniscient_attackers:
        return
    # Generate malicious updates using the first attacker's logic
    malicious_updates = omniscient_attackers[0].omniscient(clients)
    if malicious_updates is None:
        raise ValueError("No updates generated by the omniscient attacker")

    # Check if the malicious update is a single vector or a batch of updates
    is_single_update = len(
        malicious_updates.shape) == 1 or malicious_updates.shape[0] == 1

    if is_single_update:
        # If a single update is provided, all attackers perform their own attack
        omniscient_attackers[0].update = malicious_updates
        for client in omniscient_attackers[1:]:
            client.update = client.omniscient(clients)
    else:
        # If multiple updates are provided, assign each update to an attacker
        # An attack method aiming to provide the same updates for all attackers can return repeated updates.
        for client, update in zip(omniscient_attackers, malicious_updates):
            client.update = update


def main(args, cli_args):
    """
    preprocess the arguments, logics, and run the federated learning process
    """
    # if Benchmarks is True, run all combinations of attacks and defenses
    if cli_args.benchmark:
        benchmark_preprocess(args)
        fl_run(args)
    else:
        override_args(args, cli_args)
        single_preprocess(args)
        fl_run(args)


if __name__ == "__main__":
    args, cli_args = read_args()
    main(args, cli_args)
