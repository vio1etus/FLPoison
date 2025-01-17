import torch
from fl.algorithms import get_algorithm_handler
from fl.models import get_model
from fl.models.model_utils import vec2model
from fl.worker import Worker
from global_utils import actor, avg_value
from global_utils import TimingRecorder


@actor("benign", "always")
class Client(Worker):
    def __init__(self, args, worker_id, train_dataset, test_dataset=None):
        Worker.__init__(self, args, worker_id)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.global_epoch = 0
        # initialize the model
        self.model = get_model(args)
        # initialize the optimizer and lr_scheduler
        self.optimizer, self.lr_scheduler = self.get_optimizer_scheduler(
            self.model)

        self.train_loader = self.get_dataloader(
            self.train_dataset, train_flag=True)

        self.record_time(self.args.record_time)

    def record_time(self, record_time):
        if record_time:
            self.time_recorder = TimingRecorder(self.worker_id,
                                                self.args.output)
            self.local_training = self.time_recorder.timing_decorator(
                self.local_training)
            self.fetch_updates = self.time_recorder.timing_decorator(
                self.fetch_updates)
            self.omniscient = self.time_recorder.timing_decorator(
                self.omniscient) if hasattr(self, "omniscient") else None

    def set_algorithm(self, algorithm):
        self.algorithm = get_algorithm_handler(
            algorithm)(self.args, self.model, self.optimizer)
        # customize number of local epochs in client side
        self.local_epochs = self.algorithm.init_local_epochs()

    def load_global_model(self, global_weights_vec):
        self.global_weights_vec = global_weights_vec
        # load global parameters
        vec2model(self.global_weights_vec, self.model)

    def local_training(self, model=None, train_loader=None, optimizer=None, criterion_fn=None, local_epochs=None):
        """
        If you want to override the local training process in subclass, you need to take care of parameters, return values
        """
        # train
        model = self.new_if_given(model, self.model)
        train_loader = self.new_if_given(train_loader, self.train_loader)
        optimizer = self.new_if_given(optimizer, self.optimizer)
        criterion_fn = self.new_if_given(criterion_fn, self.criterion_fn)
        local_epochs = self.new_if_given(local_epochs, self.local_epochs)

        # change to iterator for infinite loop if dataloader is used
        train_iterator = iter(train_loader) if isinstance(
            train_loader, torch.utils.data.DataLoader) else train_loader
        model.train()
        acc_values, loss_values = [], []
        for epoch in range(local_epochs):
            acc, loss = self.train(model, train_iterator,
                                   optimizer, criterion_fn)
            acc_values.append(acc)
            loss_values.append(loss)
            self.step(optimizer, cur_local_epoch=epoch)
        self.lr_scheduler.step()

        return avg_value(acc_values), avg_value(loss_values)
        # client side debug usage
        # if "data_poisoning" in self.attributes:
        #     return self.client_asr_test()
        # else:
        #     return avg_value(acc_values), avg_value(loss_values)
        # return client_test()

    def fetch_updates(self, benign_flag=False):
        """produce the final client update for the server.
        benign_flag is used for benign update retrival
        """
        # update object is the trainable parameters or gradients according to the algorithm
        self.update = self.algorithm.get_local_update(
            global_weights_vec=self.global_weights_vec)
        if not benign_flag:
            # before submit, non_omniscient crafted the update maliciously
            if self.category == "attacker" and "non_omniscient" in self.attributes:
                self.update = self.non_omniscient()

        self.global_epoch += 1

    def client_test(self, model=None, test_dataset=None):
        """"
        Benign client test
        """
        model = self.new_if_given(model, self.model)
        test_dataset = self.new_if_given(test_dataset, self.test_dataset)
        test_loader = self.get_dataloader(test_dataset, train_flag=False)
        test_acc, test_loss = self.test(model, test_loader)
        return test_acc, test_loss
