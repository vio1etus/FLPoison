import torch


class Worker:
    def __init__(self, args, worker_id):
        """worker_id: int, the id of the worker. For client, it should be positive int value; for server, it should be -1.
        """
        self.args = args
        self.worker_id = worker_id
        self.synthesizer = None  # To be customized in subclass

    def __str__(self):
        return f"worker id: {self.worker_id}"

    def get_dataloader(self, dataset, train_flag=True, **kwargs):
        """
        Train poison will shuffle and do poisoning attacks on some samples according to poisoning_ratio;
        Train No poison will shuffle and will not do poisoning attacks
        Test poison will not shuffle and will poisoning all samples
        Test no poison will not shuffle and will not poisoning
        """
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=train_flag, num_workers=self.args.num_workers, pin_memory=True)
        if train_flag:
            while True:  # add infinite loop for training epoch, because dataloader will be consumed after one dataset iteration
                for images, targets in dataloader:
                    yield images, targets
        else:  # test mode for test dataset
            # return dataloader
            for images, targets in dataloader:
                yield images, targets

    def criterion_fn(self, y_pred, y_true, **kwargs):
        return torch.nn.CrossEntropyLoss()(y_pred, y_true)

    def new_if_given(self, value, default):
        return default if value is None else value

    def train(self, model, train_iterator, optimizer, criterion_fn=None):
        criterion_fn = self.new_if_given(criterion_fn, self.criterion_fn)
        optimizer.zero_grad()
        images, targets = next(train_iterator)
        images, targets = images.to(
            self.args.device), targets.to(self.args.device)
        pred_probs = model(images)
        loss = criterion_fn(pred_probs, targets)
        loss.backward()
        predicted = torch.argmax(pred_probs.data, 1)
        train_acc = (predicted == targets).sum().item()
        train_loss = loss.item()
        train_loss /= len(images)
        train_acc /= len(images)
        return train_acc, train_loss

    def step(self, optimizer, **kwargs):
        optimizer.step()

    def test(self, model, test_loader, imbalanced=False):
        model.eval()
        tail_cls_from = self.args.tail_cls_from if imbalanced else 0
        overall_correct, test_loss, num_samples = 0, 0, 0
        rest_correct, rest_samples = 0, 0

        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(
                    self.args.device), targets.to(self.args.device)
                pred_probs = model(images)
                loss = self.criterion_fn(pred_probs, targets)
                predicted = torch.argmax(pred_probs.data, 1)
                num_samples += len(targets)
                overall_correct += (predicted == targets).sum().item()
                test_loss += loss.item()

                if imbalanced:
                    # calculate the rest class accuracy, from 5-9 for 10 classes
                    rest_mask = targets >= tail_cls_from
                    rest_correct += (predicted[rest_mask]
                                     == targets[rest_mask]).sum().item()
                    rest_samples += rest_mask.sum().item()

        overall_accuracy = overall_correct / num_samples
        test_loss /= num_samples

        if imbalanced:
            rest_accuracy = rest_correct / rest_samples if rest_samples > 0 else 0
            return overall_accuracy, rest_accuracy, test_loss

        return overall_accuracy, test_loss

    def get_optimizer_scheduler(self, model, learning_rate=None, weight_decay=None):
        learning_rate = self.new_if_given(
            learning_rate, self.args.learning_rate)
        weight_decay = self.new_if_given(self.args.weight_decay, weight_decay)
        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if hasattr(self.args, 'lr_scheduler') and self.args.lr_scheduler is not None:
            if self.args.lr_scheduler == 'MultiStepLR':
                milestones = [int(i*self.args.epochs) if i <
                              1 else i for i in self.args.milestones]
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=milestones, gamma=0.1)
            elif self.args.lr_scheduler == 'StepLR':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
            elif self.args.lr_scheduler == 'ExponentialLR':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.9)
            elif self.args.lr_scheduler == "CosineAnnealingLR":
                if self.args.algorithm == "FedSGD":
                    total_epoch = self.args.epochs
                elif self.args.algorithm in ["FedOpt", "FedAvg"]:
                    total_epoch = self.args.epochs * self.args.local_epochs
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)
            else:
                raise NotImplementedError(f"{self.args.lr_scheduler} is not implemented currently.")
        else:
            # keep lr constant for all epochs
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 1.0)

        return optimizer, lr_scheduler

    def cycle(self, dataloader):
        """
        useful when the dataloader is consumed out after one epoch
        """
        while True:
            for images, targets in dataloader:
                yield images, targets
