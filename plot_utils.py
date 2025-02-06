import re
import matplotlib.pyplot as plt
import numpy as np


def parse_logs(filename):
    plt.clf()
    # read log file
    with open(filename, 'r') as f:
        content = f.read()
    epochs, accs, losses, asrs, asr_losses = [], [], [], [], []
    # regular expression pattern to extract epoch, test accuracy, test loss, asr, asr loss
    regex = (
        r"Epoch (?P<epoch>\d+)\s.*?Test Acc: (?P<test_acc>[\d\.]+)\s.*?Test loss: (?P<test_loss>[\d\.]+)(?:\s.*?ASR: (?P<asr>[\d\.]+))?(?:\s.*?ASR loss: (?P<asr_loss>[\d\.]+))?"
    )

    for match in re.finditer(regex, content):
        epochs.append(int(match.group('epoch')))
        accs.append(float(match.group('test_acc')))
        losses.append(float(match.group('test_loss')))

        # if asr and asr loss exist, add them, or add None
        asr = match.group('asr')
        asr_loss = match.group('asr_loss')
        asrs.append(float(asr) if asr else None)
        asr_losses.append(float(asr_loss) if asr_loss else None)

    return epochs, accs, losses, asrs, asr_losses


def plot_accuracy(filename):
    epochs, accs, _, asr, _ = parse_logs(filename)

    plt.plot(epochs, accs, label='Accuracy')

    # if asr statistics exists, plot asr curve
    if any(asr):
        plt.plot(epochs, asr, label='ASR', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename[:-4] + ".png")


def plot_label_distribution(train_data, client_idcs, n_clients, dataset, distribution):
    titleid_dict = {"iid": "Balanced_IID", "class-imbalanced_iid": "Class-imbalanced_IID",
                    "non-iid": "Quantity-imbalanced_Dirichlet_Non-IID", "pat": "Balanced_Pathological_Non-IID", "imbalanced_pat": "Quantity-imbalanced_Pathological_Non-IID"}
    dataset = "CIFAR-10" if dataset == "CIFAR10" else dataset
    title_id = dataset + " " + titleid_dict[distribution]
    xy_type = "client_label"  # 'label_client'
    plt.rcParams['font.size'] = 14  # set global fontsize
    # set the direction of xtick toward inside
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    labels = train_data.targets
    n_classes = labels.max()+1
    plt.figure(figsize=(12, 8))
    if xy_type == "client_label":
        label_distribution = [[] for _ in range(n_classes)]
        for c_id, idc in enumerate(client_idcs):
            for idx in idc:
                label_distribution[labels[idx]].append(c_id)

        plt.hist(label_distribution, stacked=True,
                 bins=np.arange(-0.5, n_clients + 1.5, 1),
                 label=range(n_classes), rwidth=0.5, zorder=10)
        plt.xticks(np.arange(n_clients), ["%d" %
                                          c_id for c_id in range(n_clients)])
        plt.xlabel("Client ID", fontsize=20)
    elif xy_type == "label_client":
        plt.hist([labels[idc]for idc in client_idcs], stacked=True,
                 bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),
                 label=["Client {}".format(i) for i in range(n_clients)],
                 rwidth=0.5, zorder=10)
        plt.xticks(np.arange(n_classes), train_data.classes)
        plt.xlabel("Label type", fontsize=20)

    plt.ylabel("Number of Training Samples", fontsize=20)
    plt.title(f"{title_id} Label Distribution Across Clients", fontsize=20)
    rotation_degree = 45 if n_clients > 30 else 0
    plt.xticks(rotation=rotation_degree, fontsize=16)
    plt.legend(loc="best", prop={'size': 12}).set_zorder(100)
    plt.grid(linestyle='--', axis='y', zorder=0)
    plt.tight_layout()
    plt.savefig(f"./logs/{title_id}_label_dtb.pdf",
                format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    # multiple()
    plot_accuracy(
        "./logs/FedOpt/MNIST_lenet/iid/MNIST_lenet_iid_DBA_DeepSight_500_50_0.01_FedOpt.txt")
