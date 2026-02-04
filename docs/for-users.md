# For Users
For people who just want to run and use it.

## Getting Started

### Installation
There are no strict version requirements beyond Python >= 3.10. The project uses `pyproject.toml` and supports `uv`.

Install Python and create a virtual environment:

```bash
uv venv
```

Install dependencies from `pyproject.toml`:

```bash
uv pip install -e .
```

If you are not using `uv`, you can still use `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Notes:
1. PyTorch and CUDA-related packages are pinned in `pyproject.toml`. If you need a CPU-only setup, remove the CUDA packages and adjust the `torch`/`torchvision` pins.
2. For `unrar`, you may need a system package if the EdgeCase backdoor attack fails to extract archives.

Linux (Ubuntu):

```bash
sudo apt install unrar
```

macOS:

```bash
sudo brew install unrar
```

### Recommended Hardware
CPU-intensive, GPU memory-intensive, and FP32 operation-heavy hardware configurations are recommended for the benchmark experiment. A machine with 16+ CPU cores (for example, Intel Xeon Gold 6226R, Platinum 8352V, 8377C, AMD EPYC 7T83), 60GB+ memory, and GPUs with 24GB+ memory each with good FP32 performance (for example, NVIDIA 3090, A40, 4090, L20, L40s, L40) is ideal. These configurations ensure the experiment runs efficiently on a single machine.

## Simple Usage
The `./configs` folder contains several configuration files, such as `FedSGD_MNIST_config.yaml` and `FedSGD_CIFAR10_config.yaml`. Differences between these files may include dataset, model, learning rate and optimizer, batch size, and attack or defense configurations. You can modify these files to customize experiments.

For example, to run FedSGD on MNIST with LeNet:

```bash
python main.py -config=./configs/FedSGD_MNIST_config.yaml
```

### Run A Specific Attack Or Defense
Specify the attack and defense method in the configuration file. The corresponding attack or defense parameters in `attacks` or `defenses` are used if `attack_params` or `defense_params` is not specified.

Example:

```yaml
attack: IPM
attack_params:
  scaling_factor: 0.5
defense: Mean
```

Then run:

```bash
python main.py -config=./configs/FedSGD_MNIST_config.yaml
```

### Override Parameters With Command Line Arguments
You can override any parameter in the configuration file with command line arguments. For attack or defense parameters, you need to override the whole parameter object rather than part of it.

```bash
python main.py -config=./configs/FedSGD_MNIST_config.yaml -attack_params="{'scaling_factor': 0.5}"
```

If you only override `attack` or `defense` without overriding their parameters, the default parameters in the configuration file are used.

```bash
python main.py -config=./configs/FedSGD_MNIST_config.yaml -attack=MinSum
```

You can also override other parameters with `-model`, `-data`, `-num_clients`, `-num_adv`, `-bs`, `-lr`, `lr_scheduler`, and so on.

### Benchmark Mode
To run the benchmark experiment with one thread and process, use `-b`:

```bash
python main.py -config=./configs/FedSGD_MNIST_config.yaml -b=True
```

To run multiple experiments in parallel, use `batchrun.py`. For example:

```bash
python batchrun.py -algorithms FedSGD -data MNIST -model lenet -distributions non-iid -attacks MinMax MinSum -defenses Krum Median -gidx 1 -maxp 3
```

The above command trains LeNet-5 on MNIST with non-IID data partition on the FedSGD algorithm, and iteratively runs MinMax and MinSum with Krum and Median (4 experiments in total), using GPU index 1 with 3 parallel processes. It uses the default training, attack, and defense settings in the corresponding dataset and model configuration file `{FL algorithm}_{dataset}_config.yaml`.

## Parameters Setting
Two parameter passing methods are supported: command line options and configuration files.

### Command Line Options
There are two uses for command line options. You can use them to run the program and to override default parameters in configuration files.

### Configuration Files
YAML is used for parameter storage. The parameters should be stored in the `configs` folder as `.yaml` files.

We provide two types of configuration:
1. Dataset configuration: `dataset_config.yaml`
2. Experiment configuration: `{FL algorithm}_{dataset}_config.yaml`

`./configs/dataset_config.yaml` is the default dataset configuration. Normally, you do not need to modify this file unless you want to add a new dataset.

`./configs/FedSGD_MNIST_config.yaml` is for the FedSGD algorithm on the MNIST dataset. You can modify parameters in this file and customize the attack or defense experiment parameters. There are other configuration files for different FL algorithms and datasets, like `FedSGD_CIFAR10_config.yaml`, `FedSGD_CINIC10_config.yaml`, and so on.

## Key Points When Using FLPoison
Attack inheritance: `DPBase` should appear before `Client`, like `class A(DPBase, Client)`. This ensures that `A` uses the `DPBase.client_test()` method via inheritance.

## For Developers
Check out the documentation and source code for further development.
