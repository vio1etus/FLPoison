# Welcome to FLPoison

![Python Versions](https://img.shields.io/badge/Python-3.6%2B-blue)
![Last Commit](https://img.shields.io/github/last-commit/vio1etus/FLPoison)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

<!-- ![Repo Size](https://img.shields.io/github/repo-size/vio1etus/FLPoison) -->

Check the [Wiki](../../wiki#getting-started) to get started with the project.

## Features

PyTorch's implementation of poisoning attacks and defenses in federated learning.

|     **Category**      |                                                                                      **Details**                                                                                       |
| :-------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   **FL Algorithms**   |                                                               FedAvg, FedSGD, FedOpt(see [fl/algorithms](fl/algorithms))                                                               |
| **Data Distribution** | Balanced IID, Class-imbalanced IID, Quantity-imbalanced Dirichlet Non-IID, (Quantity-Balanced\|-Imbalanced) Pathological Non-IID (see [data_utils.py](datapreprocessor/data_utils.py)) |
|     **Datasets**      |                                      MNIST, FashionMNIST, EMNIST, CIFAR10, CINIC10, CIFAR100 (see [dataset_config.py](configs/dataset_config.py))                                      |
|      **Models**       |                                                           Logistic Regression, SimpleCNN, LeNet5, ResNet-series, VGG-series                                                            |

Supported datasets and models pairs see [datamodel.pdf](docs/datamodel.pdf)

## Federated Learning Algorithms

<!-- prettier-ignore -->
| Name | Source File |Paper|
|--|--|--|
|FedSGD|[fedsgd.py](fl/algorithms/fedsgd.py)|[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a) - AISTATS '17|
|FedAvg|[fedavg.py](fl/algorithms/fedavg.py)|[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a) - AISTATS '17|
|FedOpt|[fedopt.py](fl/algorithms/fedopt.py)|[Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295) - arxiv '20, ICLR '21|

## Attacks and Defenses

Applicable algorithms include base algorithms used by original paper, as well as others not explicitly mentioned but applicable based on the described principles. `[ ]` indicates necessary modifications for compatibility, also implemented within this framework. To sum up, we implemented and adapted the attacks and defenses to be compatible with three commonly-used FL algorithms, FedSGD, FedOpt, FedAvg.

### Data Poisoning Attacks (DPAs)

Data poisoning attacks here, mainly targeted attacks, refer to attacks aimed at **embedding backdoors or bias into the model**, thus misleading it to produce the attacker's intended prediction

<!-- | 3DFed | [threedfed.py](attackers/threedfed.py) | [3DFed: Adaptive and Extensible Framework for Covert Backdoor Attack in Federated Learning](https://ieeexplore.ieee.org/document/10179401) - S&P '23|  || -->

<!-- prettier-ignore -->
| Name | Source File | Paper | Base Algorithm | Applicable Algorithms |
|:---:|:---:|:---:|:---:|:---:|
| Neurotoxin | [neurotoxin.py](attackers/neurotoxin.py) | [Neurotoxin: Durable Backdoors in Federated Learning](https://proceedings.mlr.press/v162/zhang22w.html) - ICML '22 | FedOpt | FedOpt, [FedSGD, FedAvg] |
| Edge-case Backdoor | [edgecase.py](attackers/edgecase.py) | [Attack of the Tails: Yes, You Really Can Backdoor Federated Learning](https://arxiv.org/abs/2007.05084) - NeurIPS '20 | FedOpt |FedSGD, FedOpt, [FedAvg]|
| Model Replacement Attack (Scaling Attack) | [modelreplacement.py](attackers/modelreplacement.py) | [How to Backdoor Federated Learning](https://proceedings.mlr.press/v108/bagdasaryan20a.html) - AISTATS '20 |FedOpt |FedOpt, [FedSGD, FedAvg]|
| Alternating Minimization | [altermin.py](attackers/altermin.py) | [Analyzing Federated Learning Through an Adversarial Lens](https://arxiv.org/abs/1811.12470) - ICML '19 | FedOpt |FedSGD, FedOpt, [FedAvg]|
| DBA | [dba.py](attackers/dba.py) | [DBA: Distributed Backdoor Attacks Against Federated Learning](https://openreview.net/forum?id=rkgyS0VFvr) - ICLR '19 | FedOpt |FedSGD, FedOpt, [FedAvg]|
| BadNets | [badnets.py](attackers/badnets.py) | [BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain](https://arxiv.org/abs/1708.06733) - NIPS-WS '17 | Centralized ML |[FedSGD, FedOpt, FedAvg]|
| Label Flipping Attack | [labelflipping.py](attackers/labelflipping.py) | [Poisoning Attacks against Support Vector Machines](https://arxiv.org/abs/1206.6389) - ICML'12 | Centralized ML |[FedSGD, FedOpt, FedAvg]|

### Defenses Against DPAs

<!-- prettier-ignore -->
| Name | Source File | Paper | Base Algorithm | Applicable Algorithms |
|:---:|:---:|:---:|:---:|:---:|
| FLAME | [flame.py](aggregators/flame.py) | [FLAME: Taming Backdoors in Federated Learning](https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen) - USENIX Security '22 | FedOpt | FedOpt,[FedSGD, FedAvg]|
| DeepSight | [deepsight.py](aggregators/deepsight.py) | [DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection](https://arxiv.org/abs/2201.00763) - NDSS '22 | FedOpt |FedOpt, [FedSGD, FedAvg]|
| CRFL | [crfl.py](aggregators/crfl.py) | [CRFL: Certifiably Robust Federated Learning against Backdoor Attacks](http://proceedings.mlr.press/v139/xie21a/xie21a.pdf) - ICML '21| FedOpt | FedOpt, [FedSGD, FedAvg]|
| NormClipping | [normclipping.py](aggregators/normclipping.py) | [Can You Really Backdoor Federated Learning](https://arxiv.org/abs/1911.07963) - NeurIPS '20 | FedOpt |FedOpt, [FedSGD, FedAvg]|
| FoolsGold | [foolsgold.py](aggregators/foolsgold.py) | [The Limitations of Federated Learning in Sybil Settings](https://www.usenix.org/conference/raid2020/presentation/fung) - RAID '20 | FedSGD |FedSGD, [FedOpt, FedAvg]|
| Auror | [auror.py](aggregators/auror.py) | [Auror: Defending against poisoning attacks in collaborative deep learning systems](https://dl.acm.org/doi/10.1145/2991079.2991125) - ACSAC '16 | FedSGD |FedSGD, [FedOpt, FedAvg]|

### Model Poisoning Attacks (MPAs)

Model poisoning attacks here, main untargeted attacks, refer to the attacks aimed at **preventing convergence** of the model, thus affecting the model's performance.

<!-- prettier-ignore -->
| Name | Source File | Paper | Base Algorithm | Applicable Algorithms |
|:---:|:---:|:---:|:---:|:---:|
| Mimic Attack | [mimic.py](attackers/mimic.py) | [Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing](https://openreview.net/forum?id=jXKKDEi5vJt) - ICLR '22 | FedSGD | FedSGD, [FedOpt, FedAvg] |
| Min-Max attack | [min.py](attackers/min.py) | [Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning](https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/) - NDSS '21 | FedSGD | FedSGD, [FedOpt, FedAvg] |
| Min-Sum attack | [min.py](attackers/min.py) | [Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning](https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/) - NDSS '21 | FedSGD | FedSGD, [FedOpt, FedAvg] |
| Fang attack (Adaptive attack) | [fangattack.py](attackers/fangattack.py) | [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning](https://arxiv.org/abs/1911.11815) - USENIX Security '20 | FedAvg  | [FedSGD, FedOpt], FedAvg |
| IPM attack | [ipm.py](attackers/ipm.py) | [Fall of empires: Breaking Byzantine-tolerant SGD by inner product manipulation](https://proceedings.mlr.press/v115/xie20a.html) - UAI '20 | FedSGD | FedSGD, [FedOpt, FedAvg] |
| ALIE attack | [alie.py](attackers/alie.py) | [A Little Is Enough: Circumventing Defenses For Distributed Learning](https://proceedings.neurips.cc/paper_files/paper/2019/hash/ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html) - NeurIPS '19 | FedSGD | FedSGD, [FedOpt, FedAvg] |
| Sign flipping attack | [signflipping.py](attackers/signflipping.py) | [Asynchronous Byzantine machine learning (the case of SGD)](http://proceedings.mlr.press/v80/damaskinos18a/damaskinos18a.pdf) - ICML '18 |FedSGD| FedSGD, [FedOpt, FedAvg] |
| Gaussian (noise) attack | [gaussian.py](attackers/gaussian.py) | [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) - NeurIPS '17 |FedSGD| FedSGD, [FedOpt, FedAvg] |

### Defenses Against MPAs

<!-- prettier-ignore -->
| Name | Source File | Paper |  Base Algorithm | Applicable Algorithms |
|:---:|:---:|:---:|:---:|:---:|
| LASA |[lasa.py](aggregators/lasa.py)|[Achieving Byzantine-Resilient Federated Learning via Layer-Adaptive Sparsified Model Aggregation](https://arxiv.org/pdf/2409.01435) - WACV '25 | FedOpt | FedSGD, [FedOpt, FedAvg] |
| FLDetector | [fldetector.py](aggregators/fldetector.py) | [FLDetector: Defending Federated Learning Against Model Poisoning Attacks via Detecting Malicious Clients](https://arxiv.org/abs/2207.09209) - KDD '22 | FedSGD |FedOpt, [FedOpt, FedAvg]|
| SignGuard | [signguard.py](aggregators/signguard.py) | [Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering](https://arxiv.org/abs/2109.05872) - ICDCS '22 | FedSGD |FedSGD, [FedOpt, FedAvg]|
| Bucketing | [bucketing.py](aggregators/bucketing.py) | [Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing](https://openreview.net/forum?id=jXKKDEi5vJt) - ICLR '22 | FedSGD | FedSGD, [FedOpt, FedAvg] |
| DnC | [dnc.py](aggregators/dnc.py) | [Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning](https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/) - NDSS '21 | FedSGD |FedSGD, [FedOpt, FedAvg]|
| CenteredClipping | [centeredclipping](aggregators/centeredclipping.py) | [Learning from History for Byzantine Robust Optimization](https://arxiv.org/abs/2012.10333) - ICML '21 | FedSGD |FedSGD, [FedOpt, FedAvg]|
| FLTrust | [fltrust.py](aggregators/fltrust.py) | [FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping](https://arxiv.org/abs/2012.13995) - ArXiv'20, NDSS '21 | FedOpt |FedOpt, [FedSGD, FedAvg]|
| RFA (Geometric Median) | [rfa.py](aggregators/rfa.py) | [Robust Aggregation for Federated Learning](https://ieeexplore.ieee.org/document/9721118) - ArXiv'19, TSP '22 | FedAvg | [FedSGD, FedOpt], FedAvg|
| Bulyan | [bulyan.py](aggregators/bulyan.py) | [The hidden vulnerability of distributed learning in Byzantium](https://arxiv.org/abs/1802.07927) - ICML'18 | FedSGD |FedSGD, [FedOpt,FedAvg]|
| Coordinate-wise Median | [median.py](aggregators/median.py) | [Byzantine-robust distributed learning: Towards optimal statistical rates](https://proceedings.mlr.press/v80/yin18a.html) - ICML'18 | FedSGD | FedSGD, [FedOpt, FedAvg]|
| Trimmed Mean | [trimmedmean.py](aggregators/trimmedmean.py) | [Byzantine-robust distributed learning: Towards optimal statistical rates](https://proceedings.mlr.press/v80/yin18a.html) - ICML'18 | FedSGD | FedSGD, [FedOpt, FedAvg] |
| Multi-Krum | [multikrum.py](aggregators/multikrum.py) | [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) - NeurIPS '17 | FedSGD | FedSGD, [FedOpt, FedAvg] |
| Krum | [krum.py](aggregators/krum.py) | [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) - NeurIPS '17 | FedSGD | FedSGD, [FedOpt, FedAvg] |
|SimpleClustering|[simpleclustering.py](aggregators/simpleclustering.py)|Simple majority-based clustering| FedSGD, FedAvg, FedOpt|FedSGD, FedAvg, FedOpt|

## Contributing

Bug reports, feature suggestions, and code contributions are welcome. Please open an issue or submit a pull request if you encounter any problems or have suggestions.

## Citation

If you are using FLPoison for your work, please cite our paper with:

```
@misc{sokflpoison,
      title={SoK: Benchmarking Poisoning Attacks and Defenses in Federated Learning}, 
      author={Heyi Zhang and Yule Liu and Xinlei He and Jun Wu and Tianshuo Cong and Xinyi Huang},
      year={2025},
      eprint={2502.03801},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2502.03801}, 
}
```
