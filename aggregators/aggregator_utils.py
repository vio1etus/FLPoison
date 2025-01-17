"""
This file contains utility functions shared by multiple aggregators
Methods used by only one aggregate function should not be placed here.
"""
from collections import defaultdict
from copy import deepcopy
import numpy as np
from fl.models.model_utils import add_vec2model, model2vec, vec2model


def L2_distances(updates):
    distances = defaultdict(dict)
    for i in range(len(updates)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(
                updates[i] - updates[j])
    return distances


def krum_compute_scores(distances, i, n, f):
    """Compute scores for node i.
    Returns:
        float -- krum distance score of i.
    """
    _s = sorted([dist for dist in distances[i].values()])[:n-f-1]
    return sum(_s)

def prepare_grad_updates(algorithm, updates, global_model):

    num_updates = len(updates)  # equal to num_clients
    # gradient_updates
    gradient_updates = updates if "FedSGD" or "FedOpt" in algorithm else np.array(
            [updates[cid] - model2vec(global_model) for cid in range(num_updates)])
    
    return gradient_updates
   
def prepare_updates(algorithm, updates, global_model, vector_form=True):
    """
    receives the np array updates and the global model and returns the model updates and gradient updates
    Args:
    model updates: model-form updated global model of each client for FedAvg
    gradient_updates: vector-form (pseudo-)gradient updates for FedSGD, FedOpt
    """
    num_updates = len(updates)  # equal to num_clients
    # gradient_updates
    if algorithm == 'FedAvg':
        vec_updates = updates
        gradient_updates = np.array(
            [updates[cid] - model2vec(global_model) for cid in range(num_updates)])

    elif "FedSGD" or "FedOpt" in algorithm:
        gradient_updates = updates
        vec_updates = np.array(
            [model2vec(global_model) + updates[cid] for cid in range(num_updates)])

    if vector_form:
        # vector_form return 1d np array vector model parameters
        model_updates = vec_updates
    else:
        # model-form model_updates
        model_updates = []
        for cid in range(num_updates):
            tmp = deepcopy(global_model)
            vec2model(vec_updates[cid], tmp)
            model_updates.append(tmp)
        model_updates = np.array(model_updates)

    return model_updates, gradient_updates


def wrapup_aggregated_grads(benign_grad_updates, algorithm, global_model, aggregated=False):
    """
    wrap up the aggregated result based on the algorithm type, and selected benign gradient updates, and return the aggregated result, For FedAvg, updated model parameters, for FedSGD, FedOpt, gradients update
    Args:
    benign_grad_updates: the gradient updates of slected as benign
    algorithm: the type of algorithm
    global_model: the global model
    """
    aggregated_gradient = benign_grad_updates if aggregated else np.mean(
        benign_grad_updates, axis=0)
    if algorithm == 'FedAvg':
        aggregated_model = add_vec2model(
            aggregated_gradient, global_model)
        return model2vec(aggregated_model)
    else:
        return aggregated_gradient


def normclipping(vectors, threshold, epsilon=1e-6):
    """ clipping the 2d-vectors based on the threshold
    Args:
        2d vectors (numpy.ndarray): the vectors from clients
    """
    if len(vectors.shape) != 2:
        raise ValueError(
            "The input should be 2d vectors, or you need to extend this function")
    return vectors * np.minimum(1, threshold / (np.linalg.norm(vectors, axis=1)+epsilon)).reshape(-1, 1)


def addnoise(vector, noise_mean, noise_std):
    """ add gaussian noise to the vector, z~N(0, sigma^2 * I)
    """
    # generate gaussian noise, note that the noise should be float32 to be consistent with the future torch dtype
    noise = np.random.normal(noise_mean, noise_std,
                             vector.shape).astype(np.float32)
    return vector + noise
