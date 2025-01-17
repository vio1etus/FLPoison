from copy import deepcopy
import numpy as np
import torch

# ======The two main APIs for model and 1-d numpy array conversion======


def vec2model(vector, model, plus=False, ignorebn=False):
    """
    in-place modification of model's parameters
    Convert a 1d-numpy array back into a model
    """
    curr_idx = 0
    model_state_dict = model.state_dict()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for key, value in model_state_dict.items():
        if ignorebn:
            if any(substring in key for substring in ['running_mean', 'running_var', 'num_batches_tracked']):
                continue
        numel = value.numel()
        param_tensor = torch.from_numpy(
            vector[curr_idx:curr_idx + numel].reshape(value.shape)).to(device=device, dtype=dtype)

        if plus:
            value.copy_(value + param_tensor)  # in-place addition
        else:
            value.copy_(param_tensor)  # in-place assignment
        curr_idx += numel

    # Note that the below method are only suitable for CNN without batch normalization layer
    # vector2parameter(vector, model)


def model2vec(model):
    """
    convert the model's state dict to a 1d numpy array
    """
    return state2vec(model.state_dict())
    # return parameter2vector(model)


def add_vec2model(vector, model_template):
    """
    Add the state dict vector-form (pseudo) gradient to the model's parameters and return a new model
    """
    tmp_model = deepcopy(model_template)
    vec2model(vector, tmp_model, plus=True)
    return tmp_model

# ======Below are specific implementations======


def vector2parameter(vector, model):
    """
    in-place modification of iterable model_parameters's data
    """
    current_pos = 0
    for param in model.parameters():
        numel = param.numel()  # get the number of elements in param
        param.data = torch.from_numpy(
            vector[current_pos:current_pos + numel].reshape(param.shape)).to(param.device)
        current_pos += numel


def parameter2vector(model):
    # numpy() will convert torch.float32 to np.float32
    model_parameters = model.parameters()
    return np.concatenate([param.detach().cpu().numpy().flatten() for param in model_parameters])


def set_grad_none(model):
    """
    Set the gradient of all parameters to None
    """
    for param in model.parameters():
        param.grad = None


def vector2gradient(vector, model):
    """
    in-place modification of iterable model_parameters's grad
    """
    current_pos = 0
    parameters = model.parameters()
    for param in parameters:
        numel = param.numel()  # get the number of elements in param
        param.grad = torch.from_numpy(
            vector[current_pos:current_pos + numel].reshape(param.shape)).to(param.device)
        current_pos += numel


def gradient2vector(model):
    """
    Convert gradients to a concatenated 1D numpy array
    """
    parameters = model.parameters()
    return np.concatenate([param.grad.cpu().numpy().flatten() for param in parameters])


def ol_from_vector(vector, model_template, flatten=True, return_type='dict'):
    state_template = model_template.state_dict()
    # Get keys for the last two layers (weight and bias)
    output_layer_keys = list(state_template.keys())[-2:]

    # Get the shapes of the weight and bias
    weight_shape = state_template[output_layer_keys[0]].shape
    bias_shape = state_template[output_layer_keys[1]].shape

    # Calculate sizes
    weight_size = np.prod(weight_shape)
    bias_size = np.prod(bias_shape)

    # Start with the last element of the vector for bias, then weight
    bias = vector[-bias_size:
                  ] if flatten else vector[-bias_size:].reshape(bias_shape)
    weights = vector[-(bias_size + weight_size):-bias_size] if flatten else vector[-(bias_size + weight_size):-
                                                                                   bias_size].reshape(weight_shape)
    if return_type == 'dict':
        return {'weight': weights, 'bias': bias}
    elif return_type == 'vector':
        # !DON'T change the order of weights and bias, as it's the order of the output layer and the order of the state_dict vector
        if flatten:  # concatenate 1d vectors
            return np.concatenate([weights.flatten(), bias.flatten()])
        else:
            # concatenate the weights and bias at axis 1, i.e., column-wise, to produce a 2d array with same number of rows and added bias columns
            return np.concatenate([weights, bias.reshape(bias_size, -1)], axis=1)


def ol_from_model(model, flatten=True, return_type='dict'):
    return ol_from_vector(model2vec(model), model,
                          flatten=flatten, return_type=return_type)


def vec2state(vector, model, plus=False, ignorebn=False, numpy=False):
    """
    Convert a 1d-numpy array to the state dict-form of the model
    return a new state dict
    """
    curr_idx = 0
    model_state_dict = deepcopy(model.state_dict())
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for key, value in model_state_dict.items():
        if ignorebn:
            if any(substring in key for substring in ['running_mean', 'running_var', 'num_batches_tracked']):
                continue
        numel = value.numel()
        param_tensor = torch.from_numpy(
            vector[curr_idx:curr_idx + numel].reshape(value.shape)).to(device=device, dtype=dtype)

        if plus:
            value.copy_(value + param_tensor)  # in-place addition
        else:
            value.copy_(param_tensor)  # in-place assignment
        curr_idx += numel
    if numpy:
        return {key: value.detach().cpu().numpy() for key, value in model_state_dict.items()}

    return model_state_dict


def state2vec(model_state_dict, ignorebn=False, numpy_flg=False):
    """
    Convert a state dict to a concatenated 1D numpy array.
    """
    # Collect all the tensors first
    if numpy_flg:
        arrays = [
            value.flatten()
            for name, value in model_state_dict.items()
            if (True if not ignorebn else all(substring not in name for substring in ['running_mean', 'running_var', 'num_batches_tracked']))
        ]
    else:
        arrays = [
            i.detach().cpu().numpy().flatten()
            for name, i in model_state_dict.items()
            if (True if not ignorebn else all(substring not in name for substring in ['running_mean', 'running_var', 'num_batches_tracked']))
        ]

    # Concatenate the list of arrays at once
    return np.concatenate(arrays)
