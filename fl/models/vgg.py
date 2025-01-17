from torchvision import models


def add_vgg_from_torchvision():
    """
    Add all vgg models from torchvision to model_registry
    """
    reg = {}
    # register vgg models to model_registry
    vgg_models = [i for i in models.vgg.__dict__.keys() if i.startswith('vgg')]
    for model_name in vgg_models:
        model_fn = getattr(models, model_name)  # get model function, address
        reg[model_name.lower()] = model_fn
    return reg
