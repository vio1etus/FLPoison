from global_utils import import_all_modules, Register
import os
from fl.models.vgg import add_vgg_from_torchvision

model_registry = Register()
import_all_modules(os.path.dirname(__file__), 1, "fl")
vgg_reg = add_vgg_from_torchvision()
model_registry.update(vgg_reg)
all_models = list(model_registry.keys())

model_categories = {
    "grey": ["lr"],
    "adaptive": ["lenet", "lenet_bn"],  # 1(grey) or 3(rgb) channels
    "rgb": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "vgg11", "vgg13", "vgg16", "vgg19"],
    "handy": ["simplecnn"]
}


def get_model(args):
    args.model = args.model.lower()
    if args.model not in all_models:
        raise NotImplementedError(
            f"Model not implemented, please choose from {all_models}")
    if args.model in model_categories["grey"]:
        assert args.num_channels == 1, "models designed for grey images only supports 1 channel"
        model = model_registry[args.model](
            input_dim=args.num_dims*args.num_dims, num_classes=args.num_classes)
    elif args.model in model_categories["adaptive"]:
        model = model_registry[args.model](
            num_channels=args.num_channels, num_classes=args.num_classes)
    elif args.model in model_categories["rgb"]:
        assert args.num_channels == 3, "models designed for RGB images only supports 3 channels"
        model = model_registry[args.model](num_classes=args.num_classes)
    elif args.model == "simplecnn":
        model = model_registry[args.model](input_size=(args.num_channels, args.num_dims, args.num_dims), num_classes=args.num_classes)
    else:
        raise NotImplementedError(
            f"Model not implemented, please choose from {all_models}")
    return model.to(args.device)
