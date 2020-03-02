import torch.nn as nn
import torch.optim as optim
from torch.nn.init import constant, kaiming_uniform
from fd_mobilenet_v3 import FdMobileNetV3Imp2


def get_model(learning_rate=1e-3, width_multiplier=0.32):

    model = FdMobileNetV3Imp2(classes_num=10, input_size=32, width_multiplier=width_multiplier, mode='small')

    # create different parameter groups
    weights = [
        p for n, p in model.named_parameters()
        if 'conv' in n or 'classifier.weight' in n
    ]
    biases = [model.classifier.bias]
    bn_weights = [
        p for n, p in model.named_parameters()
        if 'norm' in n and 'weight' in n
    ]
    bn_biases = [
        p for n, p in model.named_parameters()
        if 'norm' in n and 'bias' in n
    ]

    # parameter initialization
    for p in weights:
        kaiming_uniform(p)
    for p in biases:
        constant(p, 0.0)
    for p in bn_weights:
        constant(p, 1.0)
    for p in bn_biases:
        constant(p, 0.0)

    params = [
        {'params': weights, 'weight_decay': 1e-4},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.SGD(params, lr=1e-1, momentum=0.95, nesterov=True)

    loss = nn.CrossEntropyLoss().cuda()
    model = model.cuda()  # move the model to gpu
    return model, loss, optimizer
