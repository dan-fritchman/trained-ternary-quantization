import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import re

from mobilenet_v3 import MobileNetV3

def get_model(learning_rate=1e-3, num_classes=200, input_size=32):

    model = MobileNetV3(n_class=num_classes, input_size=input_size)

    # find all weights first,
    # exclude last weights for params.weights
    # rest are for weights_to_be_quantized

    # find all weights
    all_conv_weights = [
        (n, p) for n, p in model.named_parameters()
        if 'weight' in n and 'features.14.' not in n and not re.match(r'features\.(\d+)\.conv\.(1|4|8)', n) and not ('features.0.1' in n or 'features.12.1' in n)
    ]

    all_conv_biases = [
        (n, p) for n, p in model.named_parameters()
        if 'bias' in n and 'features.14.' not in n and not re.match(r'features\.(\d+)\.conv\.(1|4|8)', n) and not ('features.0.1' in n or 'features.12.1' in n)
    ]

    # I'm assuming we draw weights and biases from the last convolutional layer
    weights = all_conv_weights
    biases = all_conv_biases

    bn_weights = [
        p for n, p in model.named_parameters()
        if 'weight' in n and not 'features.14.' in n and ('features.0.1' in n or 'features.12.1' in n
        or re.match(r'features\.(\d+)\.conv\.(1|4|8)', n))
    ]
    bn_biases = [
        p for n, p in model.named_parameters()
        if 'bias' in n and not 'features.14.' in n and ('features.0.1' in n or 'features.12.1' in n
        or re.match(r'features\.(\d+)\.conv\.(1|4|8)', n))
    ]

    params = [
        {'params': weights, 'weight_decay': 3e-4},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.Adam(params, lr=learning_rate)

    loss = nn.CrossEntropyLoss() #.cuda()
    # model = model.cuda()  # move the model to gpu
    return model, loss, optimizer
