import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mobilenet_v3 import MobileNetV3

def get_model(learning_rate=1e-3, num_classes=200):

    model = MobileNetV3(n_class=num_classes, input_size=32)

    # set the first layer not trainable
    model.features[0][0].weight.requires_grad = False

    # find all weights first,
    # exclude last weights for params.weights
    # rest are for weights_to_be_quantized

    # find all weights
    all_conv_weights = [
        (n, p) for n, p in model.named_parameters()
        if 'weight' in n and 'features.14.' not in n and not re.match(r'features\.(\d+)\.conv\.(1|4|8)', n)
    ]
    weights_to_be_quantized = [
        p for n, p in all_conv_weights
        if not ('features.0.0' in n)
    ]

    # I'm assuming we draw weights and biases from the last convolutional layer
    weights = [model.features[14].weight]
    biases = [model.features[14].bias]

    bn_weights = [
        p for n, p in model.named_parameters()
        if 'weight' in n and ('features.14.' in n or 'features.0.1' in n or 'features.12.1' in n
        or re.match(r'features\.(\d+)\.conv\.(1|4|8)', n))
    ]
    bn_biases = [
        p for n, p in model.named_parameters()
        if 'bias' in n and ('features.14.' in n or 'features.0.1' in n or 'features.12.1' in n
        or re.match(r'features\.(\d+)\.conv\.(1|4|8)', n))
    ]

    params = [
        {'params': weights, 'weight_decay': 3e-4},
        {'params': weights_to_be_quantized},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.Adam(params, lr=learning_rate)

    loss = nn.CrossEntropyLoss() #.cuda()
    # model = model.cuda()  # move the model to gpu
    return model, loss, optimizer
