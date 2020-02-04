import torch.nn as nn
import torch.optim as optim
from torch.nn.init import constant
from models import * 

model_dict = {
    'mobilenet': lambda: MobileNet(),
    'mobilenetv2': lambda: MobileNetV2(),
    'mobilenet_small': lambda: MyMobileNet(width_mul=.25),
    'fd_mobilenet_small': lambda: MyMobileNet(width_mul=.25, is_fd=True),
    'mobilenetv3_small_x0.35': lambda: MobileNetV3(n_class=10, width_mult=.35),
    'mobilenetv3_small_x0.75': lambda: MobileNetV3(n_class=10, width_mult=.75),
}

'''
def get_model(num_classes=200):

    model = FdMobileNet(num_classes=num_classes)

    # create different parameter groups
    weights = [
        p for n, p in model.named_parameters()
        if 'weight' in n and 'bn' not in n and 'features.1.' not in n
    ]
    biases = [model.classifier[1].bias]
    bn_weights = [
        p for n, p in model.named_parameters()
        if ('bn' in n or 'features.1.' in n) and 'weight' in n
    ]
    bn_biases = [
        p for n, p in model.named_parameters()
        if ('bn' in n or 'features.1.' in n) and 'bias' in n
    ]

    for p in bn_weights:
        constant(p, 1.0)
    for p in bn_biases:
        constant(p, 0.0)

    params = [
        {'params': weights, 'weight_decay': 3e-4},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.SGD(params, lr=4e-2, momentum=0.95, nesterov=True)

    loss = nn.CrossEntropyLoss().cuda()
    model = model.cuda()  # move the model to gpu
    return model, loss, optimizer
'''
print(model_dict['mobilenetv3_small_x0.75']())