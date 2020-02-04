import torch.nn as nn
import torch.optim as optim

import sys
# sys.path.append('../vanilla_squeezenet/')
from models import *

def _get_model(model_name):
    model_dict = {
        'squeezenet': lambda: SqueezeNet(num_classes=10),
        'mobilenet': lambda: MyMobileNet(),
        'mobilenetv2': lambda: MobileNetV2(),
        'mobilenet_small': lambda: MyMobileNet(width_mul=.25),
        'fd_mobilenet': lambda: MyMobileNet(is_fd=True),
        'fd_mobilenet_small': lambda: MyMobileNet(width_mul=.25, is_fd=True),
        'mobilenetv3_small_x0.35': lambda: MobileNetV3(n_class=10, width_mult=.35),
        'mobilenetv3_small_x0.75': lambda: MobileNetV3(n_class=10, width_mult=.75),
        'mobilenetv3_impl2_small_x1.00': lambda: MobileNetV3Imp2(classes_num=10, input_size=32, width_multiplier=1.00, mode='small'),
        'mobilenetv3_impl2_small_x0.25': lambda: MobileNetV3Imp2(classes_num=10, input_size=32, width_multiplier=0.25, mode='small')
    }
    if model_name not in model_dict:
        print('model name not in dict')
        raise NotImplementedError
    return model_dict[model_name]()

    


'''
def get_model(learning_rate=1e-3, num_classes=200):

    model = SqueezeNet(num_classes=num_classes)
    
    # set the first layer not trainable
    model.features[0].weight.requires_grad = False

    # all conv layers except the first and the last
    all_conv_weights = [
        (n, p) for n, p in model.named_parameters()
        if 'weight' in n and not 'bn' in n and not 'features.1.' in n
    ]
    weights_to_be_quantized = [
        p for n, p in all_conv_weights
        if not ('classifier' in n or 'features.0.' in n)
    ]
    
    # the last layer
    weights = [model.classifier[1].weight]
    biases = [model.classifier[1].bias]
    
    # parameters of batch_norm layers
    bn_weights = [
        p for n, p in model.named_parameters()
        if ('bn' in n or 'features.1.' in n) and 'weight' in n
    ]
    bn_biases = [
        p for n, p in model.named_parameters()
        if ('bn' in n or 'features.1.' in n) and 'bias' in n
    ]

    params = [
        {'params': weights, 'weight_decay': 1e-4},
        {'params': weights_to_be_quantized},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.Adam(params, lr=learning_rate)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # move the model to gpu
    model = model.cuda()
    return model, criterion, optimizer
'''