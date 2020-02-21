import torch.nn as nn
import torch.optim as optim
from torch.nn.init import constant, kaiming_uniform
from models import *

def get_model_from_dict(model_name):
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

def get_model(model_name, learning_rate=1e-3):

    model = get_model_from_dict(model_name)
    
    # set the first layer not trainable
    # create different parameter groups
    #return list(model.named_parameters())
    
    # create different parameter groups

    # classifier (last layer (linear))

     # set the first layer not trainable
    model.features.conv0.weight.requires_grad = False

    # the last fc layer
    weights = [
        p for n, p in model.named_parameters()
        if 'classifier.weight' in n
    ]
    biases = [model.classifier.bias]
    
    # all conv layers except the first
    weights_to_be_quantized = [
        p for n, p in model.named_parameters()
        if 'conv' in n and ('dense' in n or 'transition' in n)
    ]
    
    # parameters of batch_norm layers
    bn_weights = [
        p for n, p in model.named_parameters()
        if 'norm' in n and 'weight' in n
    ]
    bn_biases = [
        p for n, p in model.named_parameters()
        if 'norm' in n and 'bias' in n
    ]

    params = [
        {'params': weights, 'weight_decay': 1e-4},
        {'params': weights_to_be_quantized},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.Adam(params, lr=learning_rate)

    loss = nn.CrossEntropyLoss().cuda()
    model = model.cuda()  # move the model to gpu
    return model, loss, optimizer
