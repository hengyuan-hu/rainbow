"""Common functions you may find useful in your implementation."""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


_FLOAT32_MAX = np.finfo(np.float32).max
_HALF_LOG_MAX = float(np.log(_FLOAT32_MAX) / 2)


def softmax(x, dim):
    a = x.max(dim, keepdim=True)[0] - _HALF_LOG_MAX
    x = x - a.expand_as(x)
    exp_x = torch.exp(x)
    sum_exp = exp_x.sum(dim, keepdim=True).expand_as(exp_x)
    return exp_x / sum_exp


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_zero_grads(params):
    for p in params:
        if p.grad is not None:
            assert_eq(p.grad.data.sum(), 0)


def assert_frozen(module):
    for p in module.parameters():
        assert not p.requires_grad


def weights_init(m):
    """custom weights initialization called on net_g and net_f."""
    classtype = m.__class__
    if classtype == nn.Linear or classtype == nn.Conv2d:
        m.weight.data.normal_(0.0, 0.02)
    elif classtype == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print '%s is not initialized.' % classtype


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def count_output_size(input_shape, module):
    fake_input = Variable(torch.FloatTensor(*input_shape), volatile=True)
    output_size = module.forward(fake_input).view(-1).size()[0]
    return output_size


def one_hot(x, n):
    assert x.dim() == 2
    one_hot_x = torch.zeros(x.size(0), n).cuda()
    one_hot_x.scatter_(1, x, 1)
    return one_hot_x
