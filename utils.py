"""Common functions you may find useful in your implementation."""
import torch
import torch.nn as nn
from torch.autograd import Variable


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
