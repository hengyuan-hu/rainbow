import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils


class BasicNetwork(nn.Module):
    def __init__(self, conv, fc):
        super(BasicNetwork, self).__init__()
        self.conv = conv
        self.fc = fc

    def forward(self, x):
        assert x.data.max() <= 1.0
        batch = x.size(0)
        y = self.conv(x)
        y = y.view(batch, -1)
        y = self.fc(y)
        return y


class DuelingNetwork(nn.Module):
    def __init__(self, conv, adv, val):
        super(DuelingNetwork, self).__init__()
        self.conv = conv
        self.adv = adv
        self.val = val

    def forward(self, x):
        assert x.data.max() <= 1.0
        batch = x.size(0)
        feat = self.conv(x)
        feat = feat.view(batch, -1)
        adv = self.adv(feat)
        val = self.val(feat)
        q = val - adv.mean(1, keepdim=True) + adv
        return q


# TODO: DistributionalDuelingNetwork
class DistributionalBasicNetwork(nn.Module):
    def __init__(self, conv, fc, num_actions, num_atoms):
        super(DistributionalBasicNetwork, self).__init__()
        self.conv = conv
        self.fc = fc
        self.num_actions = num_actions
        self.num_atoms = num_atoms

    def forward(self, x):
        batch = x.size(0)
        y = self.conv(x)
        y = y.view(batch, -1)
        y = self.fc(y)
        logits = y.view(batch, self.num_actions, self.num_atoms)
        probs = nn.functional.softmax(logits, 2)
        return probs


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet"""
    def __init__(self, in_features, out_features, sigma0):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.noise_std = sigma0 / math.sqrt(self.in_features)
        self.in_noise = torch.FloatTensor(in_features).cuda()
        self.out_noise = torch.FloatTensor(out_features).cuda()
        self.noise = None
        self.sample_noise()

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        normal_y =  nn.functional.linear(x, self.weight, self.bias)
        if not x.volatile:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * Variable(self.noise)
        noisy_bias = self.noisy_bias * Variable(self.out_noise)
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


# ---------------------------------------


def _build_default_conv(in_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, 32, 8, 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU()
    )
    return conv


def _build_fc(dims):
    layers = [nn.Linear(dims[0], dims[1])]
    for i in range(1, len(dims) - 1):
        layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[i], dims[i+1]))

    fc = nn.Sequential(*layers)
    return fc


def _build_noisy_fc(dims, sigma0):
    layers = [NoisyLinear(dims[0], dims[1], sigma0)]
    for i in range(1, len(dims) - 1):
        layers.append(nn.ReLU())
        layers.append(NoisyLinear(dims[i], dims[i+1], sigma0))

    fc = nn.Sequential(*layers)
    return fc


def build_basic_network(in_channels, in_size, out_dim, noisy, sigma0, net_file):
    conv = _build_default_conv(in_channels)

    in_shape = (1, in_channels, in_size, in_size)
    fc_in = utils.count_output_size(in_shape, conv)
    fc_hid = 512
    dims = [fc_in, fc_hid, out_dim]
    if noisy:
        fc = _build_noisy_fc(dims, sigma0)
    else:
        fc = _build_fc(dims)

    net = BasicNetwork(conv, fc)
    utils.init_net(net, net_file)
    return net


def build_dueling_network(in_channels, in_size, out_dim, noisy, sigma0, net_file):
    conv = _build_default_conv(in_channels)

    in_shape = (1, in_channels, in_size, in_size)
    fc_in = utils.count_output_size(in_shape, conv)
    fc_hid = 512
    adv_dims = [fc_in, fc_hid, out_dim]
    val_dims = [fc_in, fc_hid, 1]

    if noisy:
        adv = _build_noisy_fc(adv_dims, sigma0)
        val = _build_noisy_fc(val_dims, sigma0)
    else:
        adv = _build_fc(adv_dims)
        val = _build_fc(val_dims)

    net = DuelingNetwork(conv, adv, val)
    utils.init_net(net, net_file)
    return net


def build_distributional_basic_network(
        in_channels, in_size, out_dim, num_atoms, noisy, sigma0, net_file):

    conv = _build_default_conv(in_channels)

    in_shape = (1, in_channels, in_size, in_size)
    fc_in = utils.count_output_size(in_shape, conv)
    fc_hid = 512
    fc_dims = [fc_in, fc_hid, out_dim * num_atoms]
    if noisy:
        fc = _build_noisy_fc(fc_dims, sigma0)
    else:
        fc = _build_fc(fc_dims)

    net = DistributionalBasicNetwork(conv, fc, out_dim, num_atoms)
    utils.init_net(net, net_file)
    return net


if __name__ == '__main__':
    import copy
    from torch.autograd import Variable

    # qnet = build_basic_network(4, 84, 6, None)
    qnet = build_dueling_network(4, 84, 6, None)
    print qnet
    qnet_target = copy.deepcopy(qnet)

    for p in qnet.parameters():
        print p.mean().data[0], p.std().data[0]
    fake_input = Variable(torch.FloatTensor(10, 4, 84, 84))
    print qnet(fake_input).size()
