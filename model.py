import torch
import torch.nn as nn
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
        self.num_actions
        self.num_atoms = num_atoms

    def forward(self, x):
        batch = x.size(0)
        y = self.conv(x)
        y = y.view(batch, -1)
        y = self.fc(y)
        logits = y.view(batch, self.num_actions, self.num_atoms)
        # TODO: use pytorch softmax here
        probs = utils.softmax(logits, 2)
        return probs


# ---------------------------------------


def _build_default_conv(input_channels):
    conv = nn.Sequential(
        nn.Conv2d(input_channels, 32, 8, 4),
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


def build_basic_network(input_channels, input_size, output_dim, net_file):
    conv = _build_default_conv(input_channels)

    input_shape = (1, input_channels, input_size, input_size)
    fc_in = utils.count_output_size(input_shape, conv)
    fc_hid = 512
    fc = _build_fc([fc_in, fc_hid, output_dim])

    net = BasicNetwork(conv, fc)
    utils.init_net(net, net_file)
    return net


def build_dueling_network(input_channels, input_size, output_dim, net_file):
    conv = _build_default_conv(input_channels)

    input_shape = (1, input_channels, input_size, input_size)
    fc_in = utils.count_output_size(input_shape, conv)
    fc_hid = 512
    adv = _build_fc([fc_in, fc_hid, output_dim])
    val = _build_fc([fc_in, fc_hid, 1])

    net = DuelingNetwork(conv, adv, val)
    utils.init_net(net, net_file)
    return net


def build_distributional_basic_network(
        input_channels, input_size, output_dim, num_atoms, net_file):

    conv = _build_default_conv(input_channels)

    input_shape = (1, input_channels, input_size, input_size)
    fc_in = utils.count_output_size(input_shape, conv)
    fc_hid = 512
    fc = _build_fc([fc_in, fc_hid, output_dim * num_atoms])

    net = DistributionalBasicNetwork(conv, fc, output_dim, num_atoms)
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
