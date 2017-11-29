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


def build_basic_network(input_channels, input_size, output_dim, net_file):
    conv = nn.Sequential(
        nn.Conv2d(input_channels, 32, 8, 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU()
    )

    input_shape = (1, input_channels, input_size, input_size)
    fc_in = utils.count_output_size(input_shape, conv)
    fc_hid = 512
    fc = nn.Sequential(
        nn.Linear(fc_in, fc_hid),
        nn.ReLU(),
        nn.Linear(fc_hid, output_dim)
    )
    net = BasicNetwork(conv, fc)
    utils.init_net(net, net_file)
    return net


class DuelingNetwork(nn.Module):
    def __init__(self, conv, adv, val):
        super(DuelingNetwork, self).__init__()
        self.conv = conv
        self.adv = adv
        self.val = val

    def forward(self, x):
        assert x.max() <= 1.0
        batch = x.size(0)
        feat = self.conv(x)
        feat = feat.view(batch, -1)
        adv = self.adv(feat)
        val = self.val(feat)
        q = val - adv.mean(1, keepdim=True) + adv
        return q


def build_dueling_network(input_channels, input_size, output_dim, net_file):
    conv = nn.Sequential(
        nn.Conv2d(input_channels, 32, 8, 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU()
    )

    input_shape = (1, input_channels, input_size, input_size)
    fc_in = utils.count_output_size(input_shape, conv)
    fc_hid = 512
    adv = nn.Sequential(
        nn.Linear(fc_in, fc_hid),
        nn.ReLU(),
        nn.Linear(fc_hid, output_dim)
    )
    val = nn.Sequential(
        nn.Linear(fc_in, fc_hid),
        nn.ReLU(),
        nn.Linear(fc_hid, 1)
    )

    net = DuelingNetwork(conv, adv, val)
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
