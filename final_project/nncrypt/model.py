import torch
import torch.nn as nn
import torch.nn.functional as F


class Alice(nn.Module):
    def __init__(self, hp):
        super(Alice, self).__init__()
        self.depth = hp.alice.depth
        self.hidden = hp.alice.hidden
        self.input_size = hp.data.cipher + hp.data.key
        self.output_size = hp.data.cipher
        # self.mlp = nn.ModuleList(
        #     [nn.Linear(hp.data.plain + hp.data.key, self.hidden)] +
        #     [nn.Linear(self.hidden, self.hidden) for _ in range(self.depth-2)])
        # self.last = nn.Linear(self.hidden, hp.data.cipher)
        self.fc = nn.Linear(self.input_size,self.input_size)
        self.conv_net = nn.Sequential(
        	nn.Conv1d(in_channels=1, out_channels=2,kernel_size=4,stride=1,padding=1),
        	nn.Sigmoid(),
        	nn.Conv1d(in_channels=2, out_channels=4,kernel_size=2,stride=2,padding=1),
        	nn.Sigmoid(),
        	nn.Conv1d(in_channels=4, out_channels=4,kernel_size=1,stride=1,padding=0),
        	nn.Sigmoid(),
        	nn.Conv1d(in_channels=4, out_channels=1,kernel_size=1,stride=1,padding=0),
        	nn.Tanh(),
        	)

    def forward(self, p, k):
        x = torch.cat((p, k), dim=-1)
        #
        x = self.fc(x)
        x = x.view(-1,1,self.input_size)
        x = self.conv_net(x)
        x = x.view(-1,self.output_size)
        return x


class Bob(nn.Module):
    def __init__(self, hp):
        super(Bob, self).__init__()
        self.depth = hp.bob.depth
        self.hidden = hp.bob.hidden
        self.input_size = hp.data.cipher + hp.data.key
        self.output_size = hp.data.plain
        # self.mlp = nn.ModuleList(
        #     [nn.Linear(hp.data.cipher + hp.data.key, self.hidden)] +
        #     [nn.Linear(self.hidden, self.hidden) for _ in range(self.depth-2)])
        # self.last = nn.Linear(self.hidden, hp.data.plain)
        self.fc = nn.Linear(self.input_size,self.input_size)
        self.conv_net = nn.Sequential(
        	nn.Conv1d(in_channels=1, out_channels=2,kernel_size=4,stride=1,padding=1),
        	nn.Sigmoid(),
        	nn.Conv1d(in_channels=2, out_channels=4,kernel_size=2,stride=2,padding=1),
        	nn.Sigmoid(),
        	nn.Conv1d(in_channels=4, out_channels=4,kernel_size=1,stride=1,padding=0),
        	nn.Sigmoid(),
        	nn.Conv1d(in_channels=4, out_channels=1,kernel_size=1,stride=1,padding=0),
        	nn.Tanh(),
        	)

    def forward(self, c, k):
        x = torch.cat((c, k), dim=-1)

        # for idx, layer in enumerate(self.mlp):
        #     if idx == 0:
        #         x = F.relu(layer(x))
        #     else:
        #         x = F.relu(x + layer(x))

        # x = torch.tanh(self.last(x))
        x = self.fc(x)
        x = x.view(-1,1,self.input_size)
        x = self.conv_net(x)
        x = x.view(-1,self.output_size)
        return x


class Eve(nn.Module):
    def __init__(self, hp):
        super(Eve, self).__init__()
        self.depth = hp.eve.depth
        self.hidden = hp.eve.hidden
        self.input_size = hp.data.cipher
        self.output_size = hp.data.plain
        # self.mlp = nn.ModuleList(
        #     [nn.Linear(hp.data.cipher, self.hidden)] + 
        #     [nn.Linear(self.hidden, self.hidden) for _ in range(self.depth-1)])
        # self.last = nn.Linear(self.hidden, hp.data.plain)
        self.fc = nn.Linear(self.input_size, self.input_size*2)
        self.conv_net = nn.Sequential(
        	nn.Conv1d(in_channels=1, out_channels=2,kernel_size=4,stride=1,padding=1),
        	nn.Sigmoid(),
        	nn.Conv1d(in_channels=2, out_channels=4,kernel_size=2,stride=2,padding=1),
        	nn.Sigmoid(),
        	nn.Conv1d(in_channels=4, out_channels=4,kernel_size=1,stride=1,padding=0),
        	nn.Sigmoid(),
        	nn.Conv1d(in_channels=4, out_channels=1,kernel_size=1,stride=1,padding=0),
        	nn.Tanh(),
        	)

    def forward(self, c):
        x = c

        # for idx, layer in enumerate(self.mlp):
        #     if idx == 0:
        #         x = F.relu(layer(x))
        #     else:
        #         x = F.relu(x + layer(x))

        # x = torch.tanh(self.last(x))
        x = self.fc(x)
        x = x.view(-1,1,self.input_size*2)
        x = self.conv_net(x)
        x = x.view(-1,self.output_size)
        return x
