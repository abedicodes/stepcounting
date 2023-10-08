import torch
from torch import nn
from torch import optim
from torch.nn.utils import weight_norm
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

torch.manual_seed(69)


class attention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(attention, self).__init__()

        self.D = 100
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = 1
        self.bidirectional = False

        self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True,
                          bidirectional=self.bidirectional)

        self.attention = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.linear_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x, lengths):
        state = None
        x_packed = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False).cuda()
        h_packed, state = self.rnn(x_packed, state)
        h, output_lengths = rnn_utils.pad_packed_sequence(h_packed, batch_first=True)

        concatenation = torch.zeros((h.shape[0], self.hidden_dim * 2), requires_grad=True).cuda()

        for i in range(h.shape[0]):

            S = lengths[i]
            MIN, MAX = max(S - self.D, 0), min(S + self.D, S)
            H = torch.unsqueeze(h[i, MIN:MAX, :], dim=0)
            HT = torch.sum(H, dim=1)
            energy = self.attention(H)
            energies = energy.bmm(HT.unsqueeze(2)).squeeze(axis=2)
            weights = F.softmax(energies, dim=1)

            context = torch.matmul(weights, H)
            contextT = torch.squeeze(context, dim=1)
            _concatenation_ = torch.cat((contextT, HT), dim=1)
            concatenation[i] = torch.squeeze(_concatenation_)

        out = self.linear(concatenation)

        out = self.linear_out(out)

        return out

    def init_hidden(self, batch_size):
        h0 = torch.zeros(2 if self.bidirectional else 1, batch_size, self.hidden_dim).cuda()
        c0 = torch.zeros(2 if self.bidirectional else 1, batch_size, self.hidden_dim).cuda()
        return h0, c0

