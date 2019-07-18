import numpy as np
import torch
import torch.nn as nn

def size(p):
    return np.prod(p.size())

class SharedModel(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

    @property
    def num_parameters(self):
        return sum([size(param) for param in self.parameters()])

    def get_f(self, name):
        raise NotImplementedError()

    def get_num_cell_parameters(self, dag):
        raise NotImplementedError()

    def reset_parameters(self):
        raise NotImplementedError()

class LinearLowRank(nn.Module):
    def __init__(self, row, col, bias=True):
        torch.nn.Module.__init__(self)
        rank = min(row, col)
        self.U = nn.Linear(row, row, bias=bias)
        self.sigma = nn.Parameter(torch.Tensor(rank))
        if row > col:
            self.pad = torch.zeros(row - col, dtype=self.sigma.dtype, requires_grad=False)
            def dot(input):
                pad_sig = torch.cat([self.sigma, self.pad])
                return input * pad_sig
            self.dot = dot
        elif row < col:
            def dot(input):
                return (input * self.sigma).view(
                    -1, input.size()[-1])[::, :col].view(list(input.size())[:-1]+[col])
            self.dot = dot
        else:
            self.dot = lambda x: x * self.sigma
        self.V = nn.Linear(col, col, bias=bias)

    def forward(self, input):
        t1 = self.U(input)
        t2 = self.dot(t1)
        t3 = self.V(t2)
        return t3

    def sparse(self):
        return torch.sum(torch.abs(self.sigma))

    def orth(self):
        def compute(u):
            return (u.transpose(1, 0).mm(u) - torch.eye(u.size(0)).cuda()).norm(2)
        return compute(self.U.weight) + compute(self.V.weight)
        #return (self.U.weight.transpose(1, 0).mm(self.U.weight) - torch.eye(self.U.weight.size(0)).cuda()).norm(2)
