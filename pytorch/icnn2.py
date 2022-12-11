"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICNN(nn.Module):
    """
		Input convex neural network (ICNN)
		input: y
		output: z = h_L
		Architecture: h_1     = LeakyReLU^2(A_0 y + b_0)
					  h_{l+1} = LeakyReLU  (A_l y + b_l + W_{l-1} h_l)
		Projection: W_l >= 0
	"""

    def __init__(self, input_dim=2, hidden_size=64, num_layers=4):
        super().__init__()
        self.num_layers = num_layers

        input_fcs = [
            nn.Linear(input_dim, hidden_size) for _ in range(num_layers)
        ]
        input_fcs.append(nn.Linear(input_dim, 1))

        hidden_fcs = [
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers - 1)
        ]
        hidden_fcs.append(nn.Linear(hidden_size, 1, bias=False))

        self.input_fcs = nn.ModuleList(input_fcs)
        self.hidden_fcs = nn.ModuleList(hidden_fcs)

    @property
    def pos_weights(self):
        ws = []
        for m in self.hidden_fcs.modules():
            if isinstance(m, nn.Linear):
                ws.append(m.weight)
        return ws

    def project(self):
        for w in self.pos_weights:
            w.data.clamp_(min=0)

    def cvx_regularization(self):
        reg = 0.0
        for w in self.pos_weights:
            reg = reg + F.relu(w).square().sum()
        return reg

    def forward(self, y):
        h = F.leaky_relu(self.input_fcs[0](y), 0.2)
        h = h * h

        for y_fc, h_fc in zip(self.input_fcs[1:], self.hidden_fcs):
            h = F.leaky_relu(y_fc(y) + h_fc(h), 0.2)
        return h


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    icnn = ICNN(input_dim=1)
    icnn.project()

    xx = torch.linspace(-10, 10, 1000).reshape(-1, 1)
    yy = icnn(xx)

    xx = xx.detach().numpy().reshape(-1)
    yy = yy.detach().numpy().reshape(-1)

    plt.plot(xx, yy)
    plt.savefig("icnn.png")
