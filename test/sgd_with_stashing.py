# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import sys
sys.path.append(".")
from optimizer import SGDWithWeightStashing

def test(num_versions, assertion_ground_truth):
    # N is batch size; D_in is input dimension;
    # D_out is output dimension.
    N, D_in, D_H, D_out = 4, 4, 4, 4

    # Create random input and output tensors.
    x1 = torch.randn(N, D_in).cuda().requires_grad_(True)
    y1 = torch.randn(N, D_out).cuda()

    # Create copies of these tensors.
    x2 = x1.clone().detach().requires_grad_(True)
    y2 = y1.clone()

    x3 = x1.clone().detach().requires_grad_(True)
    y3 = y1.clone()

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, D_H),
        torch.nn.ReLU(),
        torch.nn.Linear(D_H, D_out)
    ).cuda()
    loss_fn = torch.nn.MSELoss()
    optimizer = SGDWithWeightStashing([model], model.parameters(), None,
                                      loss_scale= 1.0,
                                      num_versions=num_versions,
                                      lr=1e-1)

    inputs = [x1, x2, x3]
    y1_pred = model(x1)

    # Compute the prediction and loss function using the same weights
    # and inputs.
    x_grads = []

    for x in inputs:
        print('hello')
        optimizer.zero_grad()
        optimizer.load_old_params()
        y_pred = model(x)
        loss = loss_fn(y_pred, y1)
        loss.backward()
        x_grads.append(x.grad.clone().detach())
        optimizer.load_new_params()
        optimizer.step()

    # Assert that the right weight versions are used to compute the
    # gradients.
    assert (torch.equal(x_grads[0], x_grads[1]) ==
            assertion_ground_truth[0])
    assert (torch.equal(x_grads[0], x_grads[2]) ==
            assertion_ground_truth[1])

    assert not torch.equal(y1_pred, model(x1))


if __name__ == '__main__':
    test(1, [False, False])
    test(2, [True, False])
    test(3, [True, True])
