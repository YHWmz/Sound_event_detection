import torch
import torch.nn as nn


class BceLoss(nn.Module):
    """docstring for BCELoss"""
    def __init__(self):
        super().__init__()

    def forward(self, output):
        return nn.functional.binary_cross_entropy(
            input=output["clip_probs"],
            target=output["targets"])

class SMseLoss(nn.Module):
    """docstring for BCELoss"""
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss(reduction ='sum')

    # shift_t代表x2相对于x1向右移动了shift_t
    def D(self, x1, x2, shift_t):
        x2 = x2.detach()

        if shift_t>0:
            loss = self.loss_fn(x1[:, 0:x1.shape[1]-shift_t, :], x2[:, shift_t:, :])
            loss += self.loss_fn(x1[:, x1.shape[1]-shift_t:, :], x2[:, 0:shift_t, :])
        else:
            shift_t = -shift_t
            loss = self.loss_fn(x1[:, shift_t:, :], x2[:, 0:x1.shape[1] - shift_t, :])
            loss += self.loss_fn(x1[:, 0:shift_t, :], x2[:, x1.shape[1] - shift_t:, :])
        return loss/x1.shape[0]/x1.shape[1]/x1.shape[2]

    def forward(self, x, z, shift_t):
        loss = self.D(x, z, shift_t) + self.D(z, x, -shift_t)

        return loss

class SMseLoss_true(nn.Module):
    """docstring for BCELoss"""
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss(reduction ='sum')

    # shift_t代表x2相对于x1向右移动了shift_t
    def D(self, x1, x2, shift_t):
        # x2 = x2.detach()

        if shift_t>0:
            loss = self.loss_fn(x1[:, 0:x1.shape[1]-shift_t, :], x2[:, shift_t:, :])
            loss += self.loss_fn(x1[:, x1.shape[1]-shift_t:, :], x2[:, 0:shift_t, :])
        else:
            shift_t = -shift_t
            loss = self.loss_fn(x1[:, shift_t:, :], x2[:, 0:x1.shape[1] - shift_t, :])
            loss += self.loss_fn(x1[:, 0:shift_t, :], x2[:, x1.shape[1] - shift_t:, :])
        return loss/x1.shape[0]/x1.shape[1]/x1.shape[2]

    def forward(self, x, z, shift_t):
        # loss = self.D(x, z, shift_t) + self.D(z, x, -shift_t)
        loss = self.D(x, z, shift_t)

        return loss
