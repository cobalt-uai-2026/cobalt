import torch
import torch.nn as nn
import math

class GaussianNLLLoss(nn.Module):
    def __init__(self):
        super(GaussianNLLLoss, self).__init__()

    def forward(self, mu_pred, target, censorship=None, sigma_pred=None):
        # FIX: Convert default sigma to a Tensor on the correct device
        if sigma_pred is None:
            sigma_pred = torch.tensor(1.0, device=mu_pred.device)

        # Ensure sigma_pred is a tensor even if passed as float
        if not torch.is_tensor(sigma_pred):
            sigma_pred = torch.tensor(float(sigma_pred), device=mu_pred.device)

        z = (target - mu_pred) / sigma_pred
        z = torch.clamp(z, min=-50., max=50.)

        nll = 0.5 * math.log(2 * math.pi) + torch.log(sigma_pred) + 0.5 * (z ** 2)
        return nll.mean()


class RobustTobitLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RobustTobitLoss, self).__init__()
        self.reduction = reduction

    def forward(self, mu_pred, target, censorship, sigma_pred=None):
        # FIX: Convert default sigma to a Tensor
        if sigma_pred is None:
            sigma_pred = torch.tensor(1.0, device=mu_pred.device)

        if not torch.is_tensor(sigma_pred):
            sigma_pred = torch.tensor(float(sigma_pred), device=mu_pred.device)

        z = (target - mu_pred) / sigma_pred
        z = torch.clamp(z, min=-50., max=50.)

        log_pdf = -0.5 * math.log(2 * math.pi) - torch.log(sigma_pred) - 0.5 * (z ** 2)
        log_survival = torch.special.log_ndtr(-z)
        log_cdf = torch.special.log_ndtr(z)

        # Handle cases where censorship is None (assume uncensored)
        if censorship is None:
            censorship = torch.zeros_like(target)

        loss = torch.zeros_like(target)

        # Boolean masking for vectorized operations
        mask_uncensored = (censorship == 0)
        mask_right = (censorship == 1)
        mask_left = (censorship == -1)

        loss[mask_uncensored] = log_pdf[mask_uncensored]
        loss[mask_right] = log_survival[mask_right]
        loss[mask_left] = log_cdf[mask_left]

        nll = -loss

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll