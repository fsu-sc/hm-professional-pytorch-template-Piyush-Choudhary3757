import torch
import numpy as np

def mse_loss(output, target):
    """Mean squared error loss"""
    with torch.no_grad():
        return torch.mean((output - target) ** 2).item()

def mae_loss(output, target):
    """Mean absolute error loss"""
    with torch.no_grad():
        return torch.mean(torch.abs(output - target)).item()

def r2_score(output, target):
    """R² score (coefficient of determination)"""
    with torch.no_grad():
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - output) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2.item()

def explained_variance(output, target):
    """Explained variance score"""
    with torch.no_grad():
        target_var = torch.var(target)
        explained_var = 1 - torch.var(target - output) / target_var
        return explained_var.item()