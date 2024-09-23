'''
Modified from: https://github.com/Wendison/VQMIVC/blob/main/mi_estimators.py
'''

import torch 
import torch.nn as nn

import torch
import torch.nn as nn

class CLUB(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Conv1d(x_dim, hidden_size//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size//2, y_dim, kernel_size=1)
        )
        self.p_logvar = nn.Sequential(
            nn.Conv1d(x_dim, hidden_size//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size//2, y_dim, kernel_size=1),
            nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def mi_est(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # Reshape for broadcasting
        mu = mu.transpose(1, 2)  # [batchsize, t, y_dim]
        logvar = logvar.transpose(1, 2)  # [batchsize, t, y_dim]
        y_samples = y_samples.transpose(1, 2)  # [batchsize, t, y_dim]

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()

        # Reshape for negative sample computation
        prediction_1 = mu.unsqueeze(2)  # [batchsize, t, 1, y_dim]
        y_samples_1 = y_samples.unsqueeze(1)  # [batchsize, 1, t, y_dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=2)/2./logvar.exp() 

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
    

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # Reshape for computation
        mu = mu.transpose(1, 2)  # [batchsize, t, y_dim]
        logvar = logvar.transpose(1, 2)  # [batchsize, t, y_dim]
        y_samples = y_samples.transpose(1, 2)  # [batchsize, t, y_dim]
        
        return (-(mu - y_samples)**2 / logvar.exp() - logvar).sum(dim=-1).mean()
        
    
        

    
class CLUBSample_group(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample_group, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, hidden_size//2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, hidden_size//2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size//2, hidden_size//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size//2, hidden_size//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size//2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        # x_samples: [batch_size, x_dim, T]
        x_samples = x_samples.permute(0, 2, 1)  # [batch_size, T, x_dim]
        x_samples = x_samples.reshape(-1, x_samples.shape[-1])  # [batch_size * T, x_dim]
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        # x_samples, y_samples: [batch_size, dim, T]
        mu, logvar = self.get_mu_logvar(x_samples)
        
        y_samples = y_samples.permute(0, 2, 1)  # [batch_size, T, y_dim]
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])  # [batch_size * T, y_dim]
        
        return (-(mu - y_samples)**2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0) / 2

    def mi_est(self, x_samples, y_samples):
        # x_samples, y_samples: [batch_size, dim, T]
        batch_size, _, T = x_samples.shape
        
        mu, logvar = self.get_mu_logvar(x_samples)  # [batch_size * T, y_dim]
        
        y_samples = y_samples.permute(0, 2, 1)  # [batch_size, T, y_dim]
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])  # [batch_size * T, y_dim]
        
        random_index = torch.randperm(batch_size).repeat_interleave(T)
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()

        return (positive.sum(dim=1) - negative.sum(dim=1)).mean() / 2

