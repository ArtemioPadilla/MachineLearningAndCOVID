import numpy as np
import torch

def predict_model(model, X = np.zeros((1, 21)), device = 'cpu'):
  _, pred = torch.max(model(torch.tensor(X)[:,None, :].to(device).float()), 1)
  return pred.cpu().numpy()[0]
  
def prob_model(model, X = np.zeros((1, 21)), device = 'cpu'):
  log_probs = model(torch.tensor(X)[:,None, :].to(device).float())
  return np.exp(log_probs.cpu().detach().numpy())[0]