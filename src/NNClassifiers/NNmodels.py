import torch.nn as nn
import torch.nn.functional as F
class NNclassifier(nn.Module):
  
  def __init__(self, input_dim, hidden_dim, labels_dim):
    super(NNclassifier, self).__init__()

    self.input_dim = input_dim# = máxima longitud de tweets  

    # definimos las capas de convolución (CNN)
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
    self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
    

    # definir una función no líneal de activación
    self.sigmoid = nn.Sigmoid()
    
    # Capas FC
    self.fc1 = nn.Linear(32*(input_dim//2), hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    

    # Linear function 3 
    self.fc3 = nn.Linear(hidden_dim, labels_dim)
    
    # Dropout with p=0.2
    self.dropout = nn.Dropout(p=0.2)

  def forward(self, x):
    # propagación hacia adelante

    # x: [batchsize, 1, input_dim = N]

    out = self.conv1(x)        # [batchsize, 16, N]
    #out = F.max_pool1d(out, kernel_size=2)        # [batchsize, 16, N]
    out = F.relu(out)       # [batchsize, 16, N]
    
    out = self.conv2(out)        # [batchsize, 32, N]
    out = F.max_pool1d(out, kernel_size=2)        # [batchsize, 32, N//2]
    out = F.relu(out)       # [batchsize, 32, N//2]

    # aplanar (flatenning)
    out = out.view(out.size(0), -1) # [batchsize, 32*N//2]

    out = self.fc1(out)
    # Non-linearity 1
    out = self.sigmoid(out)
    # Linear function 2
    out = self.fc2(out)
    # Non-linearity 2
    out = self.sigmoid(out)
    # Linear function 3 (readout)
    out = self.fc3(out)
    
    out = self.dropout(out)
    out = F.log_softmax(out, dim=1)

    return out