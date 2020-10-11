import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torchwrap import PytorchRunner

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(40, 40)
        self.fc2 = nn.Linear(40, 80)
        self.fc3 = nn.Linear(80, 512)
        self.fc4 = nn.Linear(512, 1314)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x



def NNET():
    model = Net()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return PytorchRunner(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )