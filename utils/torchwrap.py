#torchwrap.py


import time
import copy


import torch
from data.pytorch.dataset import QuickDataset
from utils.transforms import np_to_tensor


class PytorchRunner():

    def __init__(self, model, criterion, optimizer, scheduler, **kwargs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def fit(self, x, y, **kwargs):
        # Create a pytorch dataset from pandas dataframe (assumes all values float!)
        dataset = QuickDataset(
            x=x.to_numpy(),
            y=y.to_numpy(),
            transforms=[
                np_to_tensor(),
            ]
        )

        # Split data into training and validation sets.  Create pytorch data loaders
        tlen = int(0.8 * len(dataset))
        vlen = len(dataset) - tlen
        trainset, valset = torch.utils.data.random_split(dataset, (tlen, vlen))
        loader = {
            'train': torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True),
            'val': torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True), 
        }

        self.model = train_model(
            model=self.model,
            loader=loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            num_epochs=20
        )

    def score(self, x, y, **kwargs):
        print('Score not enabled yet')
        # return self.criterion(self.model.forward(x), y)





def train_model(model, loader, criterion, optimizer, scheduler, device, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("Running on device: ", device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-'*10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # sets model to training mode
            else:
                model.eval()  # sets model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in loader[phase]:
                # Transfer data to device(s)
                inputs = inputs.to(device)
                labels = labels.float().to(device)
                _, label_ind = torch.max(labels, 1)

                # Zero parameter gradients
                optimizer.zero_grad()

                # Forward, tracking history only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # ### DEBUG
                    # print(f"INPUTS: \n{inputs}\n{inputs.shape}")
                    # print(f"LABELS: \n{labels}\n{labels.shape}")
                    # print(f"OUTPUTS: \n{outputs}\n{outputs.shape}")
                    # print(f"PREDS: \n{preds}\n{preds.shape}")
                    
                    loss = criterion(outputs, labels)

            #         # Backward if in train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Get run stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == label_ind)

                # Update scheduler if in train
                if phase == 'train':
                    scheduler.step()

                # Calculate loss and acc for the epoch
                epoch_loss = running_loss / len(loader[phase].dataset)
                epoch_acc = running_corrects.double() / len(loader[phase].dataset)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            
            print()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model