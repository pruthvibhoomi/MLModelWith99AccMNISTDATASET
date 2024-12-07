from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from torchsummary import summary
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
             nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
        )
        self.output_size = self._get_conv_output_size((1, 28, 28))
        self.fc1 = nn.Sequential(
            nn.Linear(self.output_size , 32),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

    def _get_conv_output_size(self, input_size):
        """
        Calculates the output size of the convolutional layers for a given input size.
        This is used to dynamically determine the input size for the first fully connected layer.
        """
        with torch.no_grad():
            # Create a dummy input tensor
            dummy_input = torch.zeros(1, *input_size)
            # Pass the dummy input through the convolutional layers
            output = self.conv3(self.conv2(self.conv1(dummy_input)))
            # Calculate the flattened output size
            output_size = output.numel() // output.size(0)
            return output.shape[1] * output.shape[2] * output.shape[3]



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))

torch.manual_seed(1)
batch_size = 128

import torch
import numpy as np

def elastic_transform(image, alpha, sigma):
    """
    Elastic deformation of images as described in [Simard2003].
    """
    #print(type(image))
    image = np.array(image)
    shape = image.shape

    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x = np.clip(x + dx, 0, shape[1] - 1).astype(int)
    y = np.clip(y + dy, 0, shape[0] - 1).astype(int)

    image = image[y, x]
    return image

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                         # transforms.ToPILImage(),
                    # transforms.RandomRotation(3),
                    transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9, 1.1), shear=(-15,15)),
                         transforms.ColorJitter(brightness=0.5, contrast=0.5),
                        transforms.Lambda(lambda x: elastic_transform(x, alpha=7, sigma=7)),

                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

def l1_regularization(model, lambda1):
    l1_reg = 0.
    for param in model.parameters():
        l1_reg += param.abs().sum()
    return lambda1 * l1_reg

# Define the loss function with L2 regularization
def l2_regularization(model, weight_decay):
    l2_reg = 0.
    for param in model.parameters():
        l2_reg += param.norm(2).square()
    return weight_decay * l2_reg

def loss_fn(output, target, model, l1_lambda, l2_lambda):
    loss = nn.CrossEntropyLoss()(output, target)
    reg_loss = l1_regularization(model, l1_lambda) + l2_regularization(model, l2_lambda)
    return loss + reg_loss

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        #criterion = nn.MSELoss() + lambda * torch.norm(model.parameters(), 2) ** 2
        #criterion = nn.CrossEntropyLoss() + l1_regularization(model, 0.01) + l2_regularization(model, 0.01)
        #loss = loss_fn(output, target, model, 0.01, 0.001)
        #loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')

def z_score_outliers(sample, threshold=3):
    mean = torch.mean(sample, dim=0)
    std = torch.std(sample, dim=0)
    z_scores = torch.abs((sample - mean) / std)
    outliers = z_scores > threshold
    return outliers.bool()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Reshape the data to apply z_score_outliers to each image individually
            #original_shape = data.shape
            #data_reshaped = data.reshape(original_shape[0], -1)  # Flatten each image
            #outlier = z_score_outliers(data_reshaped)
            #outlier = outlier.reshape(original_shape)  # Reshape back to original shape

            # Filter out outliers from both data and target
            #filtered_data = data[~outlier.any(dim=1, keepdim=True)]  # Filter on channel dimension
            #filtered_target = target[~outlier.any(dim=1).squeeze(1)]  # Filter target accordingly

            #if filtered_data.size(0) == 0:
            #    print("Skipping batch as all data points were classified as outliers")
            #    continue
            #data = filtered_data
            #target = filtered_target


            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            """
            for i in range(len(data)):
              if pred[i] != target[i]:
                img = data[i].numpy().transpose((1, 2, 0))
                plt.imshow(img)
                plt.title(f"Predicted: {pred[i]}, Actual: {target[i]}")
                plt.show()
            """

    test_loss /= len(test_loader.dataset)
    
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    total_params = sum(p.numel() for p in model.parameters())
    accuracy =  100. * correct / len(test_loader.dataset)
    return total_params,accuracy
    
    
# Create an SGD optimizer with exponential decay
def adjust_learning_rate(optimizer, epoch, initial_lr):
    lr = initial_lr * 0.1 ** (epoch // 2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 20
total_params = 0
accuracy = 0
for epoch in range(num_epochs):
    optimizer = adjust_learning_rate(optimizer, epoch, initial_lr=0.1)
    train(model, device, train_loader, optimizer, epoch)
    total_params, accuracy = test(model, device, test_loader)

assert total_params < 20000, f'Total parameters: {total_params:.2f}% is not less than 20000'
assert accuracy > 99.0, f'Accuracy: {accuracy:.2f}% is not greater than 99%' 
