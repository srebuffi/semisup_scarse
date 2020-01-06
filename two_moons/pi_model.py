import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path
import numpy as np
import matplotlib as mpl
# Use this to avoid printing figures on screen
mpl.use('Agg')
import matplotlib.pyplot as plt

# Number for unlabelled samples per class
N = 1000
# Noise scale
scale = 0.15
# Consistency noise
noise_level = 0.1
# Print every frame
print_every = 50
# Set seed for reproductibility
np.random.seed(1)
torch.manual_seed(1)


def generate_two_moons_dataset(N, scale):
    ''' Generate two moons dataset '''
    _x = np.linspace(0, np.pi, num=N)
    x0 = np.array([_x, np.sin(_x)]).T
    x1 = -1 * x0 + [np.pi / 2 + 0.1, 0.15]

    # Sample unlabeled points
    x0 += np.random.normal(scale=scale, size=(N, 2))
    x1 += np.random.normal(scale=scale, size=(N, 2))
    X_unlab = np.vstack([x0, x1])

    # Labeled Data points (same as in the paper)
    labeled_data = np.zeros((6, 2))
    labeled_data[0] = [1.1, 1.]
    labeled_data[1] = [2., 1.]
    labeled_data[2] = [2.9, 0.3]
    labeled_data[3] = [-0.8, -0.5]
    labeled_data[4] = [0, - 0.8]
    labeled_data[5] = [1, - 0.5]

    # Create dataset tensors
    X_data = np.vstack([X_unlab, labeled_data])
    X_data = torch.from_numpy(X_data).float()
    labels_sup = torch.FloatTensor([0, 0, 0, 1, 1, 1]).view(-1, 1)
    labels_data = torch.cat((torch.zeros(N, 1), torch.ones(N, 1), labels_sup))

    return X_data, labels_sup, labels_data


X_data, labels_sup, labels_data = generate_two_moons_dataset(N, scale)

# Init the model
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(True),
    nn.Linear(10, 10),
    nn.ReLU(True),
    nn.Linear(10, 10),
    nn.ReLU(True),
    nn.Linear(10, 10),
    nn.ReLU(True),
    nn.Linear(10, 1)
)
model[0].weight.data = model[0].weight.data.normal_(0.0, np.sqrt(2.0 / 2))
model[2].weight.data = model[2].weight.data.normal_(0.0, np.sqrt(2.0 / 10))
model[4].weight.data = model[4].weight.data.normal_(0.0, np.sqrt(2.0 / 10))
model[6].weight.data = model[6].weight.data.normal_(0.0, np.sqrt(2.0 / 10))
model[8].weight.data = model[8].weight.data.normal_(0.0, np.sqrt(2.0 / 10))

# Init optimizer (SGD+momentum)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Directory where the images will be saved
directory = 'render'
if not os.path.exists(directory):
    os.makedirs(directory)
plt.ion()

# Initial training phase
for i in range(2000):
    # Sample data and consistency ones
    noise = noise_level * torch.Tensor(*X_data.size()).random_(-1, 2)
    noise.required_grad = False
    noise_consistency = noise_level * torch.Tensor(*X_data.size()).random_(-1, 2)
    noise_consistency.required_grad = False
    if i < 200:
        consistency_lambda = 0
    elif i < 1000:
        consistency_lambda = 10. * (i - 200) / 800
    else:
        consistency_lambda = 10.
    output = model(X_data + noise.detach())
    output_consistency = model(X_data + noise_consistency.detach())

    # Compute loss and update parameters
    loss = -torch.mean(labels_sup * F.logsigmoid(output[-6:]) + (1 - labels_sup) * F.logsigmoid(-output[-6:]))
    loss += consistency_lambda * torch.mean((torch.sigmoid(output) - torch.sigmoid(output_consistency)) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Loss and Border visualisation
    if i % print_every == 1:
        with torch.no_grad():
            output = model(X_data)
            # Compute test accuracy
            test_acc = torch.sum((output > 0).float() == labels_data).item() * 100. / (2 * N + 6)
            print('Test Acc : {}'.format(test_acc))

            # Print border and heatmap
            predicted_labels = (model(X_data) > 0).float()
            plt.plot(X_data[(predicted_labels == 1).nonzero()[:, 0], 0].numpy(), X_data[(predicted_labels == 1).nonzero()[:, 0], 1].numpy(), '.', alpha=0.5, label='unlabeled x0', color='royalblue')
            plt.plot(X_data[(predicted_labels == 0).nonzero()[:, 0], 0].numpy(), X_data[(predicted_labels == 0).nonzero()[:, 0], 1].numpy(), '.', alpha=0.5, label='unlabeled x1', color='coral')
            plt.plot(X_data[-6:, 0].numpy(), X_data[-6:, 1].numpy(), '.', alpha=1., markersize=8, label='unlabeled x1', color='yellow')
            y, x = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-2, 3.5, 100))
            z = torch.sigmoid(model(torch.from_numpy(np.concatenate((np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))), axis=1)).float()))
            plt.pcolormesh(x, y, np.reshape(z, (100, 100)), cmap='RdBu', vmin=0, vmax=1, alpha=0.5)
            plt.axis('off')
            plt.savefig(directory + '/%u.jpg' % i, transparent=True, bbox_inches='tight', pad_inches=0)
plt.ioff()
