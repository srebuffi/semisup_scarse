from __future__ import print_function
from operator import itemgetter
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os.path
import datetime
import numpy as np
import models
import dataset
import utils

logger_module = None


def train(args, model, device, train_loader, optimizer, epoch, meta_or_train, outer_loop):
    global thought_targets
    global meta_labels_total
    global logger_module

    if (outer_loop > 0) and (meta_or_train == 'meta'):
        model.eval()
    else:
        model.train()

    # Training loop, train with cross entropy and consistency losses
    sum_loss = 0
    correct, total_seen = 0, 0
    for batch_idx, (data, target, index_batch) in enumerate(train_loader()):
        if meta_or_train == 'train':
            # Meta labels are a linear combination of the assigned labels and the predicted labels at the previous epoch
            target = thought_targets[index_batch]
            meta_labels = meta_labels_total[index_batch].clone()
            meta_labels.requires_grad = False
            prob_true_cl = torch.gather(meta_labels, 1, target.view(-1, 1))
            meta_labels *= (1. - args.proportion_CE)
            meta_labels = meta_labels.scatter_(1, target.view(-1, 1), prob_true_cl * (1. - args.proportion_CE) + args.proportion_CE)
            data, target, meta_labels = data.to(device), target.to(device), meta_labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = -torch.mean(torch.sum(meta_labels * F.log_softmax(output, dim=1), 1))
        elif meta_or_train == 'meta':
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_seen += len(target)
        sum_loss += loss.item()
        loss.backward()
        optimizer.step()
        if meta_or_train == 'train':
            meta_labels_total[index_batch] = F.softmax(output, dim=1).data.cpu()  # Store the predictions for the next epoch for self distillation
        if batch_idx % args.log_interval == 0:
            if meta_or_train == 'train':
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), sum_loss / (batch_idx + 1)
                ))

    logger_module.train_loss.append(sum_loss / (batch_idx + 1))
    logger_module.train_acc.append(correct * 100. / total_seen)


def fine_tune_and_assign_labels(args, model, metaloader, trainloader, testloader, device, train_set, trainset_data_save,\
     trainset_targets_save, index_meta, outer_loop):
    global logger_module
    global thought_targets

    train_data = 'train_data' if args.dataset != 'svhn' else 'data'
    train_labels = 'train_labels' if args.dataset != 'svhn' else 'labels'

    # Label assignment phase and preparing the data for the inner loop training
    setattr(train_set.data, train_data, trainset_data_save)
    setattr(train_set.data, train_labels, trainset_targets_save)

    if outer_loop == 0:
        print('First assignment pass: training only with labeled samples')
        for name, param in model.named_parameters():
            if 'fc' not in name and 'layer3' not in name and 'layer4' not in name and 'conv3' not in name:
                param.requires_grad = False
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
        nb_epochs = args.epochs_refine
    else:
        print('Finetune the classifier layer with labeled samples')
        params = []
        for name, param in model.named_parameters():
            if 'fc' in name:
                params += [{'params': param}]
            else:
                param.requires_grad = False
        optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=5e-4)
        nb_epochs = 100

    # Training loop
    for epoch in range(1, nb_epochs):
        train(args, model, device, metaloader, optimizer, epoch, 'meta', outer_loop)
    test(args, model, device, testloader)

    # Assign soft labels as computed by the model
    model.eval()
    size_train = len(train_set)
    correct_tensor = torch.zeros(size_train).type(torch.ByteTensor)
    thought_targets = torch.zeros(size_train).type(torch.LongTensor)
    correct_total = 0
    with torch.no_grad():
        for batch_idx, (data, target, index_batch) in enumerate(trainloader()):
            data, target = data.to(device), target.to(device)
            output = model(data)
            value_pred, pred = (F.softmax(output, dim=1)).max(1, keepdim=True)
            correct_total += pred.eq(target.view_as(pred)).sum().item()
            thought_targets[index_batch] = pred[:, 0].cpu()
            correct_tensor[index_batch] = (pred[:, 0].data == target).cpu()

    # Split between training set and held out set
    mask_training = torch.zeros(size_train).type(torch.LongTensor)
    if (outer_loop % 3) == 0:
        global index_random
        index_random = np.random.choice(size_train, size_train, False)
        mask_training[index_random[0:int(2. / 3 * size_train)]] = 1
    elif (outer_loop % 3) == 1:
        mask_training[index_random[int(1. / 3 * size_train):]] = 2
    else:
        mask_training[index_random[int(2. / 3 * size_train):]] = 3
        mask_training[index_random[0:int(1. / 3 * size_train)]] = 3

    # From a certain point, no more splitting and the whole dataset becomes the training set
    current_idx = (outer_loop % 3) + 1
    if outer_loop > args.milestones_outer[0]:
        mask_training = current_idx * torch.ones(size_train).type(torch.LongTensor)

    print('Correctly labelled data on train set %f' % (100. / size_train * correct_total))
    logger_module.percentage_correct_training.append(torch.sum(correct_tensor[mask_training == current_idx]).item() / torch.sum(mask_training == current_idx).item())

    # Correct the labels we know in the training set
    thought_targets[index_meta] = torch.from_numpy(trainset_targets_save[index_meta]).type(torch.LongTensor)

    # Build held-out set
    most_probable_samples = (mask_training == current_idx)
    setattr(train_set.data, train_data, trainset_data_save[np.where(most_probable_samples.numpy())[0]])
    setattr(train_set.data, train_labels, trainset_targets_save[np.where(most_probable_samples.numpy())[0]])
    thought_targets = thought_targets[most_probable_samples]


def test(args, model, device, test_loader, record=False):
    global logger_module

    # Compute test loss
    model.eval()
    test_loss = 0
    correct_tot1 = 0
    correct_tot5 = 0
    with torch.no_grad():
        for (data, target, index_batch) in test_loader():
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()  # sum up batch loss
            correct1, correct5 = utils.accuracy(output, target, topk=(1, 5))
            correct_tot1 += correct1
            correct_tot5 += correct5

    logger_module.test_loss.append(test_loss / len(test_loader.dataset))
    logger_module.test_acc.append(100. * correct_tot1 / len(test_loader.dataset))
    logger_module.test_acc5.append(100. * correct_tot5 / len(test_loader.dataset))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss / len(test_loader.dataset), correct_tot1, len(test_loader.dataset),
        100. * correct_tot1 / len(test_loader.dataset)
    ))

    # save results into a txt file
    if record:
        text_file = open(os.path.join(logger_module.save_dir, 'results_%u_%s.txt' % (logger_module.nb_labels_per_class, logger_module.network.lower())), 'a')
        text_file.write('Seed: %u' % (logger_module.seed))
        text_file.write('\n')
        text_file.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct_tot1, len(test_loader.dataset),
            100. * correct_tot1 / len(test_loader.dataset)))
        text_file.write('\n')
        text_file.close()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Alternative Training for Semi-supervised learning')
    parser.add_argument('--autoaugment', action='store_true', default=False,
                        help='Use AutoAugment data augmentation (default: False)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset (default: cifar10)')
    parser.add_argument('--epochs_refine', type=int, default=100,
                        help='Refinement epochs on labelled set')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default 0.01)')
    parser.add_argument('--milestones_outer', nargs='+', type=int, default=[60, 100],
                        help='Outer loop milestones')
    parser.add_argument('--milestones_inner', nargs='+', type=int, default=[7, 10],
                        help='Inner loop milestones (change of lr and number of epochs)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--nb_labels_per_class', type=int, default=10,
                        help='Number of labelled samples per class (default: 10)')
    parser.add_argument('--network', type=str, default='ResNet-18',
                        help='Network (default: ResNet-18)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training (default: False)')
    parser.add_argument('--proportion_CE', type=float, default=0.5,
                        help='Weight of cross entropy loss')
    parser.add_argument('--rotnet_dir', type=str, default='',
                        help='RotNet saved directory')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--save_dir', type=str, default='./data/alternative_supervised/',
                        help='Directory to save models')
    args = parser.parse_args()

    global logger_module
    logger_module = args
    logger_module.time_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Path to file
    os.makedirs(args.save_dir, exist_ok=True)
    args.name = 'alternative_%s_%s_seed%u' % (logger_module.dataset.lower(), logger_module.network.lower(), args.seed)
    logger_module.net_path = os.path.join(args.save_dir, args.name + '.pth')
    logger_module.pkl_path = os.path.join(args.save_dir, args.name + '.pkl')

    logger_module.train_loss = []
    logger_module.train_acc = []
    logger_module.test_loss = []
    logger_module.test_acc = []
    logger_module.test_acc5 = []
    logger_module.percentage_correct_training = []
    logger_module.number_training = []

    train_data = 'train_data' if args.dataset != 'svhn' else 'data'
    train_labels = 'train_labels' if args.dataset != 'svhn' else 'labels'

    with open(logger_module.pkl_path, "wb") as output_file:
        pickle.dump(vars(logger_module), output_file)

    # Set up seed and GPU usage
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize the dataset
    train_set = dataset.GenericDataset(args.dataset, 'train')
    test_set = dataset.GenericDataset(args.dataset, 'test')

    # Build meta set containing only the restricted labeled samples
    meta_set = dataset.GenericDataset(args.dataset, 'train')
    index_meta = []
    for target in range(train_set.n_classes):
        index_meta.extend(np.random.choice(np.argwhere(np.array(getattr(train_set.data, train_labels)) == target)[:, 0], args.nb_labels_per_class, False))

    setattr(meta_set.data, train_labels, list(itemgetter(*index_meta)(getattr(train_set.data, train_labels))))
    setattr(meta_set.data, train_data, list(itemgetter(*index_meta)(getattr(train_set.data, train_data))))

    # Copy train set for future reassignment
    trainset_targets_save = np.copy(getattr(train_set.data, train_labels))
    trainset_data_save = np.copy(getattr(train_set.data, train_data))

    # Dataloader iterators # TODO Autoaugment
    trainloader = dataset.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    metaloader = dataset.DataLoader(meta_set, batch_size=128, shuffle=True, num_workers=2)
    testloader = dataset.DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=1)

    # First network intialization
    model = models.load_net(logger_module.network, train_set.n_classes)

    # Load model
    if args.rotnet_dir:
        state_dict_rotnet = torch.load(os.path.join(args.rotnet_dir, 'rotNet_%s_%s_lr_best.pth' % (logger_module.dataset.lower(), logger_module.network.lower())))
        del state_dict_rotnet['fc.weight']
        del state_dict_rotnet['fc.bias']
        model.load_state_dict(state_dict_rotnet, strict=False)
    model = model.to(device)

    global thought_targets
    global meta_labels_total
    for outer_loop in range(0, args.milestones_outer[1]):
        print('Entering outer loop %u' % (outer_loop))

        # Step 1: Fine-tune network and assign Labels
        fine_tune_and_assign_labels(args, model, metaloader, trainloader, testloader, device, train_set, trainset_data_save, trainset_targets_save,\
             index_meta, outer_loop)

        # Self distillation starts from a uniform distribution
        meta_labels_total = torch.ones(len(trainloader.dataset), trainloader.dataset.n_classes) / float(trainloader.dataset.n_classes)

        # Step 1.5: Reinitialize net
        model = models.load_net(logger_module.network, train_set.n_classes)
        # Load model
        if args.rotnet_dir:
            state_dict_rotnet = torch.load(os.path.join(args.rotnet_dir, 'rotNet_%s_%s_lr_best.pth' % (logger_module.dataset.lower(), logger_module.network.lower())))
            del state_dict_rotnet['fc.weight']
            del state_dict_rotnet['fc.bias']
            model.load_state_dict(state_dict_rotnet, strict=False)
        model = model.to(device)

        # Freeze net first two blocks
        for name, param in model.named_parameters():
            if 'fc' not in name and 'layer3' not in name and 'layer4' not in name:
                param.requires_grad = False

        # Optimizer and LR scheduler
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.milestones_inner[0]], gamma=0.1)

        # Step 2: Training using predicted labels
        print('Labels assignment done. Entering inner loop')
        for epoch in range(args.milestones_inner[1]):
            scheduler.step()
            train(args, model, device, trainloader, optimizer, epoch, 'train', outer_loop)
            test(args, model, device, testloader)
            logger_module.epoch = epoch

        with open(logger_module.pkl_path, "wb") as output_file:
            pickle.dump(vars(logger_module), output_file)

        torch.save(model.state_dict(), logger_module.net_path)
    test(args, model, device, testloader, True)


if __name__ == '__main__':
    main()
