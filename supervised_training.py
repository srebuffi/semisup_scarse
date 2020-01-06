from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os.path
import numpy as np
import dataset
from tqdm import tqdm
import models
import utils


def train(epoch, model, device, dataloader, optimizer, exp_lr_scheduler, criterion, args):
    """ Train loop, predict classes. """
    loss_record = utils.AverageMeter()
    acc_record = utils.AverageMeter()
    save_path = args.save_dir + '/'
    os.makedirs(save_path, exist_ok=True)
    exp_lr_scheduler.step()
    model.train()
    for batch_idx, (data, label, index_batch) in enumerate(tqdm(dataloader())):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        #  _, pred = torch.max(output, 1)
        loss = criterion(output, label)

        # measure accuracy and record loss
        acc = utils.accuracy(output, label)
        acc_record.update(100 * acc[0].item() / data.size(0), data.size(0))
        loss_record.update(loss.item(), data.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))

    return loss_record


def test(model, device, dataloader, args):
    """ Test loop, print metrics """
    acc_record = utils.AverageMeter()
    model.eval()
    for (data, label, index_batch) in tqdm(dataloader()):
        data, label = data.to(device), label.to(device)
        output = model(data)

        # measure accuracy and record loss
        acc = utils.accuracy(output, label)
        acc_record.update(100 * acc[0].item() / data.size(0), data.size(0))

    print('Test Acc: {:.4f}'.format(acc_record.avg))
    return acc_record


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Supervised training')
    parser.add_argument('--autoaugment', action='store_true', default=False,
                        help='Use autoaugment policy, only for CIFAR10 (Default: False)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='Input batch size for training (default: 64)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset name (default: CIFAR10)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='Number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--network', type=str, default='ResNet-18',
                        help='Network model (default: ResNet-18), choose between (ResNet-18, TempEns, RevNet-18)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--rotnet_dir', type=str, default='',
                        help='RotNet saved directory')
    parser.add_argument('--save_dir', type=str, default='./data/supervised/',
                        help='Directory to save models')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')

    args = parser.parse_args()
    args.name = 'supervised_%s_%s_seed%u' % (args.dataset.lower(), args.network.lower(), args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_train = dataset.GenericDataset(dataset_name=args.dataset, split='train', autoaugment=args.autoaugment)
    dataset_test = dataset.GenericDataset(dataset_name=args.dataset, split='test')

    dloader_train = dataset.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    dloader_test = dataset.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False)

    # Load model
    model = models.load_net(args.network, dataset_train.n_classes)

    # Use rotnet pretraining
    if args.rotnet_dir:
        # Load rotNet model, manually delete layers > 2
        state_dict_rotnet = torch.load(os.path.join(args.rotnet_dir, 'rotNet_%s_%s_lr_best.pth' % (args.dataset, args.network.lower())))
        for key in state_dict_rotnet.copy().keys():
            if 'fc' in key or 'layer3' in key or 'layer4' in key:
                del state_dict_rotnet[key]
        model.load_state_dict(state_dict_rotnet, strict=False)

        # Only finetune lower layers (>2)
        for name, param in model.named_parameters():
            if 'fc' not in name and 'layer3' not in name and 'layer4' not in name:
                param.requires_grad = False

    model = model.to(device)

    # Init optimizer and loss
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(args.epochs + 1):
        loss_record = train(epoch, model, device, dloader_train, optimizer, exp_lr_scheduler, criterion, args)
        acc_record = test(model, device, dloader_test, args)

        is_best = acc_record.avg > best_acc
        best_loss = max(acc_record.avg, best_acc)
        utils.save_checkpoint(model.state_dict(), is_best, args.save_dir, checkpoint=args.name + 'supervised_training_ckpt.pth', best_model=args.name + 'supervised_training_best.pth')


if __name__ == '__main__':
    main()
